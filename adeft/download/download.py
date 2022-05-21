import adeft.locations as loc
import boto3
import botocore
import os
import gzip
import json
import shutil
import logging

from botocore import UNSIGNED
from botocore.config import Config


logger = logging.getLogger(__file__)


def setup_models_folder():
    """Create models folder if it does not exist and download models
    """
    if os.path.isdir(loc.ADEFT_MODELS_PATH):
        shutil.rmtree(loc.ADEFT_MODELS_PATH)
    os.mkdir(loc.ADEFT_MODELS_PATH)
    download_models()
    return


def download_models(models=None):
    """Download models from S3

    Models are downloaded and placed into a models directory in the users
    home directory. Each model contains a serialized AdeftClassifier,
    a dictionary mapping shortforms to dictionaries mapping longform texts to
    groundings, and a list of canonical names for each grounding.
    Within the models directory, models are stored in subdirectories named
    after the shortform they disambiguate with escape characters used to
    handle characters that cannot be used in filenames and to distinguish
    upper and lower case for compatibility with case insensitive file systems.

    Parameters
    --------
    models : Optional[iterable of str]
        List of models to be downloaded. Allows user to select specific
        models to download. If this option is set, update will be treated
        as True regardless of how it was set. These should be considered
        as mutually exclusive parameters.
    """
    s3_models = set(get_s3_models().values())
    if models is None:
        models = s3_models
    else:
        models = set(models) & set(s3_models)
    for model in models:
        # create model directory if it does not currently exist
        if not os.path.exists(os.path.join(loc.ADEFT_MODELS_PATH, model)):
            os.makedirs(os.path.join(loc.ADEFT_MODELS_PATH, model))
        for resource in (
                model + '_grounding_dict.json',
                model + '_names.json',
                model + '_model.gz'
        ):
            resource_path = os.path.join(
                loc.ADEFT_MODELS_PATH, model, resource
            )
            # if resource already exists, remove it. Ensures that
            # models path stays in a consistent state.
            _remove_if_exists(resource_path)
            download_adeft_object(
                'Models', model, resource, outpath=resource_path
            )


def setup_resources_folder():
    """Make resources folder and download resources

    Replaces content in existing resources folder if it already exists
    """
    if os.path.isdir(loc.RESOURCES_PATH):
        shutil.rmtree(loc.RESOURCES_PATH)
    os.mkdir(loc.RESOURCES_PATH)
    download_resources()


def download_resources():
    resources = ['groundings.csv']
    for resource in resources:
        resource_path = os.path.join(loc.RESOURCES_PATH, f'{resource}.gz')
        _remove_if_exists(resource_path)
        download_adeft_object(
            'Resources', f'{resource}.gz',
            outpath=resource_path
        )
        with gzip.open(resource_path, 'rb') as f_in:
            with open(
                    os.path.join(loc.RESOURCES_PATH, resource), 'wb'
            ) as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(resource_path)


def setup_test_resource_folder():
    """Make test resource folders and download content

    Replaces content in existing test_resource_folders if they already
    exist.
    """
    if os.path.isdir(loc.TEST_RESOURCES_PATH):
        shutil.rmtree(loc.TEST_RESOURCES_PATH)
    os.mkdir(loc.TEST_RESOURCES_PATH)
    os.mkdir(os.path.join(loc.TEST_RESOURCES_PATH, 'test_model'))
    os.mkdir(os.path.join(loc.TEST_RESOURCES_PATH, 'scratch'))
    os.mkdir(os.path.join(loc.TEST_RESOURCES_PATH, 'test_model', 'IR'))
    download_test_resources()
    return


def download_test_resources():
    """Download files necessary to run tests

    Downloads a test disambiguator and a set of example training data and
    places them in the test_resources folder of the .adeft directory. This
    function will error if the necessary directories do not exist. If they do
    not already exist they will be created when running
    python -m adeft.download
    """
    test_model_path = os.path.join(loc.TEST_RESOURCES_PATH, 'test_model', 'IR')
    if not os.path.exists(test_model_path):
        os.mkdir(test_model_path)
    for resource in ('IR_grounding_dict.json', 'IR_names.json', 'IR_model.gz'):
        if not os.path.exists(os.path.join(test_model_path, resource)):
            download_adeft_object(
                'Test', 'IR', resource,
                outpath=os.path.join(test_model_path, resource),
            )
    if not os.path.exists(os.path.join(loc.TEST_RESOURCES_PATH,
                                       'example_training_data.json')):
        download_adeft_object(
            'Test', 'example_training_data.json',
            outpath=os.path.join(
                loc.TEST_RESOURCES_PATH,
                'example_training_data.json'
            ),
        )


def get_available_models(path=loc.ADEFT_MODELS_PATH):
    """Returns set of all models currently in models folder"""
    if not os.path.exists(path):
        return {}
    output = {}
    for model in os.listdir(path):
        model_path = os.path.join(path, model)
        if os.path.isdir(model_path) and model != '__pycache__':
            try:
                grounding_file = '%s_grounding_dict.json' % model
                with open(os.path.join(model_path, grounding_file), 'r') as f:
                    grounding_dict = json.load(f)
                for key, value in grounding_dict.items():
                    if key in output:
                        logger.warning('Shortform %s has multiple adeft models'
                                       'This may lead to unexpected behavior'
                                       % key)
                    else:
                        output[key] = model
            except FileNotFoundError:
                continue
    return output


def get_s3_models():
    """Returns set of all models currently available on s3"""
    try:
        response = _read_s3_content(
            bucket=loc.S3_BUCKET,
            key=_get_s3_key('Models', 's3_models.json'),
        )
    except botocore.exceptions.ClientError:
        logger.warning("Online Adeft models not available.")
        return
    return json.loads(response["Body"].read())


def download_adeft_object(*args, outpath):
    logger.info(f"Downloading {'/'.join(args)}")
    return _anonymous_s3_download(loc.S3_BUCKET, _get_s3_key(*args), outpath)


def _get_s3_key(*args):
    return '/'.join((loc.S3_KEY_PREFIX, ) + args)


def _read_s3_content(bucket, key):
    config = Config(signature_version=UNSIGNED)
    s3 = boto3.client('s3', config=config, region_name='us-east-1')
    response = s3.get_object(Bucket=bucket, Key=key)
    return response


def _anonymous_s3_download(bucket, key, outpath):
    config = Config(signature_version=UNSIGNED)
    s3 = boto3.client('s3', config=config, region_name='us-east-1')
    s3.download_file(bucket, key, outpath)


def _remove_if_exists(path):
    """Remove file if it exists, otherwise do nothing

    Parameters
    ----------
    path : str
        file to attempt to remove
    """
    try:
        os.remove(path)
    except OSError:
        pass
