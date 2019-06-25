import os
import json
import wget
import shutil
import logging
import requests


from adeft.locations import ADEFT_MODELS_PATH, S3_BUCKET_URL, \
    TEST_RESOURCES_PATH


logger = logging.getLogger(__file__)


def download_models(update=False, models=None):
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
    ---------
    update : Optional[bool]
        If True, replace all existing models with versions on S3
        otherwise only download models that aren't currently available.
        Default: False

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
        models = set(models) & s3_models
        update = True

    downloaded_models = get_available_models()
    for model in models:
        # if update is False do not download model
        if not update and model in downloaded_models:
            continue
        # create model directory if it does not currently exist
        if not os.path.exists(os.path.join(ADEFT_MODELS_PATH, model)):
            os.makedirs(os.path.join(ADEFT_MODELS_PATH, model))
        for resource in (model + '_grounding_dict.json',
                         model + '_names.json',
                         model + '_model.gz'):
            resource_path = os.path.join(ADEFT_MODELS_PATH, model, resource)
            # if resource already exists, remove it since wget will not
            # overwrite existing files, choosing a new name instead
            _remove_if_exists(resource_path)
            wget.download(url=os.path.join(S3_BUCKET_URL, 'Models',
                                           model, resource),
                          out=resource_path)


def setup_test_resource_folders():
    """Make test resource folders and download content

    Replaces content in existing test_resource_folders if they already
    exist.
    """
    if os.path.isdir(TEST_RESOURCES_PATH):
        shutil.rmtree(TEST_RESOURCES_PATH)
    os.mkdir(TEST_RESOURCES_PATH)
    os.mkdir(os.path.join(TEST_RESOURCES_PATH, 'test_model'))
    os.mkdir(os.path.join(TEST_RESOURCES_PATH, 'scratch'))
    os.mkdir(os.path.join(TEST_RESOURCES_PATH, 'test_model', 'IR'))
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
    test_model_path = os.path.join(TEST_RESOURCES_PATH, 'test_model', 'IR')
    if not os.path.exists(test_model_path):
        os.mkdir(test_model_path)
    for resource in ('IR_grounding_dict.json', 'IR_names.json', 'IR_model.gz'):
        if not os.path.exists(os.path.join(test_model_path, resource)):
            wget.download(url=os.path.join(S3_BUCKET_URL, 'Test', 'IR',
                                           resource),
                          out=os.path.join(test_model_path, resource))
    if not os.path.exists(os.path.join(TEST_RESOURCES_PATH,
                                       'example_training_data.json')):
        wget.download(url=os.path.join(S3_BUCKET_URL, 'Test',
                                       'example_training_data.json'),
                      out=os.path.join(TEST_RESOURCES_PATH,
                                       'example_training_data.json'))


def get_available_models(path=ADEFT_MODELS_PATH):
    """Returns set of all models currently in models folder"""
    if not os.path.exists(path):
        return {}
    output = {}
    for model in os.listdir(path):
        model_path = os.path.join(path, model)
        if os.path.isdir(model_path) and model != '__pycache__':
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
    return output


def get_s3_models():
    """Returns set of all models currently available on s3"""
    result = requests.get(os.path.join(S3_BUCKET_URL, 'Models',
                                       's3_models.json'))
    try:
        output = result.json()
        assert isinstance(output, dict)
    except json.JSONDecodeError or AssertionError:
        output = {}
        logger.warning('Online deft models are currently unavailable')
    return output


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
