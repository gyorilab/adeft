import os
import wget
import requests

from deft.locations import MODELS_PATH, S3_BUCKET_URL


def download_models(update=False):
    """Download models from S3

    Models are downloaded and placed int deft/models. Each model contains
    a serialized deft classifier and a grounding map and is stored in a
    directory named after the shortform it disambiguates.

    Parameters
    ---------
    update : Optional[bool]
        If True, replace all existing models with versions on S3
        otherwise only download models that aren't currently available.
        Default: True
    """
    s3_models = _get_s3_models()
    downloaded_models = get_downloaded_models()
    for model in s3_models:
        # if update is False do not download model
        if not update and model in downloaded_models:
            continue
        # create model directory if it does not currently exist
        if not os.path.exists(os.path.join(MODELS_PATH, model)):
            os.makedirs(os.path.join(MODELS_PATH, model))
        for resource in (model.lower() + '_grounding_map.json',
                         model.lower() + '_names.json',
                         model.lower() + '_model.gz'):

            resource_path = os.path.join(MODELS_PATH, model, resource)
            # if resource already exists, remove it since wget will not
            # overwrite existing files, choosing a new name instead
            try:
                os.remove(resource_path)
            except OSError:
                pass
            wget.download(url=os.path.join(S3_BUCKET_URL, model, resource),
                          out=resource_path)


def get_downloaded_models():
    """Returns set of all models currently in models folder"""
    return [model for model in os.listdir(MODELS_PATH)
            if os.path.isdir(os.path.join(MODELS_PATH, model))
            and model != '__pycache__']


def _get_s3_models():
    """Returns set of all models currently available on s3"""
    result = requests.get(S3_BUCKET_URL + '/s3_models.json')
    return result.json()
