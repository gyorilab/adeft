import os
import wget
import requests

from deft.locations import MODELS_PATH, S3_BUCKET_URL


def download_all(update=False):
    s3_models = _get_s3_models()
    downloaded_models = _get_downloaded_models()
    for model in s3_models:
        if not update and model in downloaded_models:
            continue
        if not os.path.exists(os.path.join(MODELS_PATH, model)):
            os.makedirs(os.path.join(MODELS_PATH, model))
        for resource in (model.lower() + '_grounding_map.json',
                         model.lower() + '_model.gz'):
            resource_path = os.path.join(MODELS_PATH, model, resource)
            try:
                os.remove(resource_path)
            except OSError:
                pass
            print(os.path.join(S3_BUCKET_URL, model, resource))
            wget.download(url=os.path.join(S3_BUCKET_URL, model, resource),
                          out=resource_path)


def _get_downloaded_models():
    return set(model for model in os.listdir(MODELS_PATH)
               if os.path.isdir(os.path.join(MODELS_PATH, model)))


def _get_s3_models():
    result = requests.get(S3_BUCKET_URL + '/s3_models.json')
    return result.json()
