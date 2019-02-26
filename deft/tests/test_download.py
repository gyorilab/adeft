import os
import shutil

from deft.locations import MODELS_PATH
from deft.download import download_models, get_downloaded_models, \
    get_s3_models


def test_get_s3_models():
    """Test function to check which models are available on S3"""
    models = get_s3_models()
    assert 'TEST' in models


def test_download_models():
    """Test downloading of models from S3"""
    # Remove test models folder if it exists
    test_model_folder = os.path.join(MODELS_PATH, 'TEST')
    if os.path.isdir(test_model_folder):
        shutil.rmtree(test_model_folder)
    assert 'TEST' not in get_downloaded_models()

    # Download models
    download_models(models=['TEST'])
    assert 'TEST' in get_downloaded_models()
