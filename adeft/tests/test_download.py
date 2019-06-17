import os
import shutil

from adeft.locations import ADEFT_MODELS_PATH
from adeft.download import download_models, get_available_models, \
    get_s3_models


def test_get_s3_models():
    """Test function to check which models are available on S3"""
    models = get_s3_models()
    assert 'IR' in models


def test_download_models():
    """Test downloading of models from S3"""
    # Remove test models folder if it exists
    test_model_folder = os.path.join(ADEFT_MODELS_PATH, '__TEST')
    if os.path.isdir(test_model_folder):
        shutil.rmtree(test_model_folder)
    assert '__TEST' not in get_available_models()

    # Download models
    download_models(models=['__TEST'])
    assert '__TEST' in get_available_models()
