import os
import shutil

from adeft.locations import ADEFT_MODELS_PATH, TEST_RESOURCES_PATH
from adeft.download import download_models, get_available_models, \
    get_s3_models, setup_test_resource_folder


def test_get_s3_models():
    """Test function to check which models are available on S3"""
    models = get_s3_models()
    assert '__TEST' in models


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


def test_download_test_resources():
    """Test download of test resources from S3"""
    setup_test_resource_folder()
    test_model_path = os.path.join(TEST_RESOURCES_PATH, 'test_model', 'IR')
    assert os.path.exists(os.path.join(test_model_path, 'IR_model.gz'))
    assert os.path.exists(os.path.join(test_model_path, 'IR_names.json'))
    assert os.path.exists(os.path.join(test_model_path,
                                       'IR_grounding_dict.json'))
    assert os.path.exists(os.path.join(TEST_RESOURCES_PATH,
                                       'example_training_data.json'))
