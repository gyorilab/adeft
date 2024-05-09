import os

from adeft.locations import ADEFT_PATH
from adeft.download import setup_models_folder, setup_resources_folder, \
    setup_test_resource_folder


# Create adeft data folder if it does not already exist
if not os.path.exists(ADEFT_PATH):
    os.makedirs(ADEFT_PATH)

setup_models_folder()
setup_resources_folder()
setup_test_resource_folder()
