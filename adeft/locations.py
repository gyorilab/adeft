"""
Contains paths to locations on user's system where models and resources are
to be stored. These all live in adeft's home folder which defaults to
a directory "adeft" in a location determined by Python's `appdirs` package.
An alternative location can be specified by setting the environment variable
ADEFT_HOME in the user's profile.
"""

import os
from adeft import __version__
from appdirs import user_data_dir

ADEFT_HOME = os.environ.get('ADEFT_HOME')
if ADEFT_HOME is None:
    ADEFT_HOME = os.path.join(user_data_dir(), 'adeft')

ADEFT_PATH = os.path.join(ADEFT_HOME, __version__)
ADEFT_MODELS_PATH = os.path.join(ADEFT_PATH, 'models')
RESOURCES_PATH = os.path.join(ADEFT_PATH, 'resources')
GROUNDINGS_FILE_PATH = os.path.join(RESOURCES_PATH, 'groundings.csv')
TEST_RESOURCES_PATH = os.path.join(ADEFT_PATH, 'test_resources')
S3_BUCKET = "adeft"
BUCKET_REGION = "us-east-1"
S3_KEY_PREFIX = __version__
