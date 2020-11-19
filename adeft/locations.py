"""
Contains paths to locations on user's system where models and resources are
to be stored. These all live in adeft's home folder which defaults to the
hidden directory ".adeft" in the user's home directory but which can be
specified by setting the environment variable ADEFT_HOME in the user's profile.
"""

import os
from adeft import __version__


ADEFT_HOME = os.environ.get('ADEFT_HOME')
if ADEFT_HOME is None:
    ADEFT_HOME = os.path.join(os.path.expanduser('~'), '.adeft')

ADEFT_PATH = os.path.join(ADEFT_HOME, __version__)
ADEFT_MODELS_PATH = os.path.join(ADEFT_PATH, 'models')
RESOURCES_PATH = os.path.join(ADEFT_PATH, 'resources')
GROUNDINGS_FILE_PATH = os.path.join(RESOURCES_PATH, 'groundings.csv')
TEST_RESOURCES_PATH = os.path.join(ADEFT_PATH, 'test_resources')
S3_BUCKET_URL = os.path.join('http://adeft.s3.amazonaws.com', __version__)
