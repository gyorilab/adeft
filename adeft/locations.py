"""
Contains paths to locations on user's system where models and resources are
to be stored. These all live in adeft's home folder which defaults to the
hidden directory ".adeft" in the user's home directory but which can be
specified by setting the environment variable ADEFT_HOME in the user's profile.
"""

import os
import pystow
from adeft import __version__

# If the adeft resource directory does not exist, try to create it using PyStow
# Can be specified with ADEFT_HOME environment variable, otherwise defaults
# to $HOME/.data/adeft/<__version__>. The location of $HOME can be overridden with
# the PYSTOW_HOME environment variable
ADEFT_HOME = pystow.join('adeft').as_posix()

ADEFT_PATH = os.path.join(ADEFT_HOME, __version__)
ADEFT_MODELS_PATH = os.path.join(ADEFT_PATH, 'models')
RESOURCES_PATH = os.path.join(ADEFT_PATH, 'resources')
GROUNDINGS_FILE_PATH = os.path.join(RESOURCES_PATH, 'groundings.csv')
TEST_RESOURCES_PATH = os.path.join(ADEFT_PATH, 'test_resources')
S3_BUCKET_URL = os.path.join('http://adeft.s3.amazonaws.com', __version__)
