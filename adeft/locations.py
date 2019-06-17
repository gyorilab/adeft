import os
from adeft import __version__

ADEFT_PATH = os.path.join(os.path.expanduser('~'), '.adeft')
ADEFT_MODELS_PATH = os.path.join(ADEFT_PATH, 'models')
TEST_RESOURCES_PATH = os.path.join(ADEFT_PATH, 'test_resources')
S3_BUCKET_URL = os.path.join('http://adeft.s3.amazonaws.com', __version__)
