import os

DEFT_PATH = os.path.dirname(os.path.abspath(__file__))
MODELS_PATH = os.path.join(os.path.expanduser('~'),
                           '.deft_models')
S3_BUCKET_URL = 'http://deft-models.s3.amazonaws.com'
