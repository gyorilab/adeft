import os
import boto3

DEFT_PATH = os.path.dirname(os.path.abspath(__file__))
MODELS_PATH = DEFT_PATH + '/models'


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args,
                                                                 **kwargs)
        return cls._instances[cls]


class DeftDownloader(metaclass=Singleton):
    def __init__(self):
        self.downloaded_models = self._get_downloaded_models()
        s3 = boto3.resource('s3')
        self.s3_models = s3.Bucket('deft-models')

    def download_all(self, update=False):
        for s3_object in self.s3_models.objects.all():
            path, filename = os.path.split(s3_object.key)
            if filename and path and (update or path[:-1]
                                      not in self.downloaded_models):
                if not os.path.exists(os.path.join(MODELS_PATH, path)):
                    os.makedirs(os.path.join(MODELS_PATH, path))
                self.s3_models.download_file(s3_object.key,
                                             os.path.join(MODELS_PATH,
                                                          path, filename))
        self.downloaded_models = self._get_downloaded_models()

    def _get_downloaded_models(self):
        return set(model for model in os.listdir(MODELS_PATH)
                   if os.path.isdir(os.path.join(MODELS_PATH, model)))
