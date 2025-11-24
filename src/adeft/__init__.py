__version__ = '1.0.0-dev'

from adeft.download import get_available_models

available_shortforms = {shortform: model
                        for shortform, model in get_available_models().items()
                        if shortform != '__TEST'}
