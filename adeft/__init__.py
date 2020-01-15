__version__ = '0.5.4'

from adeft.download import get_available_models

available_shortforms = {shortform: model
                        for shortform, model in get_available_models().items()
                        if shortform != '__TEST'}
