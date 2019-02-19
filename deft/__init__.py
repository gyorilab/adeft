__version__ = '0.0.0'

from .download import _get_downloaded_models

available_shortforms = list(_get_downloaded_models())
