"""
Allows models to be downloaded from the command line with

.. code-block:: bash

    python -m adeft.download

Use

.. code-block:: bash

    python -m adeft.download --update

to update existing models if models have changed on S3
"""
from .download import download_models, get_available_models, get_s3_models, \
    download_test_resources, setup_test_resource_folders
