import os
import json
import time
import shutil
import logging
import tempfile
import webbrowser
from multiprocessing import Process

from .ground import create_app


logger = logging.getLogger(__name__)


def ground_with_gui(longforms, scores, grounding_map=None,
                    names_map=None, pos_labels=None, verbose=False, port=5000):
    """Opens grounding gui in browser. Returns output upon user submission

    Parameters
    ----------
    longforms : list of str
        List of longforms to ground

    scores : list of float
        List of scores associated to each longform

    grounding_map : Optional[dict]
        Dictionary mapping longforms to groundings. Intended for initial
        groundings that must be manually reviewed by user, such as those
        created by an imperfect grounding function. Default: None

    names_map : Optional[dict]
        Dictionary map groundings from grounding_map to standardized names.
        This is ignored if grounding_map is set to None. Default: None

    pos_labels : Optional[list]
        List of groundings to be considered as positive labels.
        This is ignored if grounding_map is set to None. Default: None

    verbose : Optional[bool]
        When true, display logging from flask's werkzeug server.
        Default: False

    port : Optional[int]
        Port where flask is served. Defaults to flask's default.
        Default: 5000

    Returns
    -------
    grounding_map : dict
        Dictionary mapping longforms to groundings
    names : dict
        Dictionary mapping groundings to standardized names
    pos_labels : list
        List of groundings to be considered as positive labels
    """
    # Set initial groundings etc. to empty if none are passed
    if grounding_map is None:
        grounding_map = _empty_grounding_map(longforms)
        names_map = {}
        pos_labels = []
    elif names_map is None:
        names_map = {}
    if pos_labels is None:
        pos_labels = []

    # create temporary file for storing output
    outpath = tempfile.mkdtemp()
    # initialize flask app
    app = create_app(longforms, scores, grounding_map,
                     names_map, pos_labels, outpath, verbose, port)
    # Run flask server in new process
    flask_server = Process(target=app.run)
    flask_server.start()
    # Open app in browser
    webbrowser.open('http://localhost:%d/' % port)
    # Poll until user submits groundings. Checks if output file exists
    while not os.path.exists(os.path.join(outpath, 'output.json')):
        time.sleep(1)
    # Stop server
    flask_server.terminate()
    # Get output from temporary file
    with open(os.path.join(outpath, 'output.json')) as f:
        output = json.load(f)
    # Clean up temporary file
    try:
        shutil.rmtree(outpath)
    except Exception:
        logger.warning('Could not clean up temporary file %s' % outpath)
    grounding_map = output['grounding_map']
    names = output['names']
    pos_labels = output['pos_labels']
    return grounding_map, names, pos_labels


def _empty_grounding_map(longforms):
    """Returns empty grounding map."""
    return {longform: None for longform in longforms}
