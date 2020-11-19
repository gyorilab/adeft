"""Provides a graphical user interface to select the grounding for a
set of longforms.
"""
import os
import json
import time
import shutil
import logging
import tempfile
import webbrowser
from multiprocessing import Process


logger = logging.getLogger(__name__)


def ground_with_gui(longforms, scores, grounding_map=None,
                    names=None, pos_labels=None,
                    groundings_file=None,
                    verbose=False, port=5000, no_browser=False, test=False):
    """Opens grounding GUI in browser. Returns output upon user submission.

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
    names : Optional[dict]
        Dictionary map groundings from grounding_map to standardized names.
        This is ignored if grounding_map is set to None. Default: None
    pos_labels : Optional[list]
        List of groundings to be considered as positive labels.
        This is ignored if grounding_map is set to None. Default: None
    groundings_file : Optional[str]
        Path to a headerless csv file with three columns, one for
        namespace, identifier, and standard name respectively. Rows should
        be of the form
        MESH,D011839,"Radiation, Ionizing"
        with commas used as a separator and double quotes for escape.
        If such a table is supplied, users only need to supply the namespace
        along with only one of the standard name or identifier for each
        potential grounding with a row in the table. The missing entry will
        will be inferred. The path to such a file with groundings for the
        namespaces
        CHEBI, DOID, EFO, FPLX, GO, HGNC, HP, IP, MESH, NCIT, and UP
        can be found at `adeft.locations.GROUNDINGS_FILE_PATH. Default: None
    verbose : Optional[bool]
        When true, display logging from flask's werkzeug server.
        Default: False
    port : Optional[int]
        Port where flask is served. Defaults to flask's default.
        Default: 5000
    no_browser : Optional[bool]
        When True, do not automatically open GUI in browser
        Default: False
    test : Optional[bool]
        If True the Flask app is replaced with a mock version for testing.
        Default: False

    Returns
    -------
    grounding_map : dict
        Dictionary mapping longforms to groundings
    names : dict
        Dictionary mapping groundings to standardized names
    pos_labels : list
        List of groundings to be considered as positive labels
    """
    from .ground import create_app
    # Set initial groundings as empty if None are passed
    if grounding_map is None:
        grounding_map = {longform: '' for longform in longforms}
        names_map = {longform: '' for longform in longforms}
        pos_labels = []
    else:
        grounding_map = {longform: grounding_map[longform]
                         if longform in grounding_map
                         and grounding_map[longform]
                         and grounding_map[longform] != 'ungrounded'
                         else '' for longform in longforms}
        # Set initial names as empty if None are passed
        if names is None:
            names_map = {longform: '' for longform in longforms}
        else:
            if not set(names.keys()) <= set(grounding_map.values()):
                raise ValueError('keys in names_map must be subset of values'
                                 ' of grounding_map')
            names_map = {longform: names[grounding_map[longform]]
                         if longform in grounding_map and
                         grounding_map[longform] in names
                         and names[grounding_map[longform]] else ''
                         for longform in longforms}
    labels = sorted(set(grounding for _, grounding
                        in grounding_map.items() if grounding))
    if pos_labels is None:
        pos_labels = []
    else:
        pos_labels = [i for i, label in enumerate(labels)
                      if label in pos_labels]

    # Round scores for better presentation
    scores = [round(score, 2) for score in scores]

    # create temporary file for storing output
    outpath = tempfile.mkdtemp()
    # initialize flask app
    app = create_app(longforms, scores, grounding_map,
                     names_map, labels, pos_labels, groundings_file, outpath,
                     verbose, test=test)

    # Run flask server in new process
    flask_server = Process(target=_run_app, args=(app, port))
    flask_server.start()
    # Open app in browser unless a test is being run
    if not test and not no_browser:
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


def _run_app(app, port):
    return app.run(port=port)
