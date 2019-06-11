import os
import json
import time
import tempfile
import webbrowser
from multiprocessing import Process

from .ground import create_app


def ground_with_gui(longforms, scores, grounding_map=None,
                    names_map=None, pos_labels=None):
    if grounding_map is None:
        grounding_map = _empty_grounding_map(longforms)
        names_map = {}
        pos_labels = []
    elif names_map is None:
        names_map = {}
    if pos_labels is None:
        pos_labels = []

    outpath = tempfile.mkdtemp()
    app = create_app(longforms, scores, grounding_map,
                     names_map, pos_labels, outpath)

    flask_server = Process(target=app.run)
    flask_server.start()
    webbrowser.open('http://localhost:5000/')
    while not os.path.exists(os.path.join(outpath, 'output.json')):
        time.sleep(1)
    flask_server.terminate()

    with open(os.path.join(outpath, 'output.json')) as f:
        output = json.load(f)
    return output['grounding_map'], output['names'], output['pos_labels']


def _empty_grounding_map(longforms):
    return {longform: None for longform in longforms}
