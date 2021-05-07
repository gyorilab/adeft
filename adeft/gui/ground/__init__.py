import os
import csv
import json
import logging
from collections import defaultdict

from flask import Flask, session, render_template

from adeft.locations import ADEFT_PATH
from adeft.gui.ground.ground import _convert_grounding_data


def create_app(longforms, scores,
               grounding_map, names_map, labels, pos_labels, identifiers_file,
               outpath, verbose, test=False):
    """Create and configure grounding assistant app.

    Takes same arguments as adeft.gui.ground_with_gui.
    """
    if test:
        return MockApp(outpath, grounding_map, names_map, labels, pos_labels)

    if not verbose:
        # disable all logging
        werkzeug_logger = logging.getLogger('werkzeug')
        werkzeug_logger.setLevel(logging.ERROR)
        os.environ['WERKZEUG_RUN_MAIN'] = 'true'

    identifiers_dict = defaultdict(lambda: {'name_id': {}, 'id_name': {}})
    if identifiers_file is not None:
        with open(os.path.realpath(os.path.expanduser(identifiers_file)),
                  newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='"')
            for namespace, identifier, name in reader:
                identifiers_dict[namespace]['name_id'][name] = identifier
                identifiers_dict[namespace]['id_name'][identifier] = name
    identifiers_dict = dict(identifiers_dict)

    app = Flask(__name__)
    # longforms, scores, and outpath will not change. These can be stored
    # in config variables
    app.config.from_mapping(SECRET_KEY='dev',
                            LONGFORMS=longforms,
                            SCORES=scores,
                            OUTPATH=outpath,
                            IDENTIFIERS_DICT=identifiers_dict,
                            SESSION_TYPE='filesystem',
                            SESSION_FILE_DIR=os.path.join(ADEFT_PATH,
                                                          'flask_session'),
                            SESSION_COOKIE_SAMESITE='Strict')

    # import grounding blueprint
    from adeft.gui.ground import ground

    @app.route('/')
    def initialize():
        """Load initial page for grounding app"""
        session['grounding_map'] = grounding_map
        session['names_map'] = names_map
        session['labels'] = labels
        session['pos_labels'] = pos_labels
        session['sorted_order'] = list(range(len(longforms)))
        session['matches_list'] = ['unknown']*len(longforms)

        return render_template('input.jinja2')

    app.register_blueprint(ground.bp)
    return app


class MockApp(object):
    def __init__(self, outpath, grounding_map, names_map,
                 labels, pos_labels):
        self.outpath = outpath
        self.grounding_map = grounding_map
        self.names_map = names_map
        self.pos_labels = pos_labels
        self.labels = labels

    def run(self, port=None):
        output = _convert_grounding_data(self.grounding_map,
                                         self.names_map,
                                         self.labels,
                                         self.pos_labels)
        with open(os.path.join(self.outpath, 'output.json'), 'w') as f:
            json.dump(output, f)
