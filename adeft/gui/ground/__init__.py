import os
import json
import logging

from flask import Flask, session, render_template

from adeft.gui.ground.ground import _convert_grounding_data


def create_app(longforms, scores,
               grounding_map, names_map, labels, pos_labels, outpath,
               verbose, test=False):
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

    app = Flask(__name__)
    # longforms, scores, and outpath will not change. These can be stored
    # in config variables
    app.config.from_mapping(SECRET_KEY='dev',
                            LONGFORMS=longforms,
                            SCORES=scores,
                            OUTPATH=outpath)

    # import grounding blueprint
    from adeft.gui.ground import ground

    @app.route('/')
    def initialize():
        """Load initial page for grounding app"""
        session['grounding_map'] = grounding_map
        session['names_map'] = names_map
        session['labels'] = labels
        session['pos_labels'] = pos_labels

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
