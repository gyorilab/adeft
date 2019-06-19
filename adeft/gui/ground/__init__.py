import os
import logging

from flask import Flask, session, render_template


def create_app(longforms, scores,
               grounding_map, names_map, pos_labels, outpath,
               verbose, port):
    """Create and configure grounding assistant app.

    Takes same arguments as adeft.gui.ground_with_gui.
    """
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
        labels = sorted(grounding for _, grounding in grounding_map.items())

        session['grounding_map'] = grounding_map
        session['names_map'] = names_map
        session['labels'] = labels
        session['pos_labels'] = pos_labels

        return render_template('input.jinja2')

    app.register_blueprint(ground.bp)
    return app
