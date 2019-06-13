import os
import logging


from flask import Flask, session, render_template


def create_app(groundings_list, names_list, longforms_list, pos_labels,
               outpath, verbose, port):
    """Create and configure model fixing app.

    Takes same arguments as adeft.gui.fix_with_gui
    """
    if not verbose:
        # disable all logging
        werkzeug_logger = logging.getLogger('werkzeug')
        werkzeug_logger.setLevel(logging.ERROR)
        os.environ['WERKZEUG_RUN_MAIN'] = 'true'

    app = Flask(__name__)

    app.config.from_mapping(SECRET_KEY='dev',
                            GROUNDINGS=groundings_list,
                            LONGFORMS=longforms_list,
                            OUTPATH=outpath)

    from adeft.gui.fix import fix

    @app.route('/')
    def initialize():
        session['groundings_transition'] = {grounding: grounding
                                            for grounding in
                                            app.config['GROUNDINGS']}
        session['names_transition'] = {grounding: name for grounding, name
                                       in zip(app.config['GROUNDINGS'],
                                              names_list)}
        session['pos_labels'] = pos_labels
        data = zip(app.config['GROUNDINGS'],
                   names_list,
                   app.config['LONGFORMS'],
                   pos_labels)
        return render_template('fix.jinja2', data=data)
    app.register_blueprint(fix.bp)
    return app
