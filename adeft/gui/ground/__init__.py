from flask import Flask, session, render_template


def create_app(shortform, longforms, scores, grounding_map,
               names_map, pos_labels, outpath):
    # create and configure the app
    app = Flask(__name__)
    app.config.from_mapping(SECRET_KEY='dev',
                            OUTPATH=outpath,
                            SHORTFORM=shortform)

    @app.route('/')
    def initialize():
        groundings = [grounding_map[longform] for longform in longforms]
        names = [names_map[grounding] if grounding is not None else None
                 for grounding in groundings]
        labels = set(grounding for grounding in groundings
                     if grounding is not None)
        groundings = [grounding if grounding is not None else ''
                      for grounding in groundings]
        names = [name if name is not None else '' for name in names]
        session['groundings'] = groundings
        session['names'] = names
        session['labels'] = labels
        session['pos_labels'] = pos_labels

        return render_template('index.jinja2')
    return app
