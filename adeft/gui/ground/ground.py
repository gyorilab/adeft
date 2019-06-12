import os
import json

from flask import Blueprint, request, render_template, session, current_app

from adeft.gui.ground import _get_labels


bp = Blueprint('ground', __name__)


@bp.route('/ground_add', methods=['POST'])
def add_groundings():
    name = request.form['name'].strip()
    grounding = request.form['grounding'].strip()
    names, groundings = session['names'], session['groundings']
    if name and grounding:
        selected = request.form.getlist('select')
        for value in selected:
            index = int(value)-1
            names[index] = name
            groundings[index] = grounding
    session['names'], session['groundings'] = names, groundings
    session['pos_labels'] = list(set(session['pos_labels']) & set(groundings))

    labels = _get_labels(groundings)
    data = list(zip(current_app.config['LONGFORMS'],
                    current_app.config['SCORES'],
                    session['names'], session['groundings'], labels))
    return render_template('input.jinja2', data=data,
                           pos_labels=session['pos_labels'])


@bp.route('/ground_delete', methods=['POST'])
def delete_grounding():
    names, groundings = session['names'], session['groundings']
    for key in request.form:
        if key.startswith('delete.'):
            id_ = key.partition('.')[-1]
            index = int(id_) - 1
            names[index] = groundings[index] = ''
            break
    session['names'], session['groundings'] = names, groundings
    session['pos_labels'] = list(set(session['pos_labels']) & set(groundings))
    labels = _get_labels(groundings)
    data = list(zip(current_app.config['LONGFORMS'],
                    current_app.config['SCORES'],
                    session['names'], session['groundings'], labels))
    return render_template('input.jinja2', data=data,
                           pos_labels=session['pos_labels'])


@bp.route('/ground_pos_label', methods=['POST'])
def add_positive():
    pos_labels = session['pos_labels']
    for key in request.form:
        if key.startswith('pos-label.'):
            label = key.partition('.')[-1]
            pos_labels = list(set(pos_labels) ^ set([label]))
            session['pos_labels'] = pos_labels
            break
    labels = _get_labels(session['groundings'])
    data = list(zip(current_app.config['LONGFORMS'],
                    current_app.config['SCORES'],
                    session['names'], session['groundings'], labels))
    return render_template('input.jinja2', data=data,
                           pos_labels=session['pos_labels'])


@bp.route('/ground_generate', methods=['POST'])
def generate_grounding_map():
    longforms = current_app.config['LONGFORMS']
    names = session['names']
    groundings = session['groundings']
    pos_labels = session['pos_labels']
    grounding_map = {longform: grounding if grounding else 'ungrounded'
                     for longform, grounding in zip(longforms, groundings)}
    names_map = {grounding: name for grounding, name in zip(groundings,
                                                            names)
                 if grounding and name}

    outpath = current_app.config['OUTPATH']
    output = {'grounding_map': grounding_map,
              'names': names_map,
              'pos_labels': pos_labels}
    with open(os.path.join(outpath, 'output.json'), 'w') as f:
        json.dump(output, f)

    return "Groundings Submitted."
