import os
import json

from flask import Blueprint, request, render_template, session, current_app

from adeft.gui.ground import _get_labels


bp = Blueprint('ground', __name__)


@bp.route('/ground_add', methods=['POST'])
def add_groundings():
    """Submit names and groundings for selected longforms"""
    # Get entered name and grounding from request. Strip out whitespace
    name = request.form['name'].strip()
    grounding = request.form['grounding'].strip()
    names, groundings = session['names'], session['groundings']
    # Find which longforms were selected to receive names and groundings
    # and add names and/or groundings for these longforms
    selected = request.form.getlist('select')
    for value in selected:
        index = int(value)-1
        if name:
            # set name if a value has been entered
            names[index] = name
        if grounding:
            # set grounding if a value has been entered
            groundings[index] = grounding
            # if a new grounding is entered, the positive label
            # numbers may have to shift to accomodate
            new_labels = _get_labels(groundings)
            # find index of new label
            new_label_number = new_labels.index(grounding)
            # add one to existing pos_label indices if they are
            # greater than or equal to the new label number
            # if a longform is given a different grounding and
            # previously had a positive label, the new grounding
            # will also have a positive label
            session['pos_labels'] = [i if i < new_label_number
                                     else i+1
                                     for i in session['pos_labels']]
            
    # set session variables so current names and groundings will persist
    session['names'], session['groundings'] = names, groundings
    labels = _get_labels(groundings)
    data = list(zip(current_app.config['LONGFORMS'],
                    current_app.config['SCORES'],
                    session['names'], session['groundings'], labels))
    return render_template('input.jinja2', data=data,
                           pos_labels=session['pos_labels'])


@bp.route('/ground_delete', methods=['POST'])
def delete_grounding():
    names, groundings = session['names'], session['groundings']
    pos_labels = session['pos_labels']
    starting_labels = _get_labels(groundings)
    for key in request.form:
        if key.startswith('delete.'):
            id_ = key.partition('.')[-1]
            index = int(id_) - 1
            label_number = starting_labels.index(groundings[index])
            names[index] = groundings[index] = ''
            break
    session['names'], session['groundings'] = names, groundings
    # Remove deleted grounding from pos_labels if necessary
    try:
        pos_labels = pos_labels.remove(label_number)
    except ValueError:
        pass
    # Since a label has been deleted, all labels with larger
    # label_number will have their number shift down by one
    pos_labels = [i if i < label_number else i-1
                  for i in session['pos_labels']]
    session['pos_labels'] = pos_labels
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
            label_number = int(key.partition('.')[-1]) - 1
            pos_labels = sorted(list(set(pos_labels) ^ set([label_number])))
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
