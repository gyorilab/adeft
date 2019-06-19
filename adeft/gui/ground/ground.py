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


class GroundingState(object):
    """Stores state of current page

    This class is initialized from session variables and contains
    methods for transforming state in response to user input.
    """
    def __init__(self, longforms, grounding_map, names_map,
                 labels, pos_labels):
        self.longforms = longforms
        self.grounding_map = grounding_map
        self.names_map = names_map
        self.labels = labels
        self.pos_labels = pos_labels

    def dump(self):
        """Returns state needed to reload page."""
        return self.grounding_map, self.names_map, self.labels, self.pos_labels

    def add(self, name, grounding, row_numbers):
        """Add new names and groundings"""
        positive_groundings = set(self.labels[index]
                                  for index in self.pos_labels)
        for i in row_numbers:
            if grounding:
                self.grounding_map[self.longforms[i]] = grounding
            if name:
                self.names_map[self.longforms[i]] = name

        labels = sorted(set(grounding for _, grounding in
                            self.grounding_map.items()))
        pos_labels = [i for i, label in enumerate(labels)
                      if label in positive_groundings]
        self.labels = labels
        self.pos_labels = pos_labels

    def delete(self, row_number):
        """Delete names and groundings"""
        longform = self.longforms[row_number]
        grounding = self.grounding_map[longform]
        self.grounding_map[longform] = self.names_map[longform] = ''
        if grounding not in self.grounding_map.values():
            label_number = self.labels.index(grounding)
            self.pos_labels = [i if i < label_number else i - 1
                               for i in self.pos_labels if i != label_number]
            self.labels = (self.labels[:label_number] +
                           self.labels[label_number+1:])

    def toggle_positive(self, label_number):
        """Toggle labels as positive or negative"""
        pos_labels = sorted(set(self.pos_labels) ^ set([label_number]))
        self.pos_labels = pos_labels
