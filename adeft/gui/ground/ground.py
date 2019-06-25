import os
import json

from flask import Blueprint, request, render_template, session, current_app

bp = Blueprint('ground', __name__)


@bp.route('/ground_add', methods=['POST'])
def add_groundings():
    """Submit names and groundings for selected longforms"""
    # Get entered name and grounding from request. Strip out whitespace
    name = request.form['name'].strip()
    grounding = request.form['grounding'].strip()
    selected = [int(i) for i in request.form.getlist('select')]
    state = GroundingState(current_app.config['LONGFORMS'],
                           session['grounding_map'],
                           session['names_map'],
                           session['labels'],
                           session['pos_labels'])

    state.add(name, grounding, selected)
    (session['grounding_map'],
     session['names_map'],
     session['labels'],
     session['pos_labels']) = state.dump()

    return render_template('input.jinja2')


@bp.route('/ground_delete', methods=['POST'])
def delete_grounding():
    for key in request.form:
        if key.startswith('delete.'):
            row_number = int(key.partition('.')[-1])
            break

    state = GroundingState(current_app.config['LONGFORMS'],
                           session['grounding_map'],
                           session['names_map'],
                           session['labels'],
                           session['pos_labels'])
    state.delete(row_number)

    (session['grounding_map'],
     session['names_map'],
     session['labels'],
     session['pos_labels']) = state.dump()

    return render_template('input.jinja2')


@bp.route('/ground_pos_label', methods=['POST'])
def add_positive():
    for key in request.form:
        if key.startswith('pos-label.'):
            label_number = int(key.partition('.')[-1])
            break

    state = GroundingState(current_app.config['LONGFORMS'],
                           session['grounding_map'],
                           session['names_map'],
                           session['labels'],
                           session['pos_labels'])
    state.toggle_positive(label_number)

    (session['grounding_map'],
     session['names_map'],
     session['labels'],
     session['pos_labels']) = state.dump()

    return render_template('input.jinja2')


@bp.route('/ground_generate', methods=['POST'])
def generate_grounding_map():
    output = _convert_grounding_data(session['grounding_map'],
                                     session['names_map'],
                                     session['labels'],
                                     session['pos_labels'])

    outpath = current_app.config['OUTPATH']
    with open(os.path.join(outpath, 'output.json'), 'w') as f:
        json.dump(output, f)

    return 'Groundings Submitted. You may close this browser tab.'


def _convert_grounding_data(grounding_map, names_map, labels, pos_labels):
    """Map apps representation of grounding data to external representation"""
    grounding_map = {longform: grounding if grounding
                     else 'ungrounded'
                     for longform, grounding in
                     grounding_map.items()}
    names = {grounding: names_map[longform]
             for longform, grounding in
             grounding_map.items()
             if grounding != 'ungrounded' and names_map[longform]}
    pos_labels = [label for i, label in enumerate(labels)
                  if i in pos_labels]
    output = {'grounding_map': grounding_map,
              'names': names,
              'pos_labels': pos_labels}
    return output


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
