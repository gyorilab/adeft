import os
import json
from numpy import argsort


from flask import Blueprint, request, render_template, session, current_app

here = os.path.dirname(os.path.realpath(__file__))
bp = Blueprint('ground', __name__)


@bp.route('/ground_add', methods=['POST'])
def add_groundings():
    """Submit names and groundings for selected longforms"""
    # Get entered name and grounding from request. Strip out whitespace
    name = request.form['name'].strip()
    namespace = request.form['namespace'].strip()
    identifier = request.form['identifier'].strip()
    identifiers_dict = current_app.config['IDENTIFIERS_DICT']
    matches = 'unknown'
    special_groundings = ['ignore']
    if namespace and namespace in identifiers_dict:
        if identifier and not name:
            name = identifiers_dict[namespace]['id_name'].get(identifier)
            if name is not None:
                matches = 'match'
            else:
                name = ''
        elif name and not identifier:
            identifier = identifiers_dict[namespace]['name_id'].get(name)
            if identifier is not None:
                matches = 'match'
            else:
                identifier = ''
        elif name and identifier:
            identifier2 = identifiers_dict[namespace]['name_id'].get(name)
            if identifier2:
                matches = 'match' if identifier == identifier2 else 'mismatch'
    if namespace and identifier:
        grounding = ':'.join([namespace, identifier])
    elif identifier:
        # Blank implicitly means ungrounded
        grounding = identifier if identifier != 'ungrounded' else ''
    else:
        grounding = ''
    if grounding in special_groundings:
        matches = 'special'
    selected = [int(i) for i in request.form.getlist('select')]
    state = GroundingState(current_app.config['LONGFORMS'],
                           session['grounding_map'],
                           session['names_map'],
                           session['labels'],
                           session['pos_labels'],
                           session['matches_list'])

    state.add(name, grounding, selected, matches)
    (session['grounding_map'],
     session['names_map'],
     session['labels'],
     session['pos_labels'],
     session['matches_list']) = state.dump()

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
                           session['pos_labels'],
                           session['matches_list'])
    state.delete(row_number)

    (session['grounding_map'],
     session['names_map'],
     session['labels'],
     session['pos_labels'],
     session['matches_list']) = state.dump()

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
                           session['pos_labels'],
                           session['matches_list'])
    state.toggle_positive(label_number)

    (session['grounding_map'],
     session['names_map'],
     session['labels'],
     session['pos_labels'],
     session['matches_list']) = state.dump()

    return render_template('input.jinja2')


@bp.route('/ground_sort', methods=['POST'])
def sort_rows():
    for key in request.form:
        if key.startswith('sort.'):
            sort_col = key.partition('.')[-1]
    if sort_col == 'longform':
        longforms = current_app.config['LONGFORMS']
        sorted_order = argsort(longforms, kind='stable').tolist()
    elif sort_col == 'score':
        scores = current_app.config['SCORES']
        sorted_order = argsort(scores, kind='stable')[::-1].tolist()
    elif sort_col == 'name':
        longforms = current_app.config['LONGFORMS']
        names_map = session['names_map']
        sorted_order = argsort([('a' if names_map[longform] else 'b') +
                                names_map[longform] for longform in longforms],
                               kind='stable').tolist()
    elif sort_col == 'grounding':
        longforms = current_app.config['LONGFORMS']
        grounding_map = session['grounding_map']
        sorted_order = argsort([('a' if grounding_map[longform] else 'b') +
                                grounding_map[longform]
                                for longform in longforms],
                               kind='stable').tolist()
    session['sorted_order'] = sorted_order
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
                     grounding_map.items() if grounding != 'ignore'}
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
                 labels, pos_labels, matches_list):
        self.longforms = longforms
        self.grounding_map = grounding_map
        self.names_map = names_map
        self.labels = labels
        self.pos_labels = pos_labels
        self.matches_list = matches_list

    def dump(self):
        """Returns state needed to reload page."""
        return (self.grounding_map, self.names_map, self.labels,
                self.pos_labels, self.matches_list)

    def add(self, name, grounding, row_numbers, matches):
        """Add new names and groundings"""
        positive_groundings = set(self.labels[index]
                                  for index in self.pos_labels)
        for i in row_numbers:
            if grounding:
                self.grounding_map[self.longforms[i]] = grounding
            if name:
                self.names_map[self.longforms[i]] = name
            self.matches_list[i] = matches
        labels = sorted(set(grounding for _, grounding in
                            self.grounding_map.items()
                            if ':' in grounding))
        pos_labels = [i for i, label in enumerate(labels)
                      if label in positive_groundings]
        self.labels = labels
        self.pos_labels = pos_labels

    def delete(self, row_number):
        """Delete names and groundings"""
        longform = self.longforms[row_number]
        grounding = self.grounding_map[longform]
        self.grounding_map[longform] = self.names_map[longform] = ''
        self.matches_list[row_number] = 'unknown'
        if grounding not in self.grounding_map.values() and ':' in grounding:
            label_number = self.labels.index(grounding)
            self.pos_labels = [i if i < label_number else i - 1
                               for i in self.pos_labels if i != label_number]
            self.labels = (self.labels[:label_number] +
                           self.labels[label_number+1:])

    def toggle_positive(self, label_number):
        """Toggle labels as positive or negative"""
        pos_labels = sorted(set(self.pos_labels) ^ set([label_number]))
        self.pos_labels = pos_labels
