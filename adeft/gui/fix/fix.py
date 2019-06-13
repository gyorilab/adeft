import os
import json

from flask import Blueprint, request, render_template, session, current_app


bp = Blueprint('fix', __name__)


@bp.route('/change_grounding', methods=['POST'])
def change_grounding():
    for key in request.form:
        if key.startswith('s.'):
            index = key.partition('.')[-1]
    new_name = request.form[f'new-name.{index}'].strip()
    new_ground = request.form[f'new-ground.{index}'].strip()

    original_groundings = current_app.config['GROUNDINGS']

    if new_name:
        names_transition = session['names_transition']
        names_transition[original_groundings[int(index)]] = new_name
        session['names_transition'] = names_transition
    if new_ground:
        groundings_transition = session['groundings_transition']
        groundings_transition[original_groundings[int(index)]] = new_ground
        session['groundings_transition'] = groundings_transition

    groundings = [groundings_transition[grounding]
                  for grounding in original_groundings]
    names = [names_transition[grounding]
             for grounding in original_groundings]

    data = zip(groundings, names,
               current_app.config['LONGFORMS'],
               session['pos_labels'])
    return render_template('fix.jinja2', data=data)


@bp.route('/toggle_positive', methods=['POST'])
def toggle_positive():
    for key in request.form:
        if key.startswith('pos-label.'):
            index = key.partition('.')[-1]

    pos_labels = session['pos_labels']
    pos_labels[int(index)] = not pos_labels[int(index)]
    session['pos_labels'] = pos_labels

    original_groundings = current_app.config['GROUNDINGS']
    groundings_transition = session['groundings_transition']
    names_transition = session['names_transition']

    groundings = [groundings_transition[grounding]
                  for grounding in original_groundings]
    names = [names_transition[grounding]
             for grounding in original_groundings]

    data = zip(groundings, names,
               current_app.config['LONGFORMS'],
               session['pos_labels'])
    return render_template('fix.jinja2', data=data)


@bp.route('/submit', methods=['POST'])
def submit():
    return 'Update submitted'
