import os
import json

from deft.locations import MODELS_PATH
from deft.modeling.classify import load_model
from deft.disambiguate import DeftDisambiguator, load_disambiguator

example1 = ('The insulin receptor (IR) is a transmembrane receptor that'
            ' is activated by insulin, IGF-I, IGF-II and belongs to the large'
            ' class of tyrosine kinase receptors')

example2 = ('The insulin receptor (IR) is a transmembrane receptor that'
            ' is activated by insulin, IGF-I, IGF-II and belongs to the large'
            ' class of tyrosine kinase receptors. Insulin resistance (IR)'
            ' is considered as a pathological condition in which cells fail'
            ' to respond normally to the hormone insulin')


def test_load_disambiguator():
    dd_test = load_disambiguator('TEST')
    assert dd_test.shortform == 'IR'
    assert hasattr(dd_test, 'classifier')
    assert hasattr(dd_test, 'recognizer')


def test_disambiguate():
    test_model = load_model(os.path.join(MODELS_PATH, 'TEST',
                                         'test_model.gz'))
    with open(os.path.join(MODELS_PATH, 'TEST',
                           'test_grounding_map.json')) as f:
        grounding_map = json.load(f)
    with open(os.path.join(MODELS_PATH, 'TEST', 'test_names.json')) as f:
        names = json.load(f)

    dd = DeftDisambiguator(test_model, grounding_map, names)
    for disamb in dd.disambiguate([example1, example2]):
        print(disamb)
