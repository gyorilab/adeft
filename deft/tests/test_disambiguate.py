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

example3 = ('IR is a transmembrane receptor that is activated by insulin,'
            ' IGF-1, IFG-II and belongs to the large class of tyrosine'
            ' kinase receptors')


def test_load_disambiguator():
    dd_test = load_disambiguator('TEST')
    assert dd_test.shortforms == ['IR']
    assert hasattr(dd_test, 'classifier')
    assert hasattr(dd_test, 'recognizers')


def test_disambiguate():
    test_model = load_model(os.path.join(MODELS_PATH, 'TEST',
                                         'TEST_model.gz'))
    with open(os.path.join(MODELS_PATH, 'TEST',
                           'TEST_grounding_dict.json')) as f:
        grounding_dict = json.load(f)
    with open(os.path.join(MODELS_PATH, 'TEST', 'TEST_names.json')) as f:
        names = json.load(f)

    dd = DeftDisambiguator(test_model, grounding_dict, names)
    # case where there is a unique defining pattern
    disamb1 = dd.disambiguate([example1])[0]
    assert disamb1[0] == 'HGNC:6091'
    assert disamb1[1] == 'INSR'
    assert disamb1[2] == {'HGNC:6091': 1.0, 'MESH:D011839': 0.0,
                          'ungrounded': 0.0}

    # case where there are conflicting defining patterns
    disamb2 = dd.disambiguate([example2])[0]
    preds = disamb2[2]
    nonzero = {key for key, value in preds.items() if value > 0.0}
    assert nonzero == {'HGNC:6091', 'ungrounded'}

    # case without a defining pattern
    disamb3 = dd.disambiguate([example3])[0]
    assert disamb3[0] == 'HGNC:6091'
    assert disamb3[1] == 'INSR'
