import os
import json

from adeft.locations import MODELS_PATH
from adeft.modeling.classify import load_model
from adeft.disambiguate import AdeftDisambiguator, load_disambiguator

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
    ad = load_disambiguator('__TEST')
    assert ad.shortforms == ['IR']
    assert hasattr(ad, 'classifier')
    assert hasattr(ad, 'recognizers')


def test_disambiguate():
    test_model = load_model(os.path.join(MODELS_PATH, '__TEST',
                                         '__TEST_model.gz'))
    with open(os.path.join(MODELS_PATH, '__TEST',
                           '__TEST_grounding_dict.json')) as f:
        grounding_dict = json.load(f)
    with open(os.path.join(MODELS_PATH, '__TEST', '__TEST_names.json')) as f:
        names = json.load(f)

    ad = AdeftDisambiguator(test_model, grounding_dict, names)
    # case where there is a unique defining pattern
    disamb1 = ad.disambiguate([example1])[0]
    assert disamb1[0] == 'HGNC:6091'
    assert disamb1[1] == 'INSR'
    assert disamb1[2] == {'HGNC:6091': 1.0, 'MESH:D011839': 0.0,
                          'ungrounded': 0.0}

    # case where there are conflicting defining patterns
    disamb2 = ad.disambiguate([example2])[0]
    preds = disamb2[2]
    nonzero = {key for key, value in preds.items() if value > 0.0}
    assert nonzero == {'HGNC:6091', 'ungrounded'}

    # case without a defining pattern
    disamb3 = ad.disambiguate([example3])[0]
    assert disamb3[0] == 'HGNC:6091'
    assert disamb3[1] == 'INSR'
