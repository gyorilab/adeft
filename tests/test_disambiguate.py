import os
import uuid
import json
import shutil
import pytest
import logging

from numpy import array_equal

from adeft.modeling.classify import load_model
from adeft.locations import TEST_RESOURCES_PATH
from adeft.disambiguate import AdeftDisambiguator, load_disambiguator

logger = logging.getLogger(__name__)

# Get test model path so we can write a temporary file here
TEST_MODEL_PATH = os.path.join(TEST_RESOURCES_PATH, 'test_model')
# Path to scratch directory to write files to during tests
SCRATCH_PATH = os.path.join(TEST_RESOURCES_PATH, 'scratch')

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
    ad = load_disambiguator('IR', path=TEST_MODEL_PATH)
    assert ad.shortforms == ['IR']
    assert hasattr(ad, 'classifier')
    assert hasattr(ad, 'recognizers')


def test_dump_disambiguator():
    ad1 = load_disambiguator('IR', path=TEST_MODEL_PATH)
    tempname = uuid.uuid4().hex
    ad1.dump(tempname, path=SCRATCH_PATH)
    ad2 = load_disambiguator('IR', path=SCRATCH_PATH)

    assert ad1.grounding_dict == ad2.grounding_dict
    assert ad1.names == ad2.names
    assert ad1.pos_labels == ad2.pos_labels
    assert (array_equal(ad1.classifier.estimator.named_steps['logit'].coef_,
                        ad2.classifier.estimator.named_steps['logit'].coef_))
    assert ad1.info() == ad2.info(), (ad1.info(), ad2.info())
    assert ad1.version() == ad2.version()
    try:
        shutil.rmtree(os.path.join(SCRATCH_PATH, tempname))
    except Exception:
        logger.warning('Could not clean up temporary folder %s'
                       % os.path.join(SCRATCH_PATH, tempname))


def test_disambiguate():
    test_model = load_model(os.path.join(TEST_MODEL_PATH, 'IR',
                                         'IR_model.gz'))
    with open(os.path.join(TEST_MODEL_PATH, 'IR',
                           'IR_grounding_dict.json')) as f:
        grounding_dict = json.load(f)
    with open(os.path.join(TEST_MODEL_PATH, 'IR',
                           'IR_names.json')) as f:
        names = json.load(f)

    ad = AdeftDisambiguator(test_model, grounding_dict, names)
    # case where there is a unique defining pattern
    disamb1 = ad.disambiguate(example1)
    assert disamb1[0] == 'HGNC:6091'
    assert disamb1[1] == 'INSR'
    assert disamb1[2]['HGNC:6091'] == 1.0
    assert disamb1[2]['MESH:D011839'] == 0.0

    # case where there are conflicting defining patterns
    disamb2 = ad.disambiguate(example2)
    preds = disamb2[2]
    nonzero = {key for key, value in preds.items() if value > 0.0}
    assert nonzero == {'HGNC:6091', 'MESH:D007333'}

    # case without a defining pattern
    disamb3 = ad.disambiguate(example3)
    assert disamb3[0] == 'HGNC:6091'
    assert disamb3[1] == 'INSR'


def test_modify_groundings():
    """Test updating groundings of existing model."""
    ad = load_disambiguator('IR', path=TEST_MODEL_PATH)
    ad.modify_groundings(new_groundings={'HGNC:6091': 'UP:P06213'},
                         new_names={'HGNC:6091': 'Insulin Receptor'})

    assert 'UP:P06213' in ad.pos_labels
    assert 'UP:P06213' in ad.classifier.pos_labels
    assert 'UP:P06213' in ad.classifier.estimator.classes_
    assert 'UP:P06213' in ad.names
    assert 'UP:P06213' in ad.grounding_dict['IR'].values()
    assert ad.names['UP:P06213'] == 'Insulin Receptor'


def test_update_pos_labels():
    """Test updating of positive labels in existing model."""
    ad1 = load_disambiguator('IR', path=TEST_MODEL_PATH)
    ad2 = load_disambiguator('IR', path=TEST_MODEL_PATH)
    ad2.update_pos_labels(ad1.pos_labels)
    assert ad1.classifier.stats == ad2.classifier.stats
    ad2.update_pos_labels(ad1.pos_labels + ['MESH:D007333'])
    assert set(ad2.pos_labels) == set(['HGNC:6091', 'MESH:D011839',
                                       'MESH:D007333'])


def test_modify_groundings_error():
    ad = load_disambiguator('IR', path=TEST_MODEL_PATH)
    with pytest.raises(ValueError):
        ad.modify_groundings(new_groundings={'MESH:D011839': 'HGNC:6091'})
