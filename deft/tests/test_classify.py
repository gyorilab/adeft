import os
import uuid
import json
import numpy as np
from nose.plugins.attrib import attr
from deft.modeling.classify import DeftClassifier, load_model

# Get test directory so necessary datafiles can be found from any working
# directory
TEST_DIR = os.path.dirname(os.path.abspath(__file__))

# Example data contains 1000 labeled texts with shortform IR
with open('%s/example_training_data.json' % TEST_DIR, 'r') as f:
    data = json.load(f)

# The classifier works slightly differently for multiclass than it does for
# binary labels. Both cases must be tested separately.


@attr('slow')
def test_cv_multiclass():
    params = {'C': [1.0],
              'max_features': [1000]}
    classifier = DeftClassifier('IR', ['HGNC:6091'])
    train = data['train']
    response = data['response']
    classifier.cv(train, response, param_grid=params)
    assert classifier.best_score > 0.7


@attr('slow')
def test_cv_binary():
    params = {'C': [1.0],
              'max_features': [1000]}
    train = data['train']
    response = [label if label == 'HGNC:6091' else 'other'
                for label in data['response']]
    classifier = DeftClassifier('IR', ['HGNC:6091'])
    classifier.cv(train, response, param_grid=params)
    assert classifier.best_score > 0.7


def test_serialize():
    """Test that models can correctly be saved to and loaded from gzipped json
    """
    train = data['train']
    temp_filename = '%s/%s' % (TEST_DIR, uuid.uuid4().hex)

    classifier1 = load_model('%s/example_model.gz' % TEST_DIR)
    classifier1.dump_model(temp_filename)

    classifier2 = load_model(temp_filename)
    classifier2.dump_model(temp_filename)

    classifier3 = load_model(temp_filename)

    preds1, preds2, preds3 = (classifier1.predict_proba(train),
                              classifier2.predict_proba(train),
                              classifier3.predict_proba(train))
    assert np.array_equal(preds1, preds2)
    assert np.array_equal(preds2, preds3)
    os.remove(temp_filename)
