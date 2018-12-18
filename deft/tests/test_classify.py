import os
import uuid
import json
import numpy as np
from nose.plugins.attrib import attr
from deft.classify import LongformClassifier, load_model

# Example data contains 1000 labeled texts with shortform IR
with open('example_training_data.json', 'r') as f:
    data = json.load(f)

# The classifier works slightly differently for multiclass than it does for
# binary labels. Both cases must be tested separately.


@attr('slow')
def test_classify_multiclass():
    params = {'C': [1.0, 10.0],
              'max_features': [5000, 10000]}
    classifier = LongformClassifier('IR', ['HGNC:6091'])
    train = data['train']
    response = data['response']
    classifier.train(train, response, params=params)
    print(classifier.best_score)
    assert classifier.best_score > 0.8


@attr('slow')
def test_classify_binary():
    params = {'C': [10.0],
              'max_features': [10000]}
    train = data['train']
    response = [label if label == 'HGNC:6091' else 'other'
                for label in data['response']]
    classifier = LongformClassifier('IR', ['HGNC:6091'])
    classifier.train(train, response, params=params)
    print(classifier.best_score)
    assert classifier.best_score > 0.8


def test_serialize():
    """Test that models can correctly be saved to and loaded from gzipped json
    """
    train = data['train']
    temp_filename = uuid.uuid4().hex

    classifier1 = load_model('example_model.gz')
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
