import os
import uuid
import json
import numpy as np
from nose.plugins.attrib import attr
from sklearn.metrics import f1_score

from deft.locations import MODELS_PATH
from deft.modeling.classify import DeftClassifier, load_model
from deft.download import get_downloaded_models, download_models

# Get test path so we can write a temporary file here
TESTS_PATH = os.path.dirname(os.path.abspath(__file__))

if 'TEST' not in get_downloaded_models():
    download_models(models=['TEST'])

with open(os.path.join(MODELS_PATH, 'TEST',
                       'example_training_data.json'), 'r') as f:
    data = json.load(f)


# The classifier works slightly differently for multiclass than it does for
# binary labels. Both cases must be tested separately.
@attr('slow')
def test_train():
    params = {'C': 1.0,
              'ngram_range': (1, 2),
              'max_features': 1000}
    classifier = DeftClassifier('IR', ['HGNC:6091'])
    texts = data['train']
    response = [label if label == 'HGNC:6091' else 'other'
                for label in data['response']]
    classifier.train(texts, response, **params)
    assert hasattr(classifier, 'estimator')
    assert (f1_score(response, classifier.predict(texts),
                     pos_label='HGNC:6091') > 0.75)


@attr('slow')
def test_cv_multiclass():
    params = {'C': [1.0],
              'max_features': [1000]}
    classifier = DeftClassifier('IR', ['HGNC:6091'])
    texts = data['train']
    response = data['response']
    classifier.cv(texts, response, param_grid=params)
    assert classifier.best_score > 0.7


@attr('slow')
def test_cv_binary():
    params = {'C': [1.0],
              'max_features': [1000]}
    texts = data['train']
    response = [label if label == 'HGNC:6091' else 'other'
                for label in data['response']]
    classifier = DeftClassifier('IR', ['HGNC:6091'])
    classifier.cv(texts, response, param_grid=params)
    assert classifier.best_score > 0.7


def test_serialize():
    """Test that models can correctly be saved to and loaded from gzipped json
    """
    train = data['train']
    temp_filename = os.path.join(TESTS_PATH, uuid.uuid4().hex)
    classifier1 = load_model(os.path.join(MODELS_PATH, 'TEST',
                                          'test_model.gz'))
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
