import os
import uuid
import json
import numpy as np
from sklearn.metrics import f1_score

from adeft.locations import TEST_RESOURCES_PATH
from adeft.modeling.classify import AdeftClassifier, load_model


# Get test model path so we can write a temporary file here
TEST_MODEL_PATH = os.path.join(TEST_RESOURCES_PATH, 'test_model')
# Path to scratch directory to write files to during tests
SCRATCH_PATH = os.path.join(TEST_RESOURCES_PATH, 'scratch')


with open(os.path.join(TEST_RESOURCES_PATH,
                       'example_training_data.json'), 'r') as f:
    data = json.load(f)


# The classifier works slightly differently for multiclass than it does for
# binary labels. Both cases must be tested separately.
def test_train():
    params = {'C': 1.0,
              'ngram_range': (1, 2),
              'max_features': 10}
    classifier = AdeftClassifier('IR', ['HGNC:6091', 'MESH:D011839'],
                                 random_state=1729)
    texts = data['texts']
    labels = data['labels']
    classifier.train(texts, labels, **params)
    assert hasattr(classifier, 'estimator')
    assert (f1_score(labels, classifier.predict(texts),
                     labels=['HGNC:6091', 'MESH:D011839'],
                     average='weighted') > 0.5)
    importances = classifier.feature_importances()
    INSR_features, INSR_scores = zip(*importances['HGNC:6091'])
    assert set(['irs', 'igf', 'insulin']) < set(INSR_features)
    irs_score = [score for feature, score in importances['HGNC:6091']
                 if feature == 'irs'][0]
    assert irs_score > 0
    # test that results are repeatable
    coef1 = classifier.estimator.named_steps['logit'].coef_
    classifier.train(texts, labels, **params)
    coef2 = classifier.estimator.named_steps['logit'].coef_
    assert np.array_equal(coef1, coef2)


def test_cv_multiclass():
    params = {'C': [1.0],
              'max_features': [10]}
    classifier = AdeftClassifier('IR', ['HGNC:6091', 'MESH:D011839'],
                                 random_state=1729)
    texts = data['texts']
    labels = data['labels']
    classifier.cv(texts, labels, param_grid=params, cv=2)
    assert classifier.stats['f1']['mean'] > 0.5
    assert classifier.stats['ungrounded']['f1']['mean'] > 0.5
    # Test that results are repeatable
    coef1 = classifier.estimator.named_steps['logit'].coef_
    classifier.cv(texts, labels, param_grid=params, cv=2)
    coef2 = classifier.estimator.named_steps['logit'].coef_
    assert np.array_equal(coef1, coef2)


def test_cv_binary():
    params = {'C': [1.0],
              'max_features': [10]}
    texts = data['texts']
    labels = [label if label == 'HGNC:6091' else 'ungrounded'
              for label in data['labels']]
    classifier = AdeftClassifier('IR', ['HGNC:6091'], random_state=1729)
    classifier.cv(texts, labels, param_grid=params, cv=2)
    assert classifier.stats['f1']['mean'] > 0.5
    assert classifier.stats['HGNC:6091']['f1']['mean'] > 0.5
    importances = classifier.feature_importances()
    INSR_features, INSR_scores = zip(*importances['HGNC:6091'])
    ungrounded_features, ungrounded_scores = zip(*importances['ungrounded'])
    assert set(INSR_features) == set(ungrounded_features)
    assert INSR_scores == tuple(-x for x in ungrounded_scores[::-1])
    assert [score for feature, score in importances['HGNC:6091']
            if feature == 'insulin'][0] > 0
    assert [score for feature, score in importances['HGNC:6091']
            if feature == 'group'][0] < 0


def test_training_set_digest():
    classifier = AdeftClassifier('?', ['??', '???'])
    texts = data['texts']
    digest1 = classifier._training_set_digest(texts)
    digest2 = classifier._training_set_digest(texts[::-1])
    digest3 = classifier._training_set_digest(texts[:-1])
    assert digest1 == digest2
    assert digest1 != digest3


def test_serialize():
    """Test that models can correctly be saved to and loaded from gzipped json
    """
    texts = data['texts']
    classifier1 = load_model(os.path.join(TEST_MODEL_PATH, 'IR',
                                          'IR_model.gz'))
    temp_filename = os.path.join(SCRATCH_PATH, uuid.uuid4().hex)
    classifier1.dump_model(temp_filename)

    classifier2 = load_model(temp_filename)
    classifier2.other_metadata = {'test': 'This is a test.'}
    classifier2.dump_model(temp_filename)

    classifier3 = load_model(temp_filename)

    preds1, preds2, preds3 = (classifier1.predict_proba(texts),
                              classifier2.predict_proba(texts),
                              classifier3.predict_proba(texts))
    # Check that generated predictions are the same
    assert np.array_equal(preds1, preds2)
    assert np.array_equal(preds2, preds3)
    # Check that model stats are the same
    assert classifier1.stats == classifier2.stats == classifier3.stats
    # Check that the calculated feature importance scores are the same
    assert classifier1.feature_importances() == \
        classifier2.feature_importances() == \
        classifier3.feature_importances()
    # Check timestamps are unchanged
    assert classifier1.timestamp == classifier2.timestamp == \
        classifier3.timestamp
    # Check hash of training set is unchanged
    assert classifier1.training_set_digest == \
        classifier2.training_set_digest == \
        classifier3.training_set_digest
    # Check standard deviations of feature values are unchanged
    assert np.array_equal(classifier1._std,
                          classifier2._std)
    assert np.array_equal(classifier2._std,
                          classifier3._std)
    # Check classifier versions are unchanged
    assert classifier1.version == classifier2.version == \
        classifier3.version
    # Check that model params are unchanged
    assert classifier1.params == classifier2.params == classifier3.params
    assert classifier2.other_metadata == classifier3.other_metadata
    os.remove(temp_filename)
