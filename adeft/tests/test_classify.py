import os
import uuid
import json
import numpy as np
from nose.plugins.attrib import attr
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
@attr('slow')
def test_train():
    params = {'C': 1.0,
              'ngram_range': (1, 2),
              'max_features': 10}
    classifier = AdeftClassifier('IR', ['HGNC:6091', 'MESH:D011839'],
                                 random_state=1729)
    texts = data['texts']
    labels = data['labels']
    expected_f1 = 0.7759358447458177
    expected_fimp = {'HGNC:6091': [('irs', 0.6320),
                                   ('igf', 0.6245),
                                   ('insulin', 0.6107),
                                   ('cell', 0.0),
                                   ('cells', 0.0),
                                   ('expression', 0.0),
                                   ('levels', 0.0),
                                   ('study', 0.0),
                                   ('group', -0.2162),
                                   ('induced', -0.3138)],
                     'MESH:D011839': [('cells', 0.4574),
                                      ('induced', 0.3435),
                                      ('cell', 0.2507),
                                      ('expression', 0.0),
                                      ('group', 0.0),
                                      ('igf', 0.0),
                                      ('irs', 0.0),
                                      ('study', 0.0),
                                      ('levels', -0.2277),
                                      ('insulin', -1.8653)],
                     'ungrounded': [('study', 0.4991),
                                    ('group', 0.4120),
                                    ('expression', 0.0),
                                    ('induced', 0.0),
                                    ('insulin', 0.0),
                                    ('irs', 0.0),
                                    ('levels', 0.0),
                                    ('cell', -0.0399),
                                    ('igf', -0.0579),
                                    ('cells', -0.9570)]}
    classifier.train(texts, labels, **params)
    assert hasattr(classifier, 'estimator')
    assert (f1_score(labels, classifier.predict(texts),
                     labels=['HGNC:6091', 'MESH:D011839'],
                     average='weighted') - expected_f1 < 1e-10)
    assert classifier.feature_importances() == expected_fimp


@attr('slow')
def test_cv_multiclass():
    params = {'C': [1.0],
              'max_features': [10]}
    # Classification results are now reproducible. We should be aware of any
    # modifications that will change these results.
    expected_stats = {'label_distribution': {'HGNC:6091': 101,
                                             'MESH:D011839': 173,
                                             'ungrounded': 226},
                      'f1': {'mean': 0.704288,
                             'std': 0.049344},
                      'precision': {'mean': 0.784593,
                                    'std': 0.011247},
                      'recall': {'mean': 0.661125,
                                 'std': 0.074169},
                      'ungrounded': {'f1': {'mean': 0.744231,
                                            'std': 0.005769},
                                     'pr': {'mean': 0.809735,
                                            'std': 0.039823},
                                     'rc': {'mean': 0.692077,
                                            'std': 0.039016}},
                      'MESH:D011839': {'f1': {'mean': 0.782527,
                                              'std': 0.048799},
                                       'pr': {'mean': 0.757485,
                                              'std': 0.044841},
                                       'rc': {'mean': 0.809299,
                                              'std': 0.053201}},
                      'HGNC:6091': {'f1': {'mean': 0.570274,
                                           'std': 0.049726},
                                    'pr': {'mean': 0.496275,
                                           'std': 0.123725},
                                    'rc': {'mean': 0.741818,
                                           'std': 0.121818}}}
    classifier = AdeftClassifier('IR', ['HGNC:6091', 'MESH:D011839'],
                                 random_state=1729)
    texts = data['texts']
    labels = data['labels']
    classifier.cv(texts, labels, param_grid=params, cv=2)
    assert classifier.stats == expected_stats


@attr('slow')
def test_cv_binary():
    params = {'C': [1.0],
              'max_features': [10]}
    texts = data['texts']
    expected_stats = {'label_distribution': {'HGNC:6091': 101,
                                             'ungrounded': 399},
                      'f1': {'mean': 0.536613,
                             'std': 0.115561},
                      'precision': {'mean': 0.677143,
                                    'std': 0.037143},
                      'recall': {'mean': 0.456863,
                                 'std': 0.143137},
                      'HGNC:6091': {'f1': {'mean': 0.536613,
                                           'std': 0.115561},
                                    'pr': {'mean': 0.456863,
                                           'std': 0.143137},
                                    'rc': {'mean': 0.677143,
                                           'std': 0.037143}},
                      'ungrounded': {'f1': {'mean': 0.908948,
                                            'std': 0.012234},
                                     'pr': {'mean': 0.947349,
                                            'std': 0.007651},
                                     'rc': {'mean': 0.874257,
                                            'std': 0.029124}}}
    expected_fi = {'ungrounded': [('group', 0.6063),
                                  ('induced', 0.4483),
                                  ('study', 0.2721),
                                  ('cell', 0.0),
                                  ('expression', -0.0017),
                                  ('levels', -0.0187),
                                  ('cells', -0.0761),
                                  ('irs', -0.6719),
                                  ('igf', -0.7052),
                                  ('insulin', -0.7869)],
                   'HGNC:6091': [('insulin', 0.7869),
                                 ('igf', 0.7052),
                                 ('irs', 0.6719),
                                 ('cells', 0.0761),
                                 ('levels', 0.0187),
                                 ('expression', 0.0017),
                                 ('cell', -0.0),
                                 ('study', -0.2721),
                                 ('induced', -0.4483),
                                 ('group', -0.6063)]}
    labels = [label if label == 'HGNC:6091' else 'ungrounded'
              for label in data['labels']]
    classifier = AdeftClassifier('IR', ['HGNC:6091'], random_state=1729)
    classifier.cv(texts, labels, param_grid=params, cv=2)
    assert classifier.stats == expected_stats
    assert classifier.feature_importances() == expected_fi


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
    os.remove(temp_filename)
