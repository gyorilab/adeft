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
    expected_fimp = {'HGNC:6091': [('irs', 0.6319904303127237),
                                   ('igf', 0.6244993165477428),
                                   ('insulin', 0.6107296608833942),
                                   ('cell', 0.0),
                                   ('cells', 0.0),
                                   ('expression', 0.0),
                                   ('levels', 0.0),
                                   ('study', 0.0),
                                   ('group', -0.21622950261629262),
                                   ('induced', -0.31384488262833754)],
                     'MESH:D011839': [('cells', 0.45739005257538706),
                                      ('induced', 0.34351591977927864),
                                      ('cell', 0.2507275744935278),
                                      ('expression', 0.0),
                                      ('group', 0.0),
                                      ('igf', 0.0),
                                      ('irs', 0.0),
                                      ('study', 0.0),
                                      ('levels', -0.22767940838933154),
                                      ('insulin', -1.8653220001013207)],
                     'ungrounded': [('study', 0.49914512828421254),
                                    ('group', 0.4120499676747824),
                                    ('expression', 0.0),
                                    ('induced', 0.0),
                                    ('insulin', 0.0),
                                    ('irs', 0.0),
                                    ('levels', 0.0),
                                    ('cell', -0.03990002449002259),
                                    ('igf', -0.057946023408154995),
                                    ('cells', -0.9569664542625634)]}
    classifier.train(texts, labels, **params)
    assert hasattr(classifier, 'estimator')
    assert (f1_score(labels, classifier.predict(texts),
                     labels=['HGNC:6091', 'MESH:D011839'],
                     average='weighted') == expected_f1)
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
                      'f1': {'mean': 0.7042880615479887,
                             'std': 0.04934411421388235},
                      'precision': {'mean': 0.784592875097113,
                                    'std': 0.011247286861818862},
                      'recall': {'mean': 0.6611253196930946,
                                 'std': 0.0741687979539642},
                      'ungrounded': {'f1': {'mean': 0.7442307692307693,
                                            'std': 0.005769230769230749},
                                     'pr': {'mean': 0.8097345132743363,
                                            'std': 0.03982300884955753},
                                     'rc': {'mean': 0.6920768307322929,
                                            'std': 0.039015606242497}},
                      'MESH:D011839': {'f1': {'mean': 0.7825265559278535,
                                              'std': 0.048798745276965816},
                                       'pr': {'mean': 0.7574846297781341,
                                              'std': 0.04484095161721464},
                                       'rc': {'mean': 0.8092987804878049,
                                              'std': 0.05320121951219514}},
                      'HGNC:6091': {'f1': {'mean': 0.5702739726027397,
                                           'std': 0.049726027397260286},
                                    'pr': {'mean': 0.49627450980392157,
                                           'std': 0.12372549019607842},
                                    'rc': {'mean': 0.7418181818181818,
                                           'std': 0.12181818181818183}}}
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
                      'f1': {'mean': 0.5366132723112128,
                             'std': 0.11556064073226546},
                      'precision': {'mean': 0.6771428571428572,
                                    'std': 0.037142857142857144},
                      'recall': {'mean': 0.4568627450980392,
                                 'std': 0.14313725490196078},
                      'HGNC:6091': {'f1': {'mean': 0.5366132723112128,
                                           'std': 0.11556064073226546},
                                    'pr': {'mean': 0.4568627450980392,
                                           'std': 0.14313725490196078},
                                    'rc': {'mean': 0.6771428571428572,
                                           'std': 0.037142857142857144}},
                      'ungrounded': {'f1': {'mean': 0.9089479405166632,
                                            'std': 0.012234325493189036},
                                     'pr': {'mean': 0.9473492462311557,
                                            'std': 0.007650753768844221},
                                     'rc': {'mean': 0.8742571929374545,
                                            'std': 0.029124449574622735}}}
    expected_fi = {'ungrounded': [('group', 0.6062620904314349),
                                  ('induced', 0.4483443313524849),
                                  ('study', 0.2720945730518759),
                                  ('cell', 0.0),
                                  ('expression', -0.0016821842311407226),
                                  ('levels', -0.018662036294113858),
                                  ('cells', -0.07613879680797145),
                                  ('irs', -0.671925926453233),
                                  ('igf', -0.7051661736643143),
                                  ('insulin', -0.7868607115262102)],
                   'HGNC:6091': [('insulin', 0.7868607115262102),
                                 ('igf', 0.7051661736643143),
                                 ('irs', 0.671925926453233),
                                 ('cells', 0.07613879680797145),
                                 ('levels', 0.018662036294113858),
                                 ('expression', 0.0016821842311407226),
                                 ('cell', -0.0),
                                 ('study', -0.2720945730518759),
                                 ('induced', -0.4483443313524849),
                                 ('group', -0.6062620904314349)]}
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
