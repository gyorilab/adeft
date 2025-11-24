import os
import pytest
import uuid
import json
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics import f1_score

import adeft

from adeft.locations import TEST_RESOURCES_PATH
from adeft.modeling.classify import AdeftClassifier, BaseModel


# Get test model path so we can write a temporary file here
TEST_MODEL_PATH = os.path.join(TEST_RESOURCES_PATH, 'test_model')
# Path to scratch directory to write files to during tests
SCRATCH_PATH = os.path.join(TEST_RESOURCES_PATH, 'scratch')


with open(os.path.join(TEST_RESOURCES_PATH,
                       'example_training_data.json'), 'r') as f:
    data = json.load(f)


@pytest.fixture(scope="class")
def train_model(request):
    random_seed = 1729
    params = {
        "C": 1.0,
        "ngram_range": (1, 2),
        "max_features": 10
    }
    classifier = AdeftClassifier(
        "IR", ["HGNC:6091", "MESH:D011839"],
        random_state=random_seed
    )
    texts = data["texts"]
    labels = data["labels"]
    classifier.train(texts, labels, **params)
    request.cls.classifier = classifier
    request.cls.random_seed = random_seed
    request.cls.params = params
    

@pytest.mark.usefixtures("train_model")
class TestTrain:
    def test_f1_score_on_training_data(self):
        # Just a sanity check that the model fit the training data
        # reasonably well.
        labels = data["labels"]
        texts = data["texts"]
        assert (
            f1_score(
                labels, self.classifier.predict(texts),
                labels=["HGNC:6091", "MESH:D011839"],
                average="macro"
            ) > 0.7
        )

    def test_feature_importances(self):
        importances = self.classifier.feature_importances()
        INSR_features, INSR_scores = zip(*importances['HGNC:6091'])
        assert set(['irs', 'igf', 'insulin']) < set(INSR_features)
        irs_score = [score for feature, score in importances['HGNC:6091']
                     if feature == 'irs'][0]
        assert irs_score > 0

    def test_repeatable(self):
        classifier2 = AdeftClassifier(
            "IR", ["HGNC:6091", "MESH:D011839"],
            random_state=self.random_seed
        )
        texts = data['texts']
        labels = data['labels']
        classifier2.train(texts, labels, **self.params)
        coef1 = self.classifier.estimator.pipeline.named_steps['logit'].coef_
        coef2 = classifier2.estimator.pipeline.named_steps['logit'].coef_
        assert np.array_equal(coef1, coef2)

    def test_metadata(self):
        assert self.classifier.version == adeft.__version__
        now = datetime.now()
        model_timestamp = datetime.fromisoformat(self.classifier.timestamp)
        assert now - timedelta(minutes=30) <= model_timestamp <= now


@pytest.fixture(scope="class", params=["multiclass", "binary"])
def validated_classifier(request):
    params = {"C": [1.0], "max_features": [100]}
    classifier = AdeftClassifier(
        "IR", ["HGNC:6091", "MESH:D011839"],
        random_state=1729
    )
    texts = data['texts']
    labels = data['labels']
    if request.param == "binary":
        labels = [
            label if label == "HGNC:6091" else "ungrounded"
            for label in labels
        ]
        classifier = AdeftClassifier("IR", ["HGNC:6091"], random_state=1729)
    else:
        classifier = AdeftClassifier(
            "IR", ["HGNC:6091", "MESH:D011839"],
            random_state=1729
        )
    classifier.validate(
        texts, labels, param_grid=params, n_outer_splits=2, n_inner_splits=2,
        refit=True
    )
    request.cls.classifier = classifier


@pytest.mark.usefixtures("validated_classifier")
class TestValidateClassifier:
    @pytest.mark.parametrize("fold_id", [0, 1])
    def test_validation_results_structure(self, fold_id):
        validation_results = self.classifier.validation_results[fold_id]
        assert set(validation_results.keys()) == set(
            ["sensitivity", "specificity", "support", "confusion_matrix"]
        )
        n_labels = len(self.classifier.labels)
        assert validation_results["sensitivity"].shape == (n_labels,)
        assert validation_results["specificity"].shape == (n_labels,)
        assert validation_results["support"].shape == (n_labels,)
        assert validation_results["confusion_matrix"].shape == (n_labels, n_labels)

    @pytest.mark.parametrize("fold_id", [0, 1])
    def test_validation_results_consistency(self, fold_id):
        validation_results = self.classifier.validation_results[fold_id]
        confusion_matrix = validation_results["confusion_matrix"]
        support = validation_results["support"]
        sensitivity = validation_results["sensitivity"]
        specificity = validation_results["specificity"]
        assert np.array_equal(confusion_matrix.sum(axis=1), support)
        TP = np.diag(confusion_matrix)
        FP = confusion_matrix.sum(axis=0) - TP
        FN = confusion_matrix.sum(axis=1) - TP
        TN = confusion_matrix.sum() - (TP + FP + FN)
        assert np.array_equal(TP / (TP + FN), sensitivity)
        assert np.array_equal(TN / (TN + FP), specificity)

    @pytest.mark.parametrize("fold_id", [0, 1])
    def test_validation_results_sensitivity(self, fold_id):
        validation_results = self.classifier.validation_results[fold_id]
        assert np.all(validation_results["sensitivity"] > 0.6)

    @pytest.mark.parametrize("fold_id", [0, 1])
    def test_validation_results_specificity(self, fold_id):
        validation_results = self.classifier.validation_results[fold_id]
        assert np.all(validation_results["specificity"] > 0.6)

    @pytest.mark.parametrize(
        "kind",
        ["inner_model_selection_results_", "outer_model_selection_results_"]
    )
    def test_model_selection_results(self, kind):
        if kind == "inner":
            model_selection_results = self.classifier.inner_model_selection_results_
        else:
            model_selection_results = self.classifier.outer_model_selection_results_
        assert set(model_selection_results.keys()) == set(
            ["best_score", "best_params", "cv_results"]
        )
        assert model_selection_results["best_score"] > 0.6
        assert model_selection_results["best_params"] == {
            "C": 1.0, "max_features": 100
        }


def test_training_set_digest():
    classifier = AdeftClassifier('?', ['??', '???'])
    texts = data['texts']
    digest1 = classifier._training_set_digest(texts)
    digest2 = classifier._training_set_digest(texts[::-1])
    digest3 = classifier._training_set_digest(texts[:-1])
    assert digest1 == digest2
    assert digest1 != digest3

@pytest.mark.skip(reason="Test model currently incompatible.")
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
    assert np.array_equal(classifier1.estimator._feature_stds,
                          classifier2.estimator._feature_stds)
    assert np.array_equal(classifier2.estimator._feature_stds,
                          classifier3.estimator._feature_stds)
    # Check classifier versions are unchanged
    assert classifier1.version == classifier2.version == \
        classifier3.version
    # Check that model params are unchanged
    assert classifier1.params == classifier2.params == classifier3.params
    assert classifier2.other_metadata == classifier3.other_metadata
    os.remove(temp_filename)
