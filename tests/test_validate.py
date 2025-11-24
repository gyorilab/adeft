import json
import os
import numpy as np
import pytest

from adeft.modeling.classify import BaselineLogisticRegressionModel
from adeft.modeling.validate import PooledFbetaGridSearchCV

from adeft.locations import TEST_RESOURCES_PATH

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import ParameterGrid


with open(os.path.join(TEST_RESOURCES_PATH,
                       'example_training_data.json'), 'r') as f:
    data = json.load(f)


@pytest.fixture(scope="class")
def fitted_grid_search(request):
    rng = np.random.RandomState(1234)
    texts = data['texts']
    labels = data['labels']
    estimator = BaselineLogisticRegressionModel(
        penalty="l1", max_features=1000, random_state=rng
    )
    # Set up param_grid so that there is one obvious best hyperparameter choice.
    # C = 1e-10 results in too strong regularization. Feature arrays become too
    # sparse when only higher order n-grams are used.
    param_grid = {"ngram_range": [(1, 2), (3, 4)], "C": [100.0, 1e-10]}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=rng)
    grid_search = PooledFbetaGridSearchCV(
        estimator, param_grid, pos_labels=['HGNC:6091', 'MESH:D011839'], cv=cv,
        refit=False
    )
    grid_search.fit(texts, labels)
    request.cls.grid_search = grid_search
    request.cls.param_grid = param_grid


@pytest.mark.usefixtures("fitted_grid_search")
class TestPooledFbetaGridSearchCV:
    def test_best_params(self):
        # Check that the grid search found the best params.
        assert self.grid_search.best_params_ == {"C": 100.0, "ngram_range": (1, 2)}

    def test_best_score(self):
        # Check that the best score is acceptably high.
        assert self.grid_search.best_score_ > 0.8

    def test_cv_results_keys(self):
        # check that keys of cv_results_ are as expected.
        expected_keys = {
            tuple(sorted(entry.items())) for entry in ParameterGrid(self.param_grid)
        }
        assert set(self.grid_search.cv_results_.keys()) == expected_keys
