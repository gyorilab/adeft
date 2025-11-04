import numpy as np

from collections import defaultdict
from sklearn.base import clone, BaseEstimator, MetaEstimatorMixin
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection._validation import _fit_and_score
from sklearn.utils._array_api import device, get_namespace
from sklearn.utils.metaestimators import _safe_split

from joblib import delayed, Parallel


def _count_scores(y_true, y_pred, *, labels):
    xp, _ = get_namespace(y_true, y_pred)
    MCM = multilabel_confusion_matrix(
        y_true, y_pred, labels=labels, samplewise=False
    )
    tp_sum = MCM[:, 1, 1]
    pred_sum = tp_sum + MCM[:, 0, 1]
    true_sum = tp_sum + MCM[:, 1, 0]

    tp_sum, pred_sum, true_sum = map(xp.sum, (tp_sum, pred_sum, true_sum))
    return np.array([tp_sum, pred_sum, true_sum], dtype=float)


def _fit_and_get_counts(estimator, X, y, train, test, parameters, labels):
    xp, _ = get_namespace(X)
    X_device = device(X)
    # Make sure that we can fancy index X even if train and test are provided
    # as NumPy arrays by NumPy only cross-validation splitters.
    train, test = map(lambda t: xp.asarray(t, device=X_device), (train, test))

    estimator = estimator.set_params(**clone(parameters, safe=False))
    X_train, y_train = _safe_split(estimator, X, y, train)
    X_test, y_test = _safe_split(estimator, X, y, test, train)
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)
    key = tuple(sorted(parameters.items()))
    return (key, _count_scores(y_test, y_pred, labels=labels))


class PooledFbetaGridSearchCV(MetaEstimatorMixin, BaseEstimator):
    def __init__(
            self, estimator, param_grid, *,
            beta=1.0, n_jobs=1, pos_labels, cv, refit=True
    ):
        self.estimator = estimator
        self.param_grid = param_grid
        self.beta = beta
        self.pos_labels = pos_labels
        self.n_jobs = n_jobs
        self.cv = cv
        self.refit = refit

    def fit(self, X, y):
        jobs = [
            (params, train_idx, test_idx)
            for params in ParameterGrid(self.param_grid)
            for train_idx, test_idx in self.cv.split(X, y)
        ]

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_and_get_counts)(
                clone(self.estimator), X, y, train_idx, test_idx, params, self.pos_labels)
            for params, train_idx, test_idx in jobs
        )
        agg_results = defaultdict(lambda: np.zeros(3, dtype=float))
        for key, counts in results:
            agg_results[key] += counts
        agg_results = dict(agg_results)
        fbeta_scores = {}
        for key, (tp_sum, pred_sum, true_sum) in agg_results.items():
            precision = tp_sum / pred_sum if pred_sum > 0 else 0.0
            recall = tp_sum / true_sum if true_sum > 0 else 0.0
            beta2 = self.beta**2
            fbeta = (
                (1 + beta2) * precision * recall / (beta2 * precision + recall)
                if (precision + recall) > 0 else 0.0
            )
            fbeta_scores[key] = float(fbeta)
        best_key = max(fbeta_scores, key=fbeta_scores.get)
        best_score = fbeta_scores[best_key]
        best_params = dict(best_key)
        self.best_score_ = best_score
        self.best_params_ = best_params
        self.cv_results_ = fbeta_scores

        self.best_estimator_ = clone(self.estimator).set_params(**self.best_params_)
        if self.refit:
            self.best_estimator_.fit(X, y)
        return self

    def predict(self, X):
        return self.best_estimator_.predict(X)

    def predict_proba(self, X):
        return self.best_estimator_.predict(X)
