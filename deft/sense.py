import os
import tempfile
import shutil
import numpy as np
from sklearn.model_selection import cross_val_score
from joblib import Parallel, delayed
import logging

logger = logging.getLogger('sense')


def _ovo_cross_val_scores(mean_array,
                          estimator, n_folds,
                          X, y,
                          i, j, class_i, class_j):
    indices, y = zip(*[(index, label) for index, label in
                       enumerate(y) if label == class_i or label == class_j])
    X = [X[index] for index in indices]

    y = np.array(y)
    y_binary = np.empty(y.shape, np.int)
    y_binary[y == class_i] = 0
    y_binary[y == class_j] = 1

    scores = cross_val_score(estimator, X, y, scoring='roc_auc',
                             cv=n_folds)
    mean_array[i, j] = np.mean(scores)


class DocumentSimilarity(object):
    def __init__(self, estimator, n_folds=2, n_jobs=1):
        self.estimator = estimator
        self.n_folds = n_folds
        self.n_jobs = n_jobs

    def fit(self, X, y):
        y = np.array(y)

        self.classes_ = np.unique(y)
        if len(self.classes_) == 1:
            raise ValueError("Classifier cannot be fit when only one"
                             "class is present.")

        n_classes = self.classes_.shape[0]

        mean_folder = tempfile.mkdtemp()
        mean_name = os.path.join(mean_folder, 'mean')
        mean_array = np.memmap(mean_name, dtype=np.float64,
                               shape=(n_classes, n_classes), mode='w+')

        Parallel(n_jobs=self.n_jobs)(delayed(_ovo_cross_val_scores)
                                     (mean_array,
                                      self.estimator,
                                      self.n_folds,
                                      X, y,
                                      i, j,
                                      self.classes_[i],
                                      self.classes_[j])
                                     for i in range(n_classes)
                                     for j in range(i+1, n_classes))

        mean_grid = np.array(mean_array)
        mean_grid = mean_grid + mean_grid.T
        np.fill_diagonal(mean_grid, 1.0)
        mean_grid = 2*mean_grid - 1.0

        self.mean_grid = mean_grid

        try:
            shutil.rmtree(mean_folder)
        except Exception:
            logger.info("Could not cleanup mean array automatically")
