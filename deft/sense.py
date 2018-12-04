import os
import tempfile
import shutil
import numpy as np
from sklearn.base import clone
from sklearn.multiclass import _ConstantPredictor
from sklearn.model_selection import cross_val_score
from sklearn.utils import check_X_y
from sklearn.multiclass import check_classification_targets
from sklearn.utils.metaestimators import _safe_split
from joblib import Parallel, delayed
import logging

logger = logging.getLogger('sense')


def _ovo_cross_val_scores(mean_array,
                          std_array,
                          estimator, scoring, n_folds,
                          X, y,
                          i, j, class_i, class_j):
    indices, y = zip(*[(index, label) for index, label in
                       enumerate(y) if label == class_i or label == class_j])
    X = [X[index] for index in indices]

    y = np.array(y)
    y_binary = np.empty(y.shape, np.int)
    y_binary[y == class_i] = 0
    y_binary[y == class_j] = 1

    scores = cross_val_score(estimator, X, y, scoring=scoring,
                             cv=n_folds)
    mean_array[i, j] = np.mean(scores)
    std_array[i, j] = np.std(scores)


class SenseClusterer(object):
    def __init__(self, estimator, n_folds=5, n_jobs=1, scoring='neg_log_loss'):
        self.estimator = estimator
        self.n_folds = n_folds
        self.n_jobs = n_jobs
        self.scoring = scoring

    def fit(self, X, y):
        y = np.array(y)

        print('***')
        self.classes_ = np.unique(y)
        if len(self.classes_) == 1:
            raise ValueError("SenseClassifier cannot be fit when only one"
                             "class is present.")

        n_classes = self.classes_.shape[0]

        mean_folder = tempfile.mkdtemp()
        std_folder = tempfile.mkdtemp()
        mean_name = os.path.join(mean_folder, 'mean')
        std_name = os.path.join(std_folder, 'std')

        mean_array = np.memmap(mean_name, dtype=np.float64,
                               shape=(n_classes, n_classes), mode='w+')
        std_array = np.memmap(std_name, dtype=np.float64,
                              shape=(n_classes, n_classes), mode='w+')

        Parallel(n_jobs=self.n_jobs)(delayed(_ovo_cross_val_scores)
                                     (mean_array,
                                      std_array,
                                      self.estimator,
                                      self.scoring, self.n_folds,
                                      X, y,
                                      i, j,
                                      self.classes_[i],
                                      self.classes_[j])
                                     for i in range(n_classes)
                                     for j in range(i+1, n_classes))

        self.mean_grid = mean_array.copy()
        self.std_grd = std_array.copy()

        try:
            shutil.rmtree(mean_folder)
        except Exception:
            logger.info("Could not cleanup mean array automatically")
        try:
            shutil.rmtree(std_folder)
        except Exception:
            logger.info("Could not cleanup std array automatically")
