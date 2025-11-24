import numpy as np

from imblearn.metrics import sensitivity_specificity_support


def macro_informedness(y_true, y_pred, *, labels=None):
    sens, spec, _ = sensitivity_specificity_support(
        y_true, y_pred, labels=labels, average=None
    )
    return np.mean(sens + spec - 1)
