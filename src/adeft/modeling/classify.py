import gzip
import json
import numpy as np
from hashlib import md5
from importlib import import_module
from datetime import datetime

from imblearn.metrics import sensitivity_specificity_support
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.metaestimators import _safe_split

from adeft import __version__
from adeft.modeling.validate import PooledFbetaGridSearchCV
from adeft.nlp import english_stopwords
from adeft.util import load_array, serialize_array


class BaseModel(BaseEstimator, ClassifierMixin):
    """Abstract base class for Adeft's classification models."""
    def predict(self, X):
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

    def set_params(self, **params):
        new_params = {}
        for key, val in params.items():
            if key not in self._param_prefixes:
                raise ValueError(f"received invalid param {key}")
            setattr(self, key, val)
            new_params[f"{self._param_prefixes[key]}__{key}"] = val
        self.pipeline.set_params(**new_params)
        return self

    def get_params(self, deep=True):
        params = {}
        for key in self._param_prefixes:
            params[key] = getattr(self, key)
        if deep:
            # Add nested pipeline params with sklearn convention
            nested_params = self.pipeline.get_params(deep=True)
            params.update(nested_params)
        return params

    def get_model_info(self):
        raise NotImplementedError

    @classmethod
    def load_from_model_info(cls, model_info):
        raise NotImplementedError


class BaselineLogisticRegressionModel(BaseModel):
    """Implements the original Adeft model.

    Fits a logistic regression classifier with tfidf vectorized features.
    """
    _param_prefixes = {
        'ngram_range': 'tfidf',
        'max_features': 'tfidf',
        'stop_words': 'tfidf',
        'C': 'logit',
        'penalty': 'logit',
        'class_weight': 'logit',
        'random_state': 'logit',
        }
    def __init__(self, *, stop_words=None, ngram_range=(1, 2), C=100.0, penalty='l1',
                 max_features=1000, class_weight=None, random_state=None):
        self.C = C
        self.penalty = penalty
        self.class_weight = class_weight
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.stop_words = stop_words
        self.random_state = random_state
        self.pipeline = Pipeline([('tfidf',
                                   TfidfVectorizer(ngram_range=ngram_range,
                                                   max_features=max_features,
                                                   stop_words=stop_words)),
                                  ('logit',
                                   LogisticRegression(C=C,
                                                      solver='saga',
                                                      penalty=penalty,
                                                      random_state=random_state,
                                                      max_iter=1000))])
        self._feature_stds = None


    def fit(self, X, y, sample_weight=None):
        self.pipeline.fit(X, y, logit__sample_weight=sample_weight)
        tfidf = self.pipeline.named_steps['tfidf']
        # Feature standard deviations are computed in this way to avoid
        # unnecessary conversion to dense arrays.
        X_tfidf = tfidf.transform(X)
        temp = X_tfidf.copy()
        temp.data **= 2
        second_moment = temp.mean(0)
        first_moment_squared = np.square(X_tfidf.mean(0))
        result = second_moment - first_moment_squared
        self.classes_ = self.pipeline.classes_
        self._feature_stds = np.sqrt(np.squeeze(np.asarray(result)))
        return self

    def feature_importances(self):
        """Return feature importance scores for each label

        The feature importance scores are given by multiplying the coefficients
        of the logistic regression model by the standard deviations of the
        tf-idf scores for the associated features over all texts. Note that
        there is a coefficient associated to each label feature pair.

        One can interpret the feature importance score as the change in the
        linear predictor for a given label associated to a one standard
        deviation change in a feature's value. The predicted probability being
        given by the composition of the logit link function and the linear
        predictor.

        Returns
        -------
        dict
            Dictionary with class labels as keys. The associated values
            are lists of two element tuples each with first element an ngram
            feature and second element a feature importance score
        """
        if not hasattr(self, '_feature_stds') or self._feature_stds is None:
            logger.warning('Feature importance information not available for'
                           ' this model.')
            return None
        output = {}
        tfidf = self.pipeline.named_steps['tfidf']
        logit = self.pipeline.named_steps['logit']
        feature_names = tfidf.get_feature_names_out()
        classes = logit.classes_
        # Binary and multiclass cases most be handled separately
        # When there are greater than two classes, the logistic
        # regression model will have a row of coefficients for
        # each class. When there are only two classes, there is
        # only one row of coefficients corresponding to the label classes[1]
        if len(classes) > 2:
            for index, label in enumerate(classes):
                importance = np.round(
                    logit.coef_[index] * self._feature_stds, 4
                )
                output[label] = sorted(zip(feature_names, importance),
                                       key=lambda x: -x[1])
        else:
            importance = np.round(
                np.squeeze(logit.coef_) * self._feature_stds, 4
            )
            output[classes[1]] = sorted(zip(feature_names, importance),
                                        key=lambda x: -x[1])
            output[classes[0]] = [(feature, -value)
                                  for feature, value
                                  in output[classes[1]][::-1]]
        return output

    def get_model_info(self):
        """Return a JSON object representing a model for portability.

        Returns
        -------
        dict
            A JSON object representing the attributes of the classifier needed
            to make it portable/serializable and enabling its reload.
        """
        logit = self.pipeline.named_steps['logit']
        if not hasattr(logit, 'coef_'):
            raise RuntimeError('Estimator has not been fit.')
        classes_ = serialize_array(logit.classes_)
        intercept_ = serialize_array(logit.intercept_)
        coef_ = serialize_array(logit.coef_)

        tfidf = self.pipeline.named_steps['tfidf']
        vocabulary_ = {term: int(frequency)
                       for term, frequency in tfidf.vocabulary_.items()}
        idf_ = tfidf.idf_.tolist()
        ngram_range = tfidf.ngram_range
        stop_words = tfidf.stop_words
        model_info = {
            "logit": {
                "classes_": classes_,
                "intercept_": intercept_,
                "coef_": coef_,
                "C": self.C,
                "penalty": self.penalty,
                "class_weight": self.class_weight,

            },
            "tfidf": {
                "vocabulary_": vocabulary_,
                "idf_": idf_,
                "ngram_range": ngram_range,
                "stop_words": stop_words,
                "max_features": self.max_features,
            }
        }

        return model_info

    @classmethod
    def load_from_model_info(cls, model_info):
        tfidf = TfidfVectorizer(
            ngram_range=model_info["tfidf"].get("ngram_range", (1, 1)),
            stop_words=model_info["tfidf"].get("stop_words"),
        )
        tfidf.vocabulary_ = model_info["tfidf"]["vocabulary_"]
        tfidf.idf_ = np.asarray(model_info["tfidf"]["idf_"])
        logit = LogisticRegression(
            C=model_info["logit"].get("C", 1.0),
            penalty=model_info["logit"].get("penalty", "l2"),
            class_weight=model_info["logit"].get("class_weight"),
        )

        logit.intercept_ = load_array(model_info["logit"]["intercept_"])
        logit.coef_ = load_array(model_info["logit"]["coef_"])
        logit.classes_ = load_array(model_info["logit"]["classes_"])

        estimator = cls(
            stop_words=tfidf.stop_words,
            ngram_range=tfidf.ngram_range,
            C=logit.C,
            penalty=logit.penalty,
            max_features=tfidf.max_features,
            class_weight=logit.class_weight,
        )
        estimator.pipeline = Pipeline([("tfidf", tfidf), ("logit", logit)])

        return estimator


class AdeftClassifier:
    """Validates and trains classifiers to disambiguate shortforms

    By default, fits logistic regression models with tfidf vectorized ngram
    features. It is possible to use other types of model pipelines
    writing an estimator which conforms to the interface of
    py:class:`adeft.modeling.classify.BaseModel` defined above.

    Parameters
    ----------
    shortforms : str or list of str
        Shortform to disambiguate or list of shortforms to build models
        for multiple synomous shortforms.
    pos_labels : list of str
        Labels for positive classes. These correspond to the longforms of
        interest in an application. For adeft pretrained models these are
        typically genes and other relevant biological terms. Determines
        the positive labels for the macro-averaged F_beta metric used for
        model selection.
    estimator : Optional[py:class:`sklearn.base.BaseEstimator`]
        An sklearn api compatible estimator conforming to the API of
        py:class:`adeft.modeling.classify.BaseModel` defined above.
    random_state : Optional[int]
        Controls random number generation for cross_validation splits and
        in the estimator if the default estimator is used. Default: None

    Attributes
    ----------
    labels : ndarray|None
        A readonly property containing labels seen in training data in
        sorted order.
        
    validation_results : list[dict]|None
        A list of validation results found when running the ``validate``
        method. ``validation_results`` will be ``None`` if the ``validate``
        method has not been run. Validation employs nested cross validation,
        with inner splits used for model selection and outer splits used for
        validation.  There is an entry of the `validation_results` list for
        each outer cross validation split. Each ``dict`` entry has the
        following keys and associated values:

        "sensitivity" : ndarray
            Array of classwise sensitivity scores (True Positive Rate) on hold
            out set for each class in ``labels``. Classes are ordered in the same
            order they appear in ``labels``.
        "specificity" : ndarray
            Array of classwise specificity scores (True Negative Rate) on hold
            out set for each class in ``labels``. Classes are ordered in the same
            order they appear in ``labels``.
        "support" : ndarray
            Array of support values, counts for each class label in the hold out
            Classes are ordered in the same order they appear in ``labels``.
        "confusion_matrix": ndarray
            Full un-normalized confusion matrix with classes associated to rows
            and columns appearing in the order classes appear in ``labels``.

        Note that specific care is taken to report only metrics which do not
        depend on the class balance in unseen data. This is because the class
        balances in training data may not reflect balances in unseen data, due
        to training data being sampled from high precision, low recall longform
        expansion recognition, rather than randomly sampled from the population
        of relevant documents.
        
    other_metadata : dict
        Data set here by the user will be included when the model is serialized
        and remain available when the classifier is loaded again.
    version : str
        Adeft version used when model was fit
    timestamp : str
        Human readable timestamp for when model was fit
    training_set_digest : str
        Digest of training set calculated using md5 hash. Can be
        used at a glance to determine if two models used the same
        training set.
    """
    def __init__(self, shortforms, pos_labels, estimator=None, random_state=None):
        # handle case where single string is passed
        if isinstance(shortforms, str):
            shortforms = [shortforms]
        self.shortforms = shortforms
        self.pos_labels = pos_labels
        self.random_state = random_state
        if estimator is None:
            # Add shortforms to list of stopwords
            stop = list(
                set(english_stopwords).union([sf.lower() for sf
                                              in self.shortforms])
            )
            estimator = BaselineLogisticRegressionModel(
                stop_words=stop, random_state=random_state
            )
        self.estimator = estimator
        self.other_metadata = None

        self.validation_results = None
        self.version = __version__
        self.timestamp = None
        self.training_set_digest = None

    @property
    def params(self):
        return self.estimator.get_params()

    @property
    def labels(self):
        if hasattr(self.estimator, "classes_"):
            return self.estimator.classes_
        return None

    def grid_search_to_select_model(
            self, X, y, param_grid, *, cv=None, refit=False, n_jobs=1
    ):
        """Perform a grid search to select a model

        Model selection is based on a pooled macro averaged F1 score as
        recommended in the classic Apples-to-Apples paper by Forman and
        Scholz. With positive labels determined by the ``pos_labels``.
        F1 is used for model selection because it is of no concern whether
        negatively labeled examples are identified correctly, so long as
        none are mislabled with the positive label. Negatively labeled
        examples correspond to entities which should never appear in
        biomolecular interactions.

        F1 score is not reported in validation results because the class
        balance in data labeled through Adeft's Acromine-like algorithm
        does not necessarily reflect the class balance in the true population
        of documents where a shortform is used. It is still useful however
        for model selection purposes in the absence of more specific
        misclassification costs which could be used for cost-sensitive
        learning.

        George Forman and Martin Scholz. 2010. Apples-to-apples in
        cross-validation studies: pitfalls in classifier performance
        measurement. SIGKDD Explor. Newsl. 12, 1 (June 2010), 49–57.
        https://doi.org/10.1145/1882471.1882479
        """
        
        if cv is None:
            cv = StratifiedKFold(
                n_splits=5, shuffle=True, random_state=self.random_state
            )
        estimator = clone(self.estimator)
        grid_search = PooledFbetaGridSearchCV(
            estimator, param_grid, pos_labels=self.pos_labels, cv=cv,
            refit=refit, n_jobs=n_jobs
        )
        grid_search.fit(X, y)
        estimator = grid_search.best_estimator_
        return (
            estimator,
            grid_search.best_score_,
            grid_search.cv_results_,
            grid_search.best_params_,
        )

    def train(self, X, y, **params):
        """Fits a disambiguation model

        Parameters
        ----------
        texts : iterable of str
            Training texts
        y : iterable of str
            True labels for training texts
        **params :
            Parameter values for estimator.

        """
        # Initialize pipeline
        self.estimator.set_params(**params)
        self.estimator.fit(X, y)
        self.timestamp = self._get_current_time()
        self.training_set_digest = self._training_set_digest(X)

    def validate(
            self,
            X,
            y,
            *,
            param_grid,
            n_outer_splits=5,
            n_inner_splits=5,
            n_jobs=1,
            refit=False,
    ):
        """Validate disambiguation model using nested cross validation.

        Model selection is performed in inner splits using pooled macro
        averaged F1 score. Validation results are reported on outer splits.
        Running this method will cause the attribute ``validation_results``
        to be filled with a list of validation results as described in the
        class level docstring for ``AdeftClassifier``.

        Parameters
        ----------
        X : iterable of str
            Training texts
        y : iterable of str
            True labels for training texts
        param_grid : dict
            Parameter grid for estimator, in form expected by Scikit_learn's
            ``GridSearchCV`` (Adeft's ``PooledFbetaGridSearchCV`` is what is
            actually used internally, but it's API is essentially the same as
            ``GridSearchCV``).
        n_outer_splits : Optional[int]
            Number of outer splits to use in nested cross validation.
            Default: 5
        n_inner_splits : Optional[int]
            Number of inner splits to use in nested cross validation.
            Default: 5
        refit : Optional[bool]
            If True, perform model selection on the full dataset and refit
            the model with the best parameters found. Default: False

        Notes
        -----
        Calling validate will also create an attribute
        ``inner_model_selection_results_``, containing a list of dictionaries
        for each outer CV split. Each dictionary contains keys
        ``"best_score"``, ``"best_params"``, and ``"cv_results"`` containing
        the corresponding attributes of the ``PooledFbetaGridSearchCv`` object.
        If ``refit=True``, then another attribute
        ``outer_model_selection_results_`` will be created, containing a single
        dictionary of model selection results for the final round of model
        selection.
        """
        outer_splitter = StratifiedKFold(
            n_splits=n_outer_splits, shuffle=True, random_state=self.random_state
        )
        inner_splitter = StratifiedKFold(
            n_splits=n_inner_splits, shuffle=True, random_state=self.random_state
        )
        splits = outer_splitter.split(X, y)
        validation_results = []
        model_selection_results = []
        labels = np.unique(y)
        for i, (train_idx, test_idx) in enumerate(splits):
            X_train, y_train = _safe_split(self.estimator, X, y, train_idx)
            # TODO: allow custom parameter tuning strategies to be passed in,
            # rather than just hardcoding in a single round of grid search.
            estimator, best_score, cv_results, best_params = self.grid_search_to_select_model(
                X_train, y_train, param_grid, cv=inner_splitter, refit=True,
                n_jobs=n_jobs
            )
            X_test, y_test = _safe_split(self.estimator, X, y, test_idx)
            preds = estimator.predict(X_test)
            confusion = confusion_matrix(
                y_test, preds, labels=labels
            )
            sens, spec, support = sensitivity_specificity_support(
                y_test, preds, labels=labels, average=None
            )
            validation_results.append({
                "sensitivity": sens,
                "specificity": spec,
                "support": support,
                "confusion_matrix": confusion,
            })
            model_selection_results.append({
                "best_score": best_score,
                "best_params": best_params,
                "cv_results": cv_results,
            })
        self.validation_results = validation_results
        self.inner_model_selection_results_ = model_selection_results
        if refit:
            _, best_score, cv_results, best_params = self.grid_search_to_select_model(
                X, y, param_grid, cv=outer_splitter, refit=False, n_jobs=n_jobs
            )
            self.train(X, y, **best_params)
            self.outer_model_selection_results_ = {
                "best_score": best_score,
                "best_params": best_params,
                "cv_results": cv_results,
            }
        return self

    def predict_proba(self, texts):
        """Predict class probabilities for a list-like of texts"""
        labels = self.estimator.pipeline.classes_
        preds = self.estimator.predict_proba(texts)
        return [{labels[i]: prob for i, prob in enumerate(probs)}
                for probs in preds]

    def predict(self, texts):
        """Predict class labels for a list-like of texts"""
        return self.estimator.predict(texts)

    def get_model_info(self):
        """Return a JSON object representing a model for portability.

        Returns
        -------
        dict
            A JSON object representing the attributes of the classifier needed
            to make it portable/serializable and enabling its reload.
        """

        model_info = {"estimator_info": self.estimator.get_model_info()}
        estimator_class = self.estimator.__class__
        model_info["estimator_module"] = estimator_class.__module__
        model_info["estimator_name"] = estimator_class.__qualname__

        model_info.update(
            {
                "shortforms": self.shortforms,
                "pos_labels": self.pos_labels,
            }
        )
        model_info["validation_results"] = {
            fold_id: {key: val.tolist() for key, val in entry.items()}
            for fold_id, entry in self.validation_results.items()
        }
        model_info["version"] = self.version
        model_info["timestamp"] = self.timestamp
        model_info["training_set_digest"] = self.training_set_digest
        model_info["other_metadata"] = self.other_metadata

        return model_info

    @classmethod
    def load_from_model_info(cls, model_info):
        shortforms = model_info['shortforms']
        pos_labels = model_info['pos_labels']
        estimator_info = model_info["estimator_info"]
        estimator_module = import_module(model_info["estimator_module"])
        estimator_class = getattr(estimator_module, model_info["estimator_name"])
        estimator = estimator_class.load_from_model_info(estimator_info)
        longform_model = cls(
            shortforms=shortforms, pos_labels=pos_labels, estimator=estimator
        )
        longform_model.estimator = estimator
        validation_results = model_info["validation_results"]
        longform_model.validation_results = {
            fold_id: {key: np.asarray(val) for key, val in entry.items()}
            for fold_id, entry in validation_results.items()
        }
        return longform_model

    def dump_model(self, filepath):
        """Serialize model to gzipped json

        Parameters
        ----------
        filepath : str
           Path to output file
        """
        model_info = self.get_model_info()
        json_str = json.dumps(model_info)
        json_bytes = json_str.encode('utf-8')
        with gzip.GzipFile(filepath, 'w') as fout:
            fout.write(json_bytes)

    @classmethod
    def load_model(cls, filepath):
        with gzip.GzipFile(filepath, 'r') as fin:
            json_bytes = fin.read()
        json_str = json_bytes.decode('utf-8')
        model_info = json.loads(json_str)
        return cls.load_from_model_info(model_info)

    def feature_importances(self):
        """Return feature importance scores for each label."""
        return self.estimator.feature_importances()

    def _get_current_time(self):
        unix_timestamp = datetime.now().timestamp()
        return datetime.fromtimestamp(unix_timestamp).isoformat()

    def _training_set_digest(self, texts):
        """Returns a hash corresponding to training set

        Does not depend on order of texts
        """
        hashed_texts = ''.join(md5(text.encode('utf-8')).hexdigest()
                               for text in sorted(texts))
        return md5(hashed_texts.encode('utf-8')).hexdigest()
