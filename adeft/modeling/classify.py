import gzip
import json
import logging
import warnings
import numpy as np
from hashlib import md5
from datetime import datetime
from collections import Counter, defaultdict

from sklearn.pipeline import Pipeline
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score,\
    make_scorer


from adeft import __version__
from adeft.nlp import english_stopwords

warnings.filterwarnings("ignore", category=ConvergenceWarning)

logger = logging.getLogger(__file__)


class AdeftClassifier(object):
    """Trains classifiers to disambiguate shortforms based on context

    Fits logistic regression models with tfidf vectorized ngram features.
    Uses sklearns LogisticRegression and TfidfVectorizer classes.
    Models can be serialized and loaded for later use.

    Parameters
    ----------
    shortforms : str or list of str
        Shortform to disambiguate or list of shortforms to build models
        for multiple synomous shortforms.
    pos_labels : list of str
        Labels for positive classes. These correspond to the longforms of
        interest in an application. For adeft pretrained models these are
        typically genes and other relevant biological terms.

    random_state : Optional[int]
        Optional specification of seed used when calculating crossvalidation
        folds and fitting the logistic regression model. Default: None

    Attributes
    ----------
    estimator : py:class:`sklearn.pipeline.Pipeline`
        An sklearn pipeline that transforms text data with a TfidfVectorizer
        and fits a logistic regression.
    stats : dict
       Statistics describing model performance. Only available after model is
       fit with crossvalidation
    stop : list of str
        List of stopwords to exclude when performing tfidf vectorization.
        These consist of the set of stopwords in adeft.nlp.english_stopwords
        along with the shortform(s) for which the model is being built
    params : dict
        Dictionary mapping parameters to their values. If fit with cv, this
        contains the parameters with best micro averaged f1 score over
        crossvalidation runs.
    best_score : float
        Best micro averaged f1 score for positive labels over crossvalidation
        runs. This information can also be found in the stats dict and is not
        included when models are serialized. Only available if model is fit
        with the cv method.
    grid_search : py:class:`sklearn.model_selection.GridSearchCV`
        sklearn gridsearch object if model was fit with cv. This is not
        included when model is serialized.
    confusion_info : dict
        Contains the confusion matrix for each pair of labels per
        crossvalidation split. Only available if the model has been fit with
        crossvalidation. Nested dictionary,
        `confusion_info[label1][label2][i]` gives the number of test examples
        where the true label is label1 and the classifier has made prediction
        label2 in split i.
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
    _std : py:class:`numpy.ndarray`
        Array of standard deviations of feature values over training
        set. This is used to calculate feature importance
    """
    def __init__(self, shortforms, pos_labels, random_state=None):
        # handle case where single string is passed
        if isinstance(shortforms, str):
            shortforms = [shortforms]
        self.shortforms = shortforms
        self.pos_labels = pos_labels
        self.random_state = random_state
        self.estimator = None
        self.stats = None
        self.confusion_info = None
        self.other_metadata = None
        # Add shortforms to list of stopwords
        self.stop = set(english_stopwords).union([sf.lower() for sf
                                                  in self.shortforms])
        self.best_score = None
        self.grid_search = None
        self.version = __version__
        self._std = None
        self.params = None
        self.timestamp = None
        self.training_set_digest = None

    def train(self, texts, y, C=1.0, ngram_range=(1, 2), max_features=1000,
              class_weight=None):
        """Fits a disambiguation model

        Parameters
        ----------
        texts : iterable of str
            Training texts
        y : iterable of str
            True labels for training texts
        C : Optional[float]
             L1 regularization parameter logistic regression model. Follows
             convention of support vector machines with smaller values
             corresponding to stronger regularization. Default: 1.0
        ngram_range : Optional[tuple of int]
            Range of ngram features to use. Must be a tuple of ints of the
            form (a, b) with a <= b. When ngram_range is (1, 2), unigrams and
            bigrams will be used as features. Default: (1, 2)
        max_features : int
            Maximum number of tfidf-vectorized ngrams to use as features in
            model. Selects top_features by term frequency Default: 1000
        class_weight : Optional[dict or 'balanced']
            Weights associated with classes in the form {class_label:
            weight}. If not given, all classes are supposed to have weight one.

            The “balanced” mode uses the values of y to automatically adjust
            weights inversely proportional to class frequencies in the input
            data as n_samples / (n_classes * np.bincount(y)).

            Note that these weights will be multiplied with sample_weight
            (passed through the fit method) if sample_weight is specified.
        """
        # Initialize pipeline
        seed = self.random_state
        logit_pipeline = \
            Pipeline([('tfidf',
                       TfidfVectorizer(ngram_range=ngram_range,
                                       max_features=max_features,
                                       stop_words=self.stop)),
                      ('logit',
                       LogisticRegression(C=C,
                                          solver='saga',
                                          penalty='l1',
                                          multi_class='auto',
                                          class_weight=class_weight,
                                          random_state=seed))])
        logit_pipeline.fit(texts, y)

        self.params = {'C': C, 'ngram_grange': ngram_range,
                       'max_features': max_features,
                       'class_weight': class_weight,
                       'random_state': self.random_state}
        self.estimator = logit_pipeline
        self.best_score = None
        self.grid_search = None
        self.timestamp = self._get_current_time()
        self.training_set_digest = self._training_set_digest(texts)
        self._set_variance(texts)

    def cv(self, texts, y, param_grid, n_jobs=1, cv=5):
        """Performs grid search to select and fit a disambiguation model

        Parameters
        ----------
        texts : iterable of str
             Training texts
        y : iterable of str
            True labels for the training texts
        param_grid : Optional[dict]
          Grid search parameters. Can contain all parameters from the train
          method.
        n_jobs : Optional[int]
            Number of jobs to use when performing grid_search
            Default: 1
        cv : Optional[int]
            Number of folds to use in crossvalidation. Default: 5

        Example
        -------
        >>> params = {'C': [1.0, 10.0, 100.0],
        ...    'max_features': [3000, 6000, 9000],
        ...    'ngram_range': [(1, 1), (1, 2), (1, 3)]}
        >>> classifier = LongformClassifier('IR', ['insulin receptor'])
        >>> classifier.train(texts, labels, param_grid=params, n_jobs=4)
        """
        # Initialize pipeline
        seed = self.random_state
        logit_pipeline = Pipeline([('tfidf',
                                    TfidfVectorizer(ngram_range=(1, 2),
                                                    max_features=1000,
                                                    stop_words=self.stop)),
                                   ('logit',
                                    LogisticRegression(C=100.,
                                                       solver='saga',
                                                       penalty='l1',
                                                       multi_class='auto',
                                                       random_state=seed))])

        # Create scorer for use in grid search. Best params decided using
        # f1 score. The positive labels are specified when the classifier is
        # initialized. Uses micro-average f1, precision, and recall scores.
        # This means metrics are calculated globally by counting all true
        # positives, false negatives, and false positives
        f1_scorer = make_scorer(f1_score, labels=self.pos_labels,
                                average='micro')
        pr_scorer = make_scorer(precision_score,
                                labels=self.pos_labels,
                                average='micro')
        rc_scorer = make_scorer(recall_score,
                                labels=self.pos_labels,
                                average='micro')

        scorer = {'f1': f1_scorer,
                  'pr': pr_scorer,
                  'rc': rc_scorer}
        all_labels = sorted(set(y))
        for label in all_labels:
            f1 = make_scorer(f1_score, labels=[label], average=None)
            pr = make_scorer(recall_score, labels=[label], average=None)
            rc = make_scorer(precision_score, labels=[label], average=None)
            scorer.update({'f1_%s' % label: f1,
                           'pr_%s' % label: pr,
                           'rc_%s' % label: rc})
        for label1 in all_labels:
            for label2 in all_labels:
                count_score = make_scorer(_count_score, label1=label1,
                                          label2=label2)
                scorer['count_%s_%s' % (label1, label2)] = count_score
        logger.info('Beginning grid search in parameter space:\n'
                    '%s' % param_grid)

        param_mapping = {'C': 'logit__C',
                         'class_weight': 'logit__class_weight',
                         'max_features': 'tfidf__max_features',
                         'ngram_range':  'tfidf__ngram_range'}
        inverse_param_mapping = {value: key
                                 for key, value in param_mapping.items()}

        param_grid = {param_mapping[key]: value
                      for key, value in param_grid.items()}
        num_splits = cv
        cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
        # Fit grid_search and set the estimator for the instance of the class
        grid_search = GridSearchCV(logit_pipeline, param_grid,
                                   cv=cv, n_jobs=n_jobs, scoring=scorer,
                                   refit='f1',
                                   return_train_score=False)
        grid_search.fit(texts, y)
        logger.info('Best f1 score of %s found for' % grid_search.best_score_
                    + ' parameter values:\n%s' % grid_search.best_params_)

        cv = grid_search.cv_results_
        best_index = cv['rank_test_f1'][0] - 1
        labels = dict(Counter(y))
        stats = {'label_distribution': labels,
                 'f1': {'mean':
                        np.round(cv['mean_test_f1'][best_index], 6),
                        'std':
                        np.round(cv['std_test_f1'][best_index], 6)},
                 'precision': {'mean':
                               np.round(cv['mean_test_pr']
                                        [best_index], 6),
                               'std': np.round(cv['std_test_pr']
                                               [best_index], 6)},
                 'recall': {'mean': np.round(cv['mean_test_rc']
                                             [best_index], 6),
                            'std': np.round(cv['std_test_rc']
                                            [best_index], 6)}}
        for label in all_labels:
            stats.update({label:
                          {'f1':
                           {'mean': np.round(cv['mean_test_f1_%s'
                                                % label][best_index], 6),
                            'std': np.round(cv['std_test_f1_%s'
                                               % label][best_index], 6)},
                           'pr':
                           {'mean': np.round(cv['mean_test_pr_%s'
                                                % label][best_index], 6),
                            'std': np.round(cv['std_test_pr_%s'
                                               % label][best_index], 6)},
                           'rc':
                           {'mean': np.round(cv['mean_test_rc_%s'
                                                % label][best_index], 6),
                            'std': np.round(cv['std_test_rc_%s'
                                               % label][best_index], 6)}}})

        confusion = defaultdict(lambda: defaultdict(list))
        for label1 in all_labels:
            for label2 in all_labels:
                for i in range(num_splits):
                    key = 'split%s_test_count_%s_%s' % (i, label1, label2)
                    val = int(cv[key][best_index])
                    confusion[label1][label2].append(val)
        confusion = {key: dict(value) for key, value in confusion.items()}
        params = {inverse_param_mapping[key]: value for key, value
                  in grid_search.best_params_.items()}
        params['random_state'] = self.random_state
        self.params = params
        self.estimator = grid_search.best_estimator_
        self.best_score = grid_search.best_score_
        self.grid_search = grid_search
        self.stats = stats
        self.confusion_info = confusion
        self.timestamp = self._get_current_time()
        self.training_set_digest = self._training_set_digest(texts)
        self._set_variance(texts)

    def predict_proba(self, texts):
        """Predict class probabilities for a list-like of texts"""
        labels = self.estimator.classes_
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
        logit = self.estimator.named_steps['logit']
        if not hasattr(logit, 'coef_'):
            raise RuntimeError('Estimator has not been fit.')
        classes_ = logit.classes_.tolist()
        intercept_ = logit.intercept_.tolist()
        coef_ = logit.coef_.tolist()

        tfidf = self.estimator.named_steps['tfidf']
        vocabulary_ = {term: int(frequency)
                       for term, frequency in tfidf.vocabulary_.items()}
        idf_ = tfidf.idf_.tolist()
        ngram_range = tfidf.ngram_range
        model_info = {'logit': {'classes_': classes_,
                                'intercept_': intercept_,
                                'coef_': coef_},
                      'tfidf': {'vocabulary_': vocabulary_,
                                'idf_': idf_,
                                'ngram_range': ngram_range},
                      'shortforms': self.shortforms,
                      'pos_labels': self.pos_labels}
        # Model statistics may not be available depending on
        # how the model was fit
        if hasattr(self, 'stats') and self.stats is not None:
            model_info['stats'] = self.stats
        # These attributes may not exist in older models
        if hasattr(self, '_std') and self._std is not None:
            model_info['std'] = self._std.tolist()
        if hasattr(self, 'timestamp') and self.timestamp is not None:
            model_info['timestamp'] = self.timestamp
        if hasattr(self, 'training_set_digest') and \
           self.training_set_digest is not None:
            model_info['training_set_digest'] = self.training_set_digest
        if hasattr(self, 'params') and self.params is not None:
            model_info['params'] = self.params
        if hasattr(self, 'version') and self.version is not None:
            model_info['version'] = self.version
        if hasattr(self, 'confusion_info') and self.confusion_info is not None:
            model_info['confusion_info'] = self.confusion_info
        if hasattr(self, 'other_metadata') and self.other_metadata is not None:
            model_info['other_metadata'] = self.other_metadata
        return model_info

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
        if not hasattr(self, '_std') or self._std is None:
            logger.warning('Feature importance information not available for'
                           ' this model.')
            return None
        output = {}
        tfidf = self.estimator.named_steps['tfidf']
        logit = self.estimator.named_steps['logit']
        feature_names = tfidf.get_feature_names()
        classes = logit.classes_
        # Binary and multiclass cases most be handled separately
        # When there are greater than two classes, the logistic
        # regression model will have a row of coefficients for
        # each class. When there are only two classes, there is
        # only one row of coefficients corresponding to the label classes[1]
        if len(classes) > 2:
            for index, label in enumerate(classes):
                importance = np.round(logit.coef_[index] * self._std, 4)
                output[label] = sorted(zip(feature_names, importance),
                                       key=lambda x: -x[1])
        else:
            importance = np.round(np.squeeze(logit.coef_) * self._std, 4)
            output[classes[1]] = sorted(zip(feature_names, importance),
                                        key=lambda x: -x[1])
            output[classes[0]] = [(feature, -value)
                                  for feature, value
                                  in output[classes[1]][::-1]]
        return output

    def _set_variance(self, texts):
        """Set attribute containing array of variances for features

        Parameters
        __________
        texts : iterable of str
            Training texts
        """
        tfidf = self.estimator.named_steps['tfidf']
        X = tfidf.transform(texts)
        temp = X.copy()
        temp.data **= 2
        second_moment = temp.mean(0)
        first_moment_squared = np.square(X.mean(0))
        result = second_moment - first_moment_squared
        self._std = np.sqrt(np.squeeze(np.asarray(result)))

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


def load_model(filepath):
    """Load previously serialized model

    Parameters
    ----------
    filepath : str
       path to model file

    Returns
    -------
    longform_model : py:class:`adeft.classify.AdeftClassifier`
        The classifier that was loaded from the given path.
    """
    with gzip.GzipFile(filepath, 'r') as fin:
        json_bytes = fin.read()
    json_str = json_bytes.decode('utf-8')
    model_info = json.loads(json_str)
    return load_model_info(model_info)


def load_model_info(model_info):
    """Return a longform model from a model info JSON object.

    Parameters
    ----------
    model_info : dict
        The JSON object containing the attributes of a model.

    Returns
    -------
    longform_model : py:class:`adeft.classify.AdeftClassifier`
        The classifier that was loaded from the given JSON object.
    """
    shortforms = model_info['shortforms']
    pos_labels = model_info['pos_labels']
    longform_model = AdeftClassifier(shortforms=shortforms,
                                     pos_labels=pos_labels)
    ngram_range = model_info['tfidf']['ngram_range']
    tfidf = TfidfVectorizer(ngram_range=ngram_range,
                            stop_words='english')
    logit = LogisticRegression(multi_class='auto')

    tfidf.vocabulary_ = model_info['tfidf']['vocabulary_']
    tfidf.idf_ = model_info['tfidf']['idf_']
    logit.classes_ = np.array(model_info['logit']['classes_'],
                              dtype='<U64')
    logit.intercept_ = np.array(model_info['logit']['intercept_'])
    logit.coef_ = np.array(model_info['logit']['coef_'])

    estimator = Pipeline([('tfidf', tfidf),
                          ('logit', logit)])
    longform_model.estimator = estimator
    # These attributes do not exist in older adeft models.
    # For backwards compatibility we check if they are present
    if 'stats' in model_info:
        longform_model.stats = model_info['stats']
    if 'std' in model_info:
        longform_model._std = np.array(model_info['std'])
    if 'timestamp' in model_info:
        longform_model.timestamp = model_info['timestamp']
    if 'training_set_digest' in model_info:
        longform_model.training_set_digest = model_info['training_set_digest']
    if 'params' in model_info:
        longform_model.params = model_info['params']
    if 'version' in model_info:
        longform_model.version == model_info['version']
    if 'confusion_info' in model_info:
        longform_model.confusion_info = model_info['confusion_info']
    if 'other_metadata' in model_info:
        longform_model.other_metadata = model_info['other_metadata']
    return longform_model


def _count_score(y_true, y_pred, label1=0, label2=1):
    return sum((y == label1 and pred == label2)
               for y, pred in zip(y_true, y_pred))
