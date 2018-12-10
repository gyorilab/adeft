import gzip
import json
import numpy as np
import logging
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, make_scorer


logger = logging.getLogger('classify')


class LongformClassifier(object):
    def __init__(self, shortform, pos_labels):
        self.shortform = shortform
        self.pos_labels = pos_labels

    def train(self, texts, y, params=None, n_jobs=1, cv=5):
        logit_pipeline = Pipeline([('tfidf',
                                    TfidfVectorizer(ngram_range=(1, 2),
                                                    stop_words='english')),
                                   ('logit',
                                    LogisticRegression(solver='saga',
                                                       penalty='l1',
                                                       multi_class='auto'))])

        if params is None:
            params = {'logit__C': (1.0,),
                      'tfidf__max_features': (1000,)}

        if len(set(y)) > 2:
            f1_scorer = make_scorer(f1_score, labels=self.pos_labels,
                                    average='weighted')
        else:
            f1_scorer = make_scorer(f1_score, pos_label=self.pos_labels[0],
                                    average='binary')

        logger.info('Beginning grid search in parameter space:\n'
                    f"(C={params['logit__C']})\n"
                    f"(max_features={params['tfidf__max_features']})")

        grid_search = GridSearchCV(logit_pipeline, params,
                                   cv=cv, n_jobs=n_jobs, scoring=f1_scorer)
        grid_search.fit(texts, y)
        logger.info(f'Best f1 score of {grid_search.best_score_} found for'
                    f' parameter values:\n{grid_search.best_params_}')
        self.estimator = grid_search.best_estimator_
        params = grid_search.best_params_
        self.C = params['logit__C']
        self.max_features = params['tfidf__max_features']

    def predict_proba(self, texts):
        return self.estimator.predict_proba(texts)

    def predict(self, texts):
        return self.estimator.predict(texts)

    def dump(self, filepath):
        logit = self.estimator.named_steps['logit']
        classes_ = logit.classes_.tolist()
        intercept_ = logit.intercept_.tolist()
        coef_ = logit.coef_.tolist()

        tfidf = self.estimator.named_steps['tfidf']
        vocabulary_ = {term: int(frequency)
                       for term, frequency in tfidf.vocabulary_.items()}
        idf_ = tfidf.idf_.tolist()

        model_info = {'logit': {'classes_': classes_,
                                'intercept_': intercept_,
                                'coef_': coef_},
                      'tfidf': {'vocabulary_': vocabulary_,
                                'idf_': idf_}}
        json_str = json.dumps(model_info)
        json_bytes = json_str.encode('utf-8')

        with gzip.GzipFile(filepath, 'w') as fout:
            fout.write(json_bytes)

    def load(self, filepath):
        with gzip.GzipFile(filepath, 'r') as fin:
            json_bytes = fin.read()
        json_str = json_bytes.decode('utf-8')
        model_info = json.loads(json_str)

        tfidf = TfidfVectorizer(ngram_range=(1, 2))
        logit = LogisticRegression(solver='saga',
                                   penalty='l1',
                                   multi_class='auto')
        tfidf.vocabulary_ = model_info['tfidf']['vocabulary_']
        tfidf.idf_ = model_info['tfidf']['idf_']

        logit.classes_ = np.array(model_info['logit']['classes_'])
        logit.intercept_ = np.array(model_info['logit']['intercept_'])
        logit.coef_ = np.array(model_info['logit']['coef_'])

        self.estimator = Pipeline([('tfidf', tfidf),
                                   ('logit', logit)])
