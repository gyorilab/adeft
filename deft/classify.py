import json
import logging
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, make_scorer


logger = logging.getLogger('classify')


class LongformClassifier(object):
    def __init__(self, texts, y, shortform, pos_labels):
        self.texts = texts
        self.y = y
        self.shortform = shortform
        self.pos_labels = pos_labels

    def train(self, params=None, n_jobs=1, cv=5):
        logit_pipeline = Pipeline([('tfidf',
                                    TfidfVectorizer(ngram_range=(1, 2),
                                                    stop_words='english')),
                                   ('logit',
                                    LogisticRegression(solver='saga',
                                                       penalty='l1',
                                                       multi_class='auto'))])

        if params is None:
            params = {'logit__C': (1.0,),
                      'tfidf__max_features': (10000,)}

        if len(set(self.y)) > 2:
            average = 'weighted'
        else:
            average = 'binary'
        f1_scorer = make_scorer(f1_score, labels=self.pos_labels,
                                average=average)

        logger.info('Beginning grid search in parameter space:\n'
                    f"(C={params['logit__C']})\n"
                    f"(max_features={params['tfidf__max_features']})")

        grid_search = GridSearchCV(logit_pipeline, params,
                                   cv=cv, n_jobs=n_jobs, scoring=f1_scorer)
        grid_search.fit(self.texts, self.y)
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
