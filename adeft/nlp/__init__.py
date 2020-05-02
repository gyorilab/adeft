"""Implements a set of natural language processing tools used to pre-process
text used for finding candidate longforms, recognizing defining patterns,
and learning classification models.

stopwords_min contains a small collection of stopwords for use in the
alignment based :py:class:`AdeftLongformScorer`. english_stopwords contains
a larger collection of stopwords for use in classification and anomaly
detection models.

"""

from .nlp import (stem, WatchfulStemmer, word_tokenize, word_detokenize,
                  stopwords_min, english_stopwords)
