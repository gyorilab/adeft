from collections import defaultdict

from nltk.tokenize import regexp_tokenize
from nltk.stem.snowball import EnglishStemmer


class WatchfulStemmer(object):
    """Wraps the nltk.snow EnglishStemmer.

    Keeps track of the number of times words have been mapped to particular
    stems by the wrapped stemmer. Extraction of longforms works with stemmed
    tokens but it is necessary to recover actual words from stems.

    Attributes
    ----------
    __snow : :py:class:`nltk.stem.snowball.EnglishStemmer`

    counts : defaultdict of defaultdict of int
        Contains the count of the number of times a particular word has been
        mapped to from a particular stem by the wrapped stemmer. Of the form
        counts[stem:str][word:str] = count:int
    """
    def __init__(self):
        self.__snowball = EnglishStemmer()
        self.counts = defaultdict(lambda: defaultdict(int))

    def stem(self, word):
        """Returns stemmed form of word.

        Adds one to count associated to the computed stem, word pair.

        Parameters
        ----------
        word : str
            text to stem

        Returns
        -------
        stemmed : str
            stemmed form of input word
        """
        stemmed = self.__snowball.stem(word)
        self.counts[stemmed][word] += 1
        return stemmed

    def most_frequent(self, stemmed):
        """Return the most frequent word mapped to a given stem

        Parameters
        ----------
        stemmed : str
            Stem that has previously been output by the wrapped snowball
            stemmer.

        Returns
        -------
        output : str|None
            Most frequent word that has been mapped to the input stem or None
            if the wrapped stemmer has never mapped the a word to the input
            stem. Break ties with lexicographic order
        """
        words = list(self.counts[stemmed].items())
        if words:
            words.sort(key=lambda x: x[1], reverse=True)
            candidates = [word[0] for word in words if word[1] == words[0][1]]
            output = min(candidates)
        else:
            raise ValueError('stem %s has not been observed' % stemmed)
        return output


def word_tokenize(text):
    """Custom word-tokenizer based on regular expression pattern

    Everything that is not a block of alphanumeric characters is considered as
    a separate token
    """
    return regexp_tokenize(text, r'\w+|[^\s\w]')
