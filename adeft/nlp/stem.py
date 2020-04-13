from collections import defaultdict

from nltk.stem.snowball import EnglishStemmer

from adeft.nlp.resources import greek_alphabet, greek_to_latin


_stemmer = EnglishStemmer()


def stem(word):
    """Return stem of word

    Stemming attempts to reduce words to their root form in a
    crude heuristic way. We add an attional heuristic of stripping
    a terminal s if the preceding character is upper case. These
    often denote pluralization in biology (e.g. RNAs)

    Parameters
    ----------
    word : str

    Returns
    str
        stem of input word converted to lower case
    """
    if len(word) > 1 and word[-2].isupper() and word[-1] == 's':
        updated_word = word[:-1]
    else:
        updated_word = word
    return _stemmer.stem(updated_word).lower()


class WatchfulStemmer(object):
    """Wraps the nltk.snow EnglishStemmer.

    Keeps track of the number of times words have been mapped to particular
    stems by the wrapped stemmer. Extraction of longforms works with stemmed
    tokens but it is necessary to recover actual words from stems.

    Parameters
    ----------
    counts : Optional[dict]
        counts dictionary as used internally in WatchfulStemmer. Allows for
        loading a previously saved WatchfulStemmer

    Attributes
    ----------
    __snowball : :py:class:`nltk.stem.snowball.EnglishStemmer`

    counts : defaultdict of defaultdict of int
        Contains the count of the number of times a particular word has been
        mapped to from a particular stem by the wrapped stemmer. Of the form
        counts[stem:str][word:str] = count:int
    """
    def __init__(self, counts=None):
        if counts is None:
            counts = {}
        self.counts = defaultdict(lambda: defaultdict(int),
                                  {key: defaultdict(int, value)
                                   for key, value in counts.items()})

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
        stemmed = stem(word)
        self.counts[stemmed][word.lower()] += 1
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
        output : str or None
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

    def dump(self):
        """Returns dictionary of info needed to reconstruct stemmer"""
        return dict(self.counts)


def greek_aware_stem(text):
    out = stem(text)
    out = _expand_greek_unicode(out)
    out = _replace_greek_latin(out)
    return out.lower()


def _expand_greek_unicode(text):
    for greek_uni, greek_spelled_out in greek_alphabet.items():
        text = text.replace(greek_uni, greek_spelled_out)
    return text


def _replace_greek_latin(s):
    """Replace Greek spelled out letters with their latin character."""
    for greek_spelled_out, latin in greek_to_latin.items():
        s = s.replace(greek_spelled_out, latin)
    return s
