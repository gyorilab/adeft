from nltk import ngrams
from multiset import Multiset


def is_plausible(shortform, longform):
    """Returns True if candidate longform meets minimum standards of plausibility

    Parameters
    ----------
    shortform: str
        a shortform

    longform: str
        a candidate longform for the given shortform
    Returns
    -------
    bool:
        True if all of the letters in the shortform are included in the
        candidate longform and the length of the candidate longform
        exceeds the length of the candidate shortform plus after removing all
        white space from the candidate longform
    """
    contains_all_letters = set(shortform.lower()) <= set(longform)
    long_enough = len(''.join(longform.split())) - 1 > len(shortform)
    return contains_all_letters and long_enough


def char_ngram_similarity(text1, text2, n=1):
    """Return character ngram similarity between two strings


    Parameters
    ----------
    text1: str

    text2: str

    n: Optional[int]
        ngram size. Default: 1
    """
    ngrams1 = Multiset(ngrams(text1, n))
    ngrams2 = Multiset(ngrams(text2, n))
    return len(2*ngrams1.intersection(ngrams2))/(len(ngrams1) + len(ngrams2))
