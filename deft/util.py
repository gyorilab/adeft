import json
import string

from nltk import ngrams
from multiset import Multiset

from deft.nlp import word_tokenize


def contains_shortform(sentence, shortform):
    """Count the occurences of a shortform in a sentence in standard pattern"""
    return sentence.count('(%s)' % shortform)


def get_max_candidate_longform(sentence, shortform, exclude=None):
    """Returns maximal candidate longform from within a sentence.

    a maximal longform candidate from a sentence containing a given
    shortform in the standard pattern consists of the non-punctuation
    tokens within the sentence between the beginning of the sentence and
    the first occurence of the standard pattern (<shortform>).
    If a set of excluded tokens is supplied, the maximal candidate longform
    contains all non-punctuation tokens preceding the first occurence of
    the standard pattern up until the first preceding excluded token.

    Parameters
    ----------
    sentence : str
        A sentence containing the pattern <longform> (<shortform>)

    shortform : str

    exclude : Optional[set of str]
        set tokens that are not permitted to appear in a candidate longform
        default: None

    Returns
    -------
    candidate : list of str
        list of tokens appearing in the maximal candidate longform
    """
    if exclude is None:
        exclude = set([])

    # tokenize sentence into list of words
    tokens = word_tokenize(sentence)

    # Loop through tokens. The nltk word tokenizer used will split off
    # the parentheses surrounding the shortform into separate tokens.
    for index in range(len(tokens) - 2):
        if tokens[index] == '(' and tokens[index+1] == shortform \
           and tokens[index+2] == ')':
            # The shortform has been found in parentheses

            # Capture all tokens in the sentence up until but excluding
            # the left parenthese containing the shortform, excluding
            # punctuation
            candidate = [token for token in tokens[:index]
                         if token not in string.punctuation]

            # convert tokens to lower case
            candidate = [token.lower() for token in candidate]
            # Keep only the tokens preceding the left parenthese up until
            # but not including the first stop word
            i = len(candidate)-1
            while i >= 0 and candidate[i] not in exclude:
                i -= 1
            candidate = candidate[i+1:]
            if candidate:
                return candidate
            else:
                return None


def is_plausible(shortform, longform):
    """Returns True if candidate longform meets minimum standards of plausibility

    Parameters
    ----------
    shortform : str
        a shortform

    longform : str
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
    text1 : str

    text2 : str

    n : Optional[int]
        ngram size. Default: 1
    """
    ngrams1 = Multiset(ngrams(text1, n))
    ngrams2 = Multiset(ngrams(text2, n))
    return len(2*ngrams1.intersection(ngrams2))/(len(ngrams1) + len(ngrams2))


def is_jsonable(x):
    """Tests whether an object can be serialized to json"""
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False
