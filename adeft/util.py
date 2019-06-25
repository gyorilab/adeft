"""Utility functions used by Adeft internally.

"""
import re
from unicodedata import category

from adeft.nlp import tokenize


def get_candidate_fragments(text, shortform, window=100):
    """Return candidate longform fragments from text

    Identifies candidate longforms by searching for defining patterns (DP)
    in the text. Candidate longforms consist of non-punctuation tokens within
    a specified range of characters before the DP up until either the start
    of the text, the end of a previous DP or optionally a token from a set of
    excluded tokens.

    Parameters
    ----------
    text : str
        Text to search for defining patterns (DP)
    shortform : str
        Shortform to disambiguate
    window : Optional[int]
        Specifies range of characters before a defining pattern (DP)
        to consider when finding longforms. If set to 30, candidate
        longforms would be taken from the string
        "ters before a defining pattern". Default: 100
    """
    # Find defining patterns by matching a regular expression
    matches = re.finditer(r'\(\s*%s\s*\)' % shortform, text)
    # Keep track of the index of the end of the previous
    # Longform candidates cannot contain a previous DP and any text
    # before them
    end_previous = -1
    result = []
    for match in matches:
        # coordinates of current match
        span = match.span()
        # beginning of window containing longform candidate
        left = max(end_previous+1, span[0]-window)
        # fragment of text in this window
        fragment = text[left:span[0]]
        result.append(fragment)
        end_previous = span[1]
    return result


def get_candidate(fragment, exclude=None):
    """Return tokens in candidate fragment up until last excluded word

    Parameters
    ----------
    fragment : str
        The fragment to return tokens from.
    exclude : Optional[set of str]
        Terms that are to be excluded from candidate longforms.
        Default: None
    """
    if exclude is None:
        exclude = set()
    tokens = [token.lower() for token, _
              in tokenize(fragment)
              if len(token) > 1 or not category(token).startswith('P')]
    index = len(tokens)
    # Take only tokens from end up to but not including the last
    # excluded in the fragment
    while index > 0:
        index -= 1
        if tokens[index] in exclude:
            tokens = tokens[index+1:]
            break
    return tokens
