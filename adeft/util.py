"""Utility functions used by Adeft internally.

"""
import re
from unicodedata import category

from adeft.nlp import word_tokenize, word_detokenize


def get_candidate_fragments(text, shortform, window=100):
    """Return candidate longform fragments from text

    Gets fragments of text preceding defining patterns (DPs) to search
    for candidate longforms. Each fragment contains either a specified range
    of characters before a DP, or characters up until either the start
    of the sentence or the end of a previous DP.


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
    matches = re.finditer(r'\s\(%s\)' % re.escape(shortform), text)
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
        if not fragment:
            continue
        result.append(fragment)
        end_previous = span[1]
    return result


def get_candidate(fragment):
    """Return tokens in candidate fragment up until last excluded word

    Parameters
    ----------
    fragment : str
        The fragment to return tokens from.
    """
    tokens = word_tokenize(fragment)
    longform_map = {}
    i, j = len(tokens) - 1, 0
    processed_tokens = []
    while i >= 0:
        if len(tokens[i][0]) > 1 or not category(tokens[i][0]).startswith('P'):
            processed_tokens.append(tokens[i][0].lower())
            longform_map[j+1] = word_detokenize(tokens[i:])
            j += 1
        i -= 1
    processed_tokens.reverse()
    return processed_tokens, longform_map
