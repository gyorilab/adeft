import re
import string

from adeft.nlp import tokenize


def get_candidate_fragments(text, shortform, window=100):
    """Returns candidate longform fragments from text

    Identifies candidate longforms by searching for defining patterns (DP)
    in the text. Candidate longforms consist of non-punctuation tokens within
    a specified range of characters before the DP up until either the start
    of the text, the end of a previous DP or optionally a token from a set of
    excluded tokens.

    Parameters
    ----------
    text: Text to search for defining patterns (DP)

    shortform : Shortform to disambiguate

    window : Optional[int]
        Specifies range of characters before a defining pattern (DP)
        to consider when finding longforms. If set to 30, candidate
        longforms would be taken from the string
        "ters before a defining pattern". Default: 100

    exclude : Optional[set of str]
        Terms that are to be excluded from candidate longforms.
        Default: None
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
    """Return tokens in candidate fragment up until last excluded word"""
    if exclude is None:
        exclude = set()
    tokens = [token.lower() for token, _
              in tokenize(fragment)
              if token not in string.punctuation]
    index = len(tokens)
    # Take only tokens from end up to but not including the last
    # excluded in the fragment
    while index > 0:
        index -= 1
        if tokens[index] in exclude:
            tokens = tokens[index+1:]
            break
    return tokens
