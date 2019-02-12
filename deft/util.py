import json
import string


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




def is_jsonable(x):
    """Tests whether an object can be serialized to json"""
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False
