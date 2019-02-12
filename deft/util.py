import re
import json
import string

from deft.nlp import word_tokenize


def get_candidate_fragments(text, shortform, window=100, exclude=None):
    """Returns candidate longform fragments from text"""
    if exclude is None:
        exclude = set()
    matches = re.finditer(r'\(\s*%s\s*\)' % shortform, text)
    end_previous = -1
    result = []
    for match in matches:
        span = match.span()
        left = max(end_previous+1, span[0]-window)
        fragment = text[left:span[0]]
        tokens = word_tokenize(fragment)
        result.append([token for token in tokens
                       if token not in string.punctuation
                       and token not in exclude])
    return result


def strip_defining_patterns(text, shortform):
    """Strip instances of defining pattern from text"""
    return re.sub(r'\s?\(\s*%s\s*\)' % shortform, '', text)


def contains_shortform(sentence, shortform):
    """Count the occurences of a shortform in a sentence in standard pattern"""
    return sentence.count('(%s)' % shortform)

 
def is_jsonable(x):
    """Tests whether an object can be serialized to json"""
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False
