import json
import string

from deft.nlp import word_tokenize


    if exclude is None:
def is_jsonable(x):
    """Tests whether an object can be serialized to json"""
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False
