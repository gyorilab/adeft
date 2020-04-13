from fuzzywuzzy import fuzz

from adeft.nlp.stem import stem
from adeft.util import get_candidate


def _equivalence_helper(text1, text2):
    if text1.lower() == text2.lower():
        return True
    stem1, stem2 = stem(text1), stem(text2)
    if text1.endswith('ies') and stem1.endswith('i') \
       and text2.endswith('y') and text2[:-1].lower() == stem2[:-1]:
        return True
    if text1.endswith('es') and text1[:-2].lower() == text2.lower():
        return True
    if text1.endswith('es') and text2.endswith('e') and \
       text1[:-2].lower() == text2[:-1].lower():
        return True
    if text1.endswith('s') and text1[:-1].lower() == text2.lower():
        return True
    return False


def equivalent_up_to_plural(text1, text2):
    return (_equivalence_helper(text1, text2) or
            _equivalence_helper(text2, text1))


def text_similarity(text, grounding_text):
    if text.lower() == grounding_text.lower():
        output = 1.0
    elif len(get_candidate(grounding_text)[0]) > 1:
        output = fuzz.ratio(text.lower(), grounding_text.lower())/100
    elif equivalent_up_to_plural(text, grounding_text):
        output = 0.95
    else:
        output = 0.0
    return output
