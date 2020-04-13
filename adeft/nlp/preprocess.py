import re
from itertools import chain, combinations, product

from adeft.nlp.resources import dashes


def word_tokenize(text):
    """Simple word tokenizer based on a regular expression pattern

    Everything that is not a block of alphanumeric characters is considered as
    a separate token.

    Parameters
    ----------
    text : str
        Text to tokenize

    Returns
    -------
    tokens : list of tuple
        Tokens in the input text along with their text coordinates. Each
        tuple has a token as the first element and the tokens coordinates
        as its second element.
    """
    pattern = re.compile(r'\w+|[^\s\w]')
    matches = re.finditer(pattern, text)
    return [(m.group(), (m.start(), m.end()-1)) for m in matches]


def word_detokenize(tokens):
    """Return inverse of the Adeft word tokenizer

    The inverse is inexact. For simplicity, all white space characters are
    replaced with a space. An exact inverse is not necessary for adeft's
    purposes.

    Parameters
    ----------
    tokens : list of tuple
        List of tuples of the form (word, (start, end)) giving tokens
        and coordinates as output by Adeft's word tokenizer

    Returns
    -------
    output : str
        The original string that produced the input tokens, with the caveat
        that every white space character will be replaced with a space.
    """
    # Edge cases: input text is empty string or only has one token
    if len(tokens) == 0:
        return ''
    elif len(tokens) == 1:
        return tokens[0][0]

    # This looks messy but is simple conceptually.
    # At each step add the current token and a number of spaces determined
    # by the coordinates of the previous token and the current token.
    output = [tokens[0][0]] + [' ']*(tokens[1][1][0] - tokens[0][1][1] - 1)
    for index in range(1, len(tokens)-1):
        output.append(tokens[index][0])
        output.extend([' ']*(tokens[index+1][1][0]
                             - tokens[index][1][1] - 1))
    output.append(tokens[-1][0])
    return ''.join(output)


def expand_dashes(text):
    text = _normalize_dashes(text)
    if text.count('-') > 4:
        output = [text]
    else:
        tokens = _dash_tokenize(text)
        output = [' '.join(x) for x in product(*[_expand_token(token)
                                                 for token in tokens])]
    return output


def _powerset(n):
    return chain.from_iterable(combinations(range(n), r)
                               for r in range(n+1))


def _normalize_dashes(text):
    out = ''
    for char in text:
        if char in dashes:
            out += '-'
        else:
            out += char
    out = '-'.join([x for x in out.split('-') if x])
    return out


def _expand_token(text):
    tokens = text.split('-')
    if len(tokens) > 5:
        return [text, text.replace('-', '')]
    out = []
    for subset in _powerset(len(tokens) - 1):
        result = tokens[0]
        for i, token in enumerate(tokens[1:]):
            if i in subset:
                result += token
            else:
                result += ' ' + token
        out.append(result)
    return out


def _dash_tokenize(text):
    pattern = re.compile(r'[\w-]+|[^\s\w]')
    matches = re.finditer(pattern, text)
    return [m.group() for m in matches]
