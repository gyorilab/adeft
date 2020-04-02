import os
import re
import json
from collections import defaultdict

from nltk.stem.snowball import EnglishStemmer
from nltk.tokenize.punkt import PunktParameters, PunktSentenceTokenizer

from adeft.locations import RESOURCES_PATH

TOKENIZER_LOCATION = os.path.join(RESOURCES_PATH, 'tokenizer_params.json')


class WatchfulStemmer(object):
    """Wraps the nltk.snow EnglishStemmer.

    Keeps track of the number of times words have been mapped to particular
    stems by the wrapped stemmer. Extraction of longforms works with stemmed
    tokens but it is necessary to recover actual words from stems.

    Parameters
    ----------
    counts : Optional[dict]
        counts dictionary as used internally in WatchfulStemmer. Allows for
        loading a previously saved WatchfulStemmer

    Attributes
    ----------
    __snowball : :py:class:`nltk.stem.snowball.EnglishStemmer`

    counts : defaultdict of defaultdict of int
        Contains the count of the number of times a particular word has been
        mapped to from a particular stem by the wrapped stemmer. Of the form
        counts[stem:str][word:str] = count:int
    """
    def __init__(self, counts=None):
        if counts is None:
            counts = {}
        self.__snowball = EnglishStemmer()
        self.counts = defaultdict(lambda: defaultdict(int),
                                  {key: defaultdict(int, value)
                                   for key, value in counts.items()})

    def stem(self, word):
        """Returns stemmed form of word.

        Adds one to count associated to the computed stem, word pair.

        Parameters
        ----------
        word : str
            text to stem

        Returns
        -------
        stemmed : str
            stemmed form of input word
        """
        stemmed = self.__snowball.stem(word)
        self.counts[stemmed][word] += 1
        return stemmed

    def most_frequent(self, stemmed):
        """Return the most frequent word mapped to a given stem

        Parameters
        ----------
        stemmed : str
            Stem that has previously been output by the wrapped snowball
            stemmer.

        Returns
        -------
        output : str or None
            Most frequent word that has been mapped to the input stem or None
            if the wrapped stemmer has never mapped the a word to the input
            stem. Break ties with lexicographic order
        """
        words = list(self.counts[stemmed].items())
        if words:
            words.sort(key=lambda x: x[1], reverse=True)
            candidates = [word[0] for word in words if word[1] == words[0][1]]
            output = min(candidates)
        else:
            raise ValueError('stem %s has not been observed' % stemmed)
        return output

    def dump(self):
        """Returns dictionary of info needed to reconstruct stemmer"""
        return dict(self.counts)


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


stopwords_min = set(['a', 'an', 'the', 'and', 'or', 'of', 'with', 'at',
                     'from', 'into', 'to', 'for', 'on', 'by', 'be', 'been',
                     'am', 'is', 'are', 'was', 'were', 'in', 'that', 'as'])

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'stopwords.json'), 'r') as f:
    english_stopwords = json.load(f)


class AdeftSentenceTokenizer(object):
    """Split text into sentences using custom trained PunktSentenceTokenizer

    Parameters
    ----------
    text : str

    Returns
    -------
    list of str
        List of sentences in the input text as decided by the punkt tokenizer
    """
    def __init__(self):
        with open(TOKENIZER_LOCATION) as f:
            params_dict = json.load(f)
        params = PunktParameters()
        params.abbrev_types = set(params_dict['abbrev_types'])
        params.collocations = set([tuple(colloc)
                                  for colloc in params_dict['collocations']])
        params.ortho_context = params_dict['ortho_context']
        params.sent_starters = set(params_dict['sent_starters'])
        self.tokenizer = PunktSentenceTokenizer(params)

    def __call__(self, text):
        return self.tokenizer.tokenize(text)


sentence_tokenize = AdeftSentenceTokenizer()
