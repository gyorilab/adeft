import pandas as pd
import numpy as np
import string
from collections import defaultdict
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.snowball import EnglishStemmer
# from scipy import spatial
# from gensim.models import KeyedVectors as kv

# model = kv.load_word2vec_format('../scratch/PubMed-and-PMC-w2v.bin',
#                                 binary=True)

_stop = set(['a', 'an', 'the', 'and', 'or', 'of', 'with', 'at', 'from',
             'into', 'to', 'for', 'on', 'by', 'be', 'being', 'been', 'am',
             'is', 'are', 'was', 'were', 'in', 'that', 'as'])

_greek_alphabet = {
    u'\u0391': 'Alpha',
    u'\u0392': 'Beta',
    u'\u0393': 'Gamma',
    u'\u0394': 'Delta',
    u'\u0395': 'Epsilon',
    u'\u0396': 'Zeta',
    u'\u0397': 'Eta',
    u'\u0398': 'Theta',
    u'\u0399': 'Iota',
    u'\u039A': 'Kappa',
    u'\u039B': 'Lamda',
    u'\u039C': 'Mu',
    u'\u039D': 'Nu',
    u'\u039E': 'Xi',
    u'\u039F': 'Omicron',
    u'\u03A0': 'Pi',
    u'\u03A1': 'Rho',
    u'\u03A3': 'Sigma',
    u'\u03A4': 'Tau',
    u'\u03A5': 'Upsilon',
    u'\u03A6': 'Phi',
    u'\u03A7': 'Chi',
    u'\u03A8': 'Psi',
    u'\u03A9': 'Omega',
    u'\u03B1': 'alpha',
    u'\u03B2': 'beta',
    u'\u03B3': 'gamma',
    u'\u03B4': 'delta',
    u'\u03B5': 'epsilon',
    u'\u03B6': 'zeta',
    u'\u03B7': 'eta',
    u'\u03B8': 'theta',
    u'\u03B9': 'iota',
    u'\u03BA': 'kappa',
    u'\u03BB': 'lamda',
    u'\u03BC': 'mu',
    u'\u03BD': 'nu',
    u'\u03BE': 'xi',
    u'\u03BF': 'omicron',
    u'\u03C0': 'pi',
    u'\u03C1': 'rho',
    u'\u03C3': 'sigma',
    u'\u03C4': 'tau',
    u'\u03C5': 'upsilon',
    u'\u03C6': 'phi',
    u'\u03C7': 'chi',
    u'\u03C8': 'psi',
    u'\u03C9': 'omega',
}


class SnowCounter(object):
    def __init__(self):
        self.__snow = EnglishStemmer()
        self.counts = defaultdict(lambda: defaultdict(int))

    def stem(self, word):
        stemmed = self.__snow.stem(word)
        self.counts[stemmed][word] += 1
        return stemmed

    def most_frequent(self, stemmed):
        candidates = self.counts[stemmed].items()
        if candidates:
            output = max(candidates, key=lambda x: x[1])[0]
        else:
            output = None
        return output


class AcroMine(object):
    """Suffix trie to hold longform candidates and their frequencies"""
    def __init__(self):
        self._internal_dict = {}
        self._longforms = {}
        self.__snow = SnowCounter()
        
    def add(self, tokens):
        """Add a list of tokens to the suffix trie"""
        current = self._internal_dict
        meta = {}
        for index, token in enumerate(tokens):
            if token not in current:
                if index > 0:
                    prev_lf = meta['longform']
                else:
                    prev_lf = []
                new = [{'count': 1,
                        'freqt': 0,
                        'freqt2': 0,
                        'LH': np.log(len(prev_lf) + 1),
                        'longform': prev_lf + [token]}, {}]
                self._longforms[tuple(new[0]['longform'][::-1])] = \
                    new[0]['LH']
                current[token] = new
                meta = new[0]
                current = new[1]
            else:
                count = current[token][0]['count']
                current[token][0]['count'] += 1
                current[token][0]['LH'] += \
                    np.log(len(current[token][0]['longform']))
                self._longforms[tuple(current[token][0]['longform'])[::-1]] = \
                    current[token][0]['LH']
                if index > 0:
                    meta['freqt'] += 1
                    meta['freqt2'] += 2*count + 1
                    ft = meta['freqt']
                    ft2 = meta['freqt2']
                    meta['LH'] += (ft - 1)/(ft2 - 2*count + 1)
                    meta['LH'] -= ft/ft2
                    current[token][0]['longform'] = \
                        meta['longform'] + [token]
                    self._longforms[tuple(meta['longform'][::-1])] = \
                        meta['LH']
                else:
                    current[token][0]['longform'] = [token]
                meta = current[token][0]
                current = current[token][1]

    def consume(self, texts):
        split = [sent_tokenize(text) for text in texts]
        ER_sentences = [sentence for text in split for sentence in text
                        if '(ER)' in sentence]
        ER_tokens = [word_tokenize(sentence) for sentence in ER_sentences]
        candidates = [self._get_candidates(sentence, 'ER')[::-1]
                      for sentence in ER_tokens]
        for candidate in candidates:
            self.add(candidate)

    def top(self, n):
        """Return top scoring candidates from the acromine."""

        output = sorted(self._longforms.items(), key=lambda x: x[1],
                        reverse=True)[0:n]
        print(output)
        output = [tuple(self.__snow.most_frequent(token) for token in longform)
                  for longform in output]
        return output

    def _get_candidates(self, tokens, shortform):
        for index in range(len(tokens) - 3):
            if tokens[index] == '(' and tokens[index+1] == shortform and \
               tokens[index+2] == ')':
                output = [token for token in tokens[:index]
                          if token not in string.punctuation]
                for i, token in enumerate(output):
                    if token in _greek_alphabet:
                        output[i] = _greek_alphabet[token]
                output = [self.__snow.stem(token) for token in output]
                i = len(output)-1
                while i >= 0 and output[i] not in _stop:
                    i -= 1
                return output[i+1:]

        
def sublist(x, y):
    return any(y[pos:pos+len(x)] == x
               for pos in range(0, len(y) - len(x) + 1))


ER_df = pd.read_pickle('../scratch/ER_statements.pkl')
ER_df = ER_df.groupby('text_id').first()
ER_df = ER_df[~ER_df.fulltext.isna()]
ER_df.fulltext = ER_df.fulltext.apply(lambda x: ' '.join(x.split()))

fulltexts = ER_df.fulltext.values
