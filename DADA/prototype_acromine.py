import pandas as pd
from string import tokenization
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.snowball import EnglishStemmer


class AcroMine(object):
    """Suffix trie to hold longform candidates and their frequencies"""
    def __init__(self):
        self._internal_dict = {}
        self._longforms = {}

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
                        'LH': 1,
                        'longform': prev_lf + [token]}, {}]
                self._longforms[tuple(new[0]['longform'][::-1])] = \
                    new[0]['LH']
                current[token] = new
                meta = new[0]
                current = new[1]
            else:
                count = current[token][0]['count']
                current[token][0]['count'] += 1
                current[token][0]['LH'] += 1
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

    def top(self, n):
        """Return top scoring candidates from the acromine."""
        return sorted(self._longforms.items(), key=lambda x: x[1],
                      reverse=True)[0:n]


def get_candidates(tokens, shortform):
    for index in range(len(tokens) - 3):
        if tokens[index] == '(' and tokens[index+1] == shortform and \
           tokens[index+2] == ')':
            return tokens[:index]


def sublist(x, y):
    return any(y[pos:pos+len(x)] == x
               for pos in range(0, len(y) - len(x) + 1))


ER_df = pd.read_pickle('ER_statements.pkl')
ER_df = ER_df.groupby('text_id').first()
ER_df = ER_df[~ER_df.fulltext.isna()]
ER_df.fulltext = ER_df.fulltext.apply(lambda x: ' '.join(x.split()))

fulltexts = ER_df.fulltext.values

sentences = []
for index, fulltext in enumerate(fulltexts):
    sentences.append(sent_tokenize(fulltext))
    if index % 100 == 0:
        print(index)

ER_sentences = [[sentence for sentence in s if '(ER)' in sentence]
                for s in sentences]
