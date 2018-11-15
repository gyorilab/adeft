import numpy as np
import string
from collections import defaultdict
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.snowball import EnglishStemmer

# Default set of stop words to exclude from candidate longforms
_stop = set(['a', 'an', 'the', 'and', 'or', 'of', 'with', 'at', 'from',
             'into', 'to', 'for', 'on', 'by', 'be', 'being', 'been', 'am',
             'is', 'are', 'was', 'were', 'in', 'that', 'as'])

# Facilitate mapping of unicode greek letters to ascii text in candidate
# longforms
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
    """Wraps EnglishStemmer.

    Keeps track of the number of times words have been mapped to particular
    stems by the wrapped stemmer. The acromine algorithm works with stemmed
    forms but it is useful to be able to recover actual words from stems.

    Attributes
    ----------
    __snow: :py:class:`nltk.stem.snowball.EnglishStemmer

    counts: defaultdict of defaultdict of int
        Contains the count of the number of times a particular word has been
        mapped to from a particular stem by the wrapped stemmer. Of the form
        counts[stem:str][word:str] = count:int
"""
    def __init__(self):
        self.__snow = EnglishStemmer()
        self.counts = defaultdict(lambda: defaultdict(int))

    def stem(self, word):
        """Returns stemmed form of word.

        Adds one to count associated to the computed stem, word pair.

        Parameters
        ----------
        word: str
            text to stem

        Returns
        -------
        stemmed: str
            stemmed form of input word
        """
        stemmed = self.__snow.stem(word)
        self.counts[stemmed][word] += 1
        return stemmed

    def most_frequent(self, stemmed):
        """Return the most frequent word mapped to a given stem

        Parameters
        ----------
        stemmed: str
            Stem that has previously been output by the wrapped snowball
            stemmer.

        Returns
        -------
        output: str|None
            Most frequent word that has been mapped to the input stem or None
            if the wrapped stemmer has never mapped the a word to the input
            stem.
        """
        candidates = self.counts[stemmed].items()
        if candidates:
            output = max(candidates, key=lambda x: x[1])[0]
        else:
            output = None
        return output


class ContinuousMiner(object):
    """Finds possible longforms corresponding to an abbreviation in a text corpus

    An online method. Updates likelihoods of terms as it consumes
    additional texts. Based on a trie datastructure. Likelihoods for longforms
    are updated at the time of insertions into the trie.

    Parameters
    ----------
    shortform: str
        Search for candidate longforms associated to this shortform

    stop_words: Optional[set of str]
        Terms that are to be excluded from candidate longforms.
        Default: set([])

    Attributes
    ----------
    _internal_trie: dict
        Stores trie datastructure that algorithm is based upon

    _longforms: dict
        Dictionary mapping candidate longforms to their likelihoods as
        produced by the acromine algorithm

    _snow: :py:class:`deft.acromine.SnowCounter`
        English stemmer that keeps track of counts of the number of times a
        given word has been mapped to a given stem
    """
    def __init__(self, shortform, stop_words=set([])):
        self.shortform = shortform
        self._internal_dict = {}
        self._longforms = {}
        self._snow = SnowCounter()
        self.stop_words = stop_words

    def consume(self, texts):
        """Consume a corpus of texts and use them to train the Acromine

        Each text is tokenized into sentences. Sentences are identified that
        contain the pattern f'({self.shortform}). These sentences are then
        tokenized into lists of words and punctuation is removed from these
        lists. The remaining words are then stemmed and converted to lower
        case. A maximal candidate longform is found for each of these
        sentences. Every suffix is then considered as a candidate longform and
        added to the internal trie, which updates the likelihoods for each of
        the candidates.

        If there are no excluded stop words, for the sentence "The growth of
        estrogen receptor (ER)-positive breast cancer cells is inhibited by
        all-trans-retinoic acid (RA)." The maximal candidate longform would be
        "the growth of estrogen receptor". It and the candidates
        "growth of estrogen receptor", "of estrogen receptor",
        "estrogen receptor", "receptor" would then be added to the internal
        trie.

        Parameters:
        texts: list of str
            A list of texts
        """
        # split each text into a list of sentences
        split = [sent_tokenize(text) for text in texts]
        # extract sentences defining shortform through standard pattern
        sentences = [sentence for text in split for sentence in text
                     if f'({self.shortform})' in sentence]
        # tokenize these sentences into lists of words
        tokens = [word_tokenize(sentence) for sentence in sentences]
        # extract maximal candidate longforms from each such sentence
        candidates = [self._get_candidates(sentence)[::-1]
                      for sentence in tokens]
        # add each candidate to the internal trie
        for candidate in candidates:
            self._add(candidate)

    def top(self, n):
        """Return top scoring candidates from the acromine."""

        output = sorted(self._longforms.items(), key=lambda x: x[1],
                        reverse=True)[0:n]
        output = [(tuple(self._snow.most_frequent(token)
                         for token in longform),
                   LH)
                  for longform, LH in output]
        return output

    def _add(self, tokens):
        """Add a list of tokens to the internal trie and update likelihoods

        Parameters
        ----------
        tokens: str
            A list of tokens to add to the internal trie.

        """
        current = self._internal_trie
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

    def _get_candidates(self, tokens):
        for index in range(len(tokens) - 3):
            if tokens[index] == '(' and tokens[index+1] == self.shortform and \
               tokens[index+2] == ')':
                output = [token for token in tokens[:index]
                          if token not in string.punctuation]
                for i, token in enumerate(output):
                    if token in _greek_alphabet:
                        output[i] = _greek_alphabet[token]
                output = [self._snow.stem(token) for token in output]
                i = len(output)-1
                while i >= 0 and output[i] not in self.stop_words:
                    i -= 1
                return output[i+1:]
