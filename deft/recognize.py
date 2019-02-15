import logging

from nltk.stem.snowball import EnglishStemmer

from deft.nlp import word_tokenize
from deft.util import get_candidate_fragments


logger = logging.getLogger('recognize')

_stemmer = EnglishStemmer()


class _TrieNode(object):
    __slots__ = ['grounding', 'children']
    """TrieNode struct for use in recognizer

    Attributes
    ----------
    concept : str|None
        concept has a str value containing an agent ID at terminal nodes of
        the recognizer trie, otherwise it has a None value.

    children : dict
        dict mapping tokens to child nodes
    """
    def __init__(self, grounding=None):
        self.grounding = grounding
        self.children = {}


class DeftRecognizer(object):
    """Class for recognizing longforms by searching for defining patterns (DP)

    Searches text for the pattern "<longform> (<shortform>)" for a collection
    of longforms supplied by the user.

    Parameters
    ----------
    shortform : str
        shortform to be recognized

    longforms : iterable of str
        Contains candidate longforms.

    exclude : Optional[set]
        set of tokens to ignore when searching for longforms.
        Default: None

    build_corpus : Optional[bool]
        If True, self.recognize will return a tuple of values, a set of the
        recognized longforms and the input text with all sentences matching
        the standard pattern with the given shortform. Typically, this should
        only be set to be True in the CorpusBuilder

    Attributes
    ----------
    _trie : :py:class:`deft.recognizer.__TrieNode`
        Trie used to search for longforms. Edges correspond to stemmed tokens
        from longforms. They appear in reverse order to the bottom of the trie
        with terminal nodes containing the associated longform in their data.
    """
    def __init__(self, shortform, grounding_map, window=100, exclude=None):
        self.shortform = shortform
        self.grounding_map = grounding_map
        self._trie = self._init_trie()
        self.window = window
        if exclude is None:
            self.exclude = set([])
        else:
            self.exclude = exclude
        
    def recognize(self, text):
        """Find longforms in text by matching the standard pattern

        Parameters
        ----------
        sentence : str
            Sentence where we seek to disambiguate shortform

        Returns
        -------
        longform : set of str
            longform corresponding to shortform in sentence if the standard
            pattern is matched. Returns None if the pattern is not matched
        """
        groundings = set()
        fragments = get_candidate_fragments(text, self.shortform,
                                            window=self.window,
                                            exclude=self.exclude)
        for fragment in fragments:
            if not fragment:
                continue
            # search for longform in trie
            grounding = self._search(tuple(_stemmer.stem(token)
                                           for token in fragment[::-1]))
            # if a longform is recognized, add it to output list
            if grounding:
                groundings.add(grounding)
        return groundings

    def _init_trie(self):
        """Initialize search trie from iterable of longforms

        Parameters
        ---------
        longforms : iterable of str
            longforms to add to the trie. They will be tokenized and stemmed,
            then their tokens will be added to the trie in reverse order.

        Returns
        -------
        root : :py:class:`deft.recogize._TrieNode`
            Root of search trie used to recognize longforms
        """
        root = _TrieNode()
        for longform, grounding in self.grounding_map.items():
            edges = tuple(_stemmer.stem(token)
                          for token in word_tokenize(longform))[::-1]
            current = root
            for index, token in enumerate(edges):
                if token not in current.children:
                    if index == len(edges) - 1:
                        new = _TrieNode(grounding)
                    else:
                        new = _TrieNode()
                    current.children[token] = new
                    current = new
                else:
                    current = current.children[token]
        return root

    def _search(self, tokens):
        """Returns longform from maximal candidate preceding shortform

        Parameters
        ----------
        tokens : tuple of str
            contains tokens that precede the occurence of the pattern
            "<longform> (<shortform>)" up until the start of the containing
            sentence or an excluded word is reached. Tokens must appear in
            reverse order.

        Returns
        -------
        str|None:
            Agent ID corresponding to associated longform in the concept map
            if one exists, otherwise None.
        """
        current = self._trie
        for token in tokens:
            if token not in current.children:
                break
            if current.children[token].grounding is not None:
                return current.children[token].grounding
            else:
                current = current.children[token]
        else:
            return None
