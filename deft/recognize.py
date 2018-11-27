from deft.extraction import Processor
from nltk.stem.snowball import EnglishStemmer
import logging


logger = logging.getLogger('recognizer')

_snow = EnglishStemmer()


class _TrieNode(object):
    __slots__ = ['longform', 'children']
    """Barebones TrieNode struct for use in recognizer

    Attributes
    ----------
    concept: str|None
        concept has a str value containing an agent ID at terminal nodes of
        the recognizer trie, otherwise it has a None value.

    children: dict
        dict mapping tokens to child nodes
    """
    def __init__(self, longform=None):
        self.longform = longform
        self.children = {}


class Recognizer(object):
    __slots__ = ['shortform', 'longforms', '_trie', '_processor']
    """Class for recognizing concepts based on matching the standard pattern

    Searches text for the pattern "<longform> (<shortform>)" for longforms that
    have been identified by the miner and added to a concept map. Uses concept
    map to find a grounding for the shortform.

    Parameters
    ----------
    shortform: str
        shortform to be recognized

    longforms: dict of tuple: str
        Keyed on tuples containing the stemmed tokens of longforms extracted by
        the miner. Tokens must be in reverse order for easy insertion into the
        the trie. Maps these keys to representative readable strings.".

    Attributes
    ----------
    _trie: :py:class:`deft.recognizer.__TrieNode`
        Trie used to search for longforms that appear in the concept map. Edges
        correspond to stemmed tokens from longforms appearing in reverse order.
        Terminal nodes correspond to longforms with the concept appearing in
        the node data.

    _processor: :py:class:`deft.extraction:Processor`
        Processor capable of recognizing maximal longform candidates associated
        to a shortform. The trie will search for longforms nested within the
        maximal candidates.
    """

    def __init__(self, shortform, longforms, exclude=None):
        self.shortform = shortform
        self.longforms = longforms
        self._trie = self._init_trie(longforms)
        self._processor = Processor(shortform, exclude)

    def recognize(self, text):
        """Find the concept associated to a shortform in text by pattern matching

        Parameters
        ----------
        text: str
            Plaintext for which a dismabiguation of a shortform is sought

        Returns
        -------
        str|None
            Longform corresponding to shortform appearing in text if the
            pattern <longform> (<shortform>) appears and the
            longform appears input map of longforms. None if the pattern does
            not appear or if there is no corresponding entry in the map.
        """
        # Extract maximal longform candidates from the text
        candidates = self._processor.extract(text)
        # Search the trie for longforms appearing in each maximal candidate
        # As in the miner, tokens are stemmed and put in reverse order
        longforms = set([self._search(tuple(_snow.stem(token)
                                            for token in candidate[::-1]))
                         for candidate in candidates])
        # There should only be one concept matching the pattern. If not make
        # a note of it in the logger
        if len(longforms) > 1:
            logger.info(f'The standard pattern with shortform {self.shortform}'
                        'occurs in text multiple times with different'
                        ' groundings.\n'
                        '{text}')
        # Returns a concept if one is found matching the pattern, else None
        # Picks one at random if multiple concepts are found
        return longforms.pop() if longforms else None

    def _init_trie(self, longforms):
        """Initialize search trie from concept_map

        Parameters
        ---------
        longforms: dict of tuple: str
            Keyed on tuples containing the stemmed tokens of longforms
            extracted by the miner. Maps these keys to readable longform
            texts.

        Returns
        -------
        root: :py:class:`deft.recogizer.__TrieNode`
            Root of search trie used to recognize longforms
        """
        root = _TrieNode()
        for key, longform in self.longforms.items():
            current = root
            for index, token in enumerate(key):
                if token not in current.children:
                    if index == len(key) - 1:
                        new = _TrieNode(longform)
                    else:
                        new = _TrieNode()
                    current.children[token] = new
                    current = new
                else:
                    current = current.children[token]
        return root

    def _search(self, tokens):
        """Returns concept associated to a longform from a maximal candidate

        Parameters
        ----------
        tokens: tuple of str
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
            if current.children[token].longform is not None:
                return current.children[token].longform
            else:
                current = current.children[token]
        else:
            return None
