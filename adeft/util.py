"""Utility functions used by Adeft internally.

"""
import re
from unicodedata import category

from adeft.nlp import word_tokenize, word_detokenize


def get_candidate_fragments(text, shortform, window=100):
    """Return candidate longform fragments from text

    Gets fragments of text preceding defining patterns (DPs) to search
    for candidate longforms. Each fragment contains either a specified range
    of characters before a DP, or characters up until either the start
    of the sentence or the end of a previous DP.


    Parameters
    ----------
    text : str
        Text to search for defining patterns (DP)
    shortform : str
        Shortform to disambiguate
    window : Optional[int]
        Specifies range of characters before a defining pattern (DP)
        to consider when finding longforms. If set to 30, candidate
        longforms would be taken from the string
        "ters before a defining pattern". Default: 100
    """
    # Find defining patterns by matching a regular expression
    matches = re.finditer(r'\s\(%s\)' % re.escape(shortform), text)
    # Keep track of the index of the end of the previous
    # Longform candidates cannot contain a previous DP and any text
    # before them
    end_previous = -1
    result = []
    for match in matches:
        # coordinates of current match
        span = match.span()
        # beginning of window containing longform candidate
        left = max(end_previous+1, span[0]-window)
        # fragment of text in this window
        fragment = text[left:span[0]]
        if not fragment:
            continue
        result.append(fragment)
        end_previous = span[1]
    return result


def get_candidate(fragment):
    """Return tokens in candidate fragment up until last excluded word

    Parameters
    ----------
    fragment : str
        The fragment to return tokens from.

    use_stemming : Optional[bool]
        If True, stem apply stemming to tokens. Default: True
    """
    fragment = fragment.strip()
    tokens = word_tokenize(fragment)
    longform_map = {}
    i, j = len(tokens) - 1, 0
    processed_tokens = []
    while i >= 0:
        if len(tokens[i][0]) > 1 or not category(tokens[i][0]).startswith('P'):
            processed_tokens.append(tokens[i][0])
            longform_map[j+1] = word_detokenize(tokens[i:])
            j += 1
        i -= 1
    longform_map[len(processed_tokens)] = fragment
    processed_tokens.reverse()
    return processed_tokens, longform_map


class _TrieNode(object):
    """TrieNode structure for use in recognizer

    Attributes
    ----------
    longform : str or None
        Set to associated longform at leaf nodes in the trie, otherwise None.
        Each longform corresponds to a path in the trie from root to leaf.

    children : dict
        dict mapping tokens to child nodes
    """
    __slots__ = ['data', 'children']

    def __init__(self, data=None):
        self.data = data
        self.children = {}


class SearchTrie(object):
    def __init__(self, lexicon, expander=None, token_map=None):
        """Initialize search trie with longforms in grounding map
        """
        if expander is None:
            def expander(x):
                return [x]
        if token_map is None:
            def token_map(x):
                return x
        root = _TrieNode()
        self._trie = root
        for longform in lexicon:
            for expansion in expander(longform):
                edges = tuple(token_map(token)
                              for token in get_candidate(expansion)[0][::-1])
                self.add(edges, longform)
        self.token_map = token_map

    def add(self, tokens, data):
        current = self._trie
        for index, token in enumerate(tokens):
            if token not in current.children:
                if index == len(tokens) - 1:
                    new = _TrieNode(data)
                else:
                    new = _TrieNode()
                current.children[token] = new
                current = new
            else:
                current = current.children[token]
                if index == len(tokens) - 1:
                    current.data = data

    def search(self, tokens):
        """Find longform expansion based on grounding map

        Parameters
        ----------
        tokens : list of str
            contains tokens that precede the occurence of the pattern
            "<longform> (<shortform>)" up until start of window

        Returns
        -------
        str
            Identified longform expansion
        """
        current = self._trie
        result = None
        match_text = []
        for token, mapped_token in tuple((token, self.token_map(token))
                                         for token in tokens[::-1]):
            if mapped_token not in current.children:
                break
            match_text.append(token)
            if current.children[mapped_token].data is not None:
                result = current.children[mapped_token].data
            current = current.children[mapped_token]
        match_text = ' '.join(match_text[::-1])
        return result, match_text
