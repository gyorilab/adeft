"""Discover candidate longforms from a given corpus using the Acromine
algorithm."""
from collections import deque
import logging

from adeft.nlp import WatchfulStemmer
from adeft.util import get_candidate_fragments, get_candidate

logger = logging.getLogger(__file__)


class _TrieNode(object):
    """Node in Trie associated to a candidate longform

    The children of a node associated to a candidate longform c are all
    observed candidates t that can be obtained by prepending a single token
    to c.

    Contains the current likelihood for the candidate longform as well
    as all information needed to calculate it. The likelihood's are
    updated as candidates are added to the Trie.


    Parameters
    ----------
    longform : list of str
        List of tokens within a candidate longform in reverse order.
        For the candidate longform "in the endoplasmic reticulum" the
        list will take the form ["reticulum", "endoplasmic", "the", "in"],
        ignoring that the tokens should actually be stemmed.

    Attributes
    ----------

    longform : list of str
        Explained above

    count : int
        Current co-occurence frequency of candidate longform with shortform

    sum_ft : int
        Sum of the co-occurence frequencies of all previously observed
        candidate longforms that are children of the associated candidate
        longform (longforms that can be obtained by prepending
        one token to the associated candidate longform).

    sum_ft2 : int
        Sum of the squares of the co-occurence freqencies of all previously
        observed candidate longforms that are children of the associated
        longform.
    score : float
        Likelihood score of the associated candidate longform.
        It is given by count - sum_ft**2/sum_ft

        See
        [Okazaki06] Naoaki Okazaki and Sophia Ananiadou. "Building an
        abbreviation dicationary using a term recognition approach".
        Bioinformatics. 2006. Oxford University Press.

        for more information

    parent : :py:class:`adeft.discover._TrieNode`
        link to node's parent

    children : dict of :py:class:`adeft.discover._TrieNode`
        dictionary of child nodes

    best_ancestor_score : float
        best score among all of nodes ancestors

    best_ancestor : :py:class:`adeft.discover._TrieNode`
        ancestor of node with best score
    """
    __slots__ = ['longform', 'count', 'sum_ft', 'sum_ft2', 'score',
                 'parent', 'children', 'best_ancestor_score', 'best_ancestor']

    def __init__(self, longform=(), parent=None):
        self.longform = longform
        if longform:
            self.count = 1
            self.sum_ft = self.sum_ft2 = 0
            self.score = 1
        self.parent = parent
        self.children = {}
        self.best_ancestor_score = -1.
        self.best_ancestor = None

    def is_root(self):
        """True if node is at the root of the trie"""
        return self.parent is None

    def increment_count(self):
        """Update count and likelihood when observing a longform again

        Update when a previously observed longform is seen again
        """
        self.count += 1
        self.score += 1

    def update_likelihood(self, count):
        """Update likelihood when observing a child of associated longform

        Update when observing a candidate longform which can be obtained
        by prepending one token to the associated longform.

        Parameters
        ----------
        count : int
            Current co-occurence frequency of child longform with shortform
        """
        self.score += self.sum_ft2/self.sum_ft if self.sum_ft else 0
        self.sum_ft += 1
        self.sum_ft2 += 2*count - 1
        self.score -= self.sum_ft2/self.sum_ft


class AdeftMiner(object):
    """Finds possible longforms corresponding to an abbreviation in a text
    corpus.

    Makes use of the `Acromine <http://www.chokkan.org/research/acromine/>`_
    algorithm developed by Okazaki and Ananiadou.

    [Okazaki06] Naoaki Okazaki and Sophia Ananiadou. "Building an
    abbreviation dicationary using a term recognition approach".
    Bioinformatics. 2006. Oxford University Press.

    Parameters
    ----------
    shortform : str
        Shortform to disambiguate

    window : Optional[int]
        Specifies range of characters before a defining pattern (DP)
        to consider when finding longforms. If set to 30, candidate
        longforms would be taken from the string
        "ters before a defining pattern". Default: 100

    exclude : Optional[set of str]
        Terms that are to be excluded from candidate longforms.
        Default: None

    Attributes
    ----------
    _internal_trie : :py:class:`adeft.discover._TrieNode`
        Stores trie data-structure used to implement the algorithm

    _longforms : dict
        Dictionary mapping candidate longforms to their likelihoods as
        produced by the acromine algorithm

    _stemmer : :py:class:`adeft.nlp.stem.SnowCounter`
        English stemmer that keeps track of counts of the number of times a
        given word has been mapped to a given stem. Wraps the class
        EnglishStemmer from nltk.stem.snowball
    """
    def __init__(self, shortform, window=100, exclude=None):
        self.shortform = shortform
        self._internal_trie = _TrieNode()
        self._longforms = {}
        self._stemmer = WatchfulStemmer()
        self.window = window
        if exclude is None:
            self.exclude = set()
        else:
            self.exclude = exclude

    def process_texts(self, texts):
        """Update longform candidate scores from a corpus of texts

        Runs co-occurence statistics in a corpus of texts to compute
        scores for candidate longforms associated to the shortform. This
        is an online method, additional texts can be processed after training
        has taken place.

        Parameters
        ----------
        texts : list of str
            A list of texts
        """
        for text in texts:
            # lonform candidates taken from a window of text before each
            # defining pattern
            fragments = get_candidate_fragments(text, self.shortform,
                                                self.window)
            for fragment in fragments:
                if fragment:
                    candidate = get_candidate(fragment, self.exclude)
                    self._add(candidate)

    def top(self, limit=None):
        """Return top scoring candidates.

        Parameters
        ----------
        limit : Optional[int]
            Limit for the number of candidates to return. Default: None

        Returns
        ------
        candidates : list of tuple
            List of tuples, each containing a candidate string and its
            likelihood score. Sorted first in descending order by
            likelihood score, then by length from shortest to longest, and
            finally by lexicographic order.
        """
        if not self._longforms:
            return []

        candidates = sorted(self._longforms.items(), key=lambda x:
                            (-x[1], len(x[0]), x[0]))
        if limit is not None and limit < len(candidates):
            candidates = candidates[0:limit]
        # Map stems back to the most frequent word that had been mapped to them
        # and convert longforms in tuple format into readable strings.
        candidates = [(' '.join(self._stemmer.most_frequent(token)
                                for token in longform),
                       score)
                      for longform, score in candidates]
        return candidates

    def get_longforms(self, cutoff=1):
        """Return a list of extracted longforms with their scores

        Runs a breadth first search to search for nodes with score
        greater than or equal to the scores of all children and strictly less
        than the scores of all ancestors.

        Parameters
        ----------
        cutoff : Optional[int]
            Return only longforms with a score greater than the cutoff.
            Default: 1

        Returns
        -------
        longforms : list of tuple
            list of longforms along with their scores. It is sorted first in
            descending order by score, then by the length of the longform from
            shortest to longest, and finally by lexicographic order.
        """
        # Forward pass
        longforms = set()
        # The root contains no longform. Initialize queue with all of its
        # children
        queue = deque(self._internal_trie.children.values())
        while queue:
            node = queue.popleft()
            # if a node has a better score than its parent's best
            # ancestor it becomes its own best ancestor
            if node.score > node.parent.best_ancestor_score:
                node.best_ancestor_score = node.score
                node.best_ancestor = node
            # otherwise set its best ancestor to its parents best ancestor
            else:
                node.best_ancestor_score = node.parent.best_ancestor_score
                node.best_ancestor = node.parent.best_ancestor
            # a nodes score cannot exceed the count of its expected longform.
            # if the count for a child is less or equal to the best ancestor
            # score, the node is not added to the queue. track how many
            # children are added to the queue
            worthy = 0
            for child in node.children.values():
                if child.count > node.best_ancestor_score:
                    queue.append(child)
                    worthy += 1
            # if no children are added, the becomes a leaf. the optimal
            # longforms are given by the best ancestors of the leaves.
            if worthy == 0:
                longforms.add((node.best_ancestor.longform,
                               node.best_ancestor.score))

        # Convert longforms as tuples in reverse order into reader strings
        # mapping stems back to the most frequent token that had been mapped
        longforms = [(longform, score) for longform, score in longforms
                     if score > cutoff]

        # Map stems to the most frequent word that had been mapped to them.
        # Convert longforms as tuples in reverse order into reader strings
        # mapping stems back to the most frequent token that had been
        # mapped to them. tuple of stemmed tokens can be recovered by
        # tokenizing, stemming, and reversing
        longforms = [(self._make_readable(longform), score)
                     for longform, score in longforms]

        # Sort in preferred order
        longforms = sorted(longforms, key=lambda x: (-x[1], len(x[0]), x[0]))

        # Reset best ancestor and best_ancestor score values for all children
        # of the root. This is required for the algorithm to be able to run
        # successfully in subsequent calls to this method
        return longforms

    def _add(self, tokens):
        """Add a list of tokens to the internal trie and update likelihoods.

        Parameters
        ----------
        tokens : str
            A list of tokens to add to the internal trie.

        """
        # start at top of trie
        current = self._internal_trie
        # apply snowball stemmer to each token and put them in reverse order
        tokens = tuple(self._stemmer.stem(token) for token in tokens)[::-1]
        for token in tokens:
            if token not in current.children:
                # candidate longform is observed for the first time
                # add a new entry for it in the trie
                longform = current.longform + (token, )
                new = _TrieNode(longform, parent=current)
                # update likelihood of current node to account for the new
                # child unless current node is the root
                if not current.is_root():
                    current.update_likelihood(1)
                    self._longforms[current.longform[::-1]] = current.score
                # Add newly observed longform to the dictionary of candidates
                self._longforms[new.longform[::-1]] = new.score
                # set newly observed longform to be the child of current node
                current.children[token] = new
                # update current node to the newly formed node
                current = new
            else:
                # candidate longform has been observed before
                # update count for candidate longform and associated LH value
                current.children[token].increment_count()
                # Update entry for candidate longform in the candidates
                # dictionary
                self._longforms[current.children[token].longform[::-1]] = \
                    current.children[token].score
                if not current.is_root():
                    # we are not at the top of the trie. observed candidate
                    # has a parent

                    # update likelihood of candidate's parent
                    count = current.children[token].count
                    current.update_likelihood(count)
                    # Update candidates dictionary
                    self._longforms[current.longform[::-1]] = current.score
                current = current.children[token]

    def _make_readable(self, tokens):
        """Convert longform from internal representation to a human readable one
        """
        return ' '.join(self._stemmer.most_frequent(token)
                        for token in tokens[::-1])
