"""Discover candidate longforms from a given corpus using the Acromine
algorithm."""
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
                 'parent', 'children']

    def __init__(self, longform=(), parent=None):
        self.longform = longform
        if longform:
            self.count = 1
            self.sum_ft = self.sum_ft2 = 0
            self.score = 1
        self.parent = parent
        self.children = {}

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

    def get_longforms(self, cutoff=0, scale=True, smoothing_param=4):
        """Return a list of extracted longforms with their scores

        Traverse the candidates trie to search for nodes with score
        greater than or equal to the scores of all children and strictly
        greater than the scores of all ancestors.

        Parameters
        ----------
        cutoff : Optional[int]
            Return only longforms with a score greater than the cutoff.
            Default: 0

        scale : Optional[bool]
            Whether or not to scale likelihood scores. If True, a likelihood
            score is transformed to (score-1)/(count + smoothing_param-1) where
            smoothing_param can be supplied by the user. Default: True

        smoothing_param : Optional[float]
            Value of smoothing parameter to use in scaling transformation. This
            is ignored if scale is set to False. Default: 4

        Returns
        -------
        longforms : list of tuple
            list of longforms along with their scores. It is sorted first in
            descending order by score, then by the length of the longform from
            shortest to longest, and finally by lexicographic order.
        """
        if scale:
            def score_func(score, count):
                return (score-1)/(count+smoothing_param-1)
        else:
            def score_func(score, count):
                return score
        root = self._internal_trie
        longforms = self._get_longform_helper(root, score_func)

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

    def _get_longform_helper(self, node, score_func):
        if not node.children:
            return [(node.longform, score_func(node.score, node.count))]
        else:
            result = []
            for child in node.children.values():
                child_longforms = self._get_longform_helper(child, score_func)
                result.extend([(longform, score) for longform, score in
                               child_longforms if node.is_root() or
                               score > score_func(node.score, node.count)])
            if not result:
                result = [(node.longform, score_func(node.score, node.count))]
            return result

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
                    self._longforms[current.longform[::-1]] = \
                        current.score
                current = current.children[token]

    def _make_readable(self, tokens):
        """Convert longform from internal representation to a human readable one
        """
        return ' '.join(self._stemmer.most_frequent(token)
                        for token in tokens[::-1])


def _norm_score(score, count, beta):
    numerator = score-1
    denominator = count+beta-1
    return 0 if denominator == 0 else numerator/denominator
