"""Discover candidate longforms from a given corpus using the Acromine
algorithm."""
import json
import logging
import numpy as np
from copy import deepcopy

from adeft.nlp import WatchfulStemmer
from adeft.score import AlignmentBasedScorer
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

    encoded_tokens : list of list of int
        Tokens for associated longform candidate encoded in form required
        by the alignment based scorer.

    word_prizes : list of float
        Alignment based scorer word prizes for associated longform candidate.

    stop_count : int
        Count of leading stopwords in parent candidate.

    best_ancestor_align_score : float
        Best alignment based score for all ancestors of a candidate.

    sum_parent_word_scores : float
        Sum of word scores for parent candidate.

    best_char_scores : float
        List of alignment based scorer char_scores for highest alignment
        based scoring ancestor of node.

    alignment_score : float
        Alignment based score of node. Computed using parameters specified
        when compute_alignment_scores was run. Default parameters are chosen
        if user does not run this function explicitly.

    best_ancestor_score : float
        Based likelihood score among ancestors of node.

    best_descendent_score : float
        Based likelihood score among descendents of node.
    """
    __slots__ = ['longform', 'count', 'sum_ft', 'sum_ft2', 'score',
                 'parent', 'children', 'encoded_tokens', 'word_prizes',
                 'best_ancestor_align_score', 'sum_parent_word_scores',
                 'best_char_scores', 'alignment_score', 'best_ancestor_score',
                 'best_descendent_score', 'stop_count']

    def __init__(self, longform=(), parent=None, shortform=None):
        self.longform = longform
        if longform:
            self.count = 1
            self.sum_ft = self.sum_ft2 = 0
            self.score = 1
        else:
            self.score = -1
        self.parent = parent
        self.children = {}
        self.encoded_tokens = []
        self.word_prizes = []
        self.sum_parent_word_scores = 0
        self.best_ancestor_align_score = -1
        self.alignment_score = 0
        self.stop_count = 0
        self.best_ancestor_score = -1
        self.best_descendent_score = -1
        if shortform is not None:
            self.best_char_scores = [-1e20]*len(shortform)

    def is_root(self):
        """True if node is at the root of the trie"""
        return self.parent is None

    def increment_count(self, increment=1):
        """Update count and likelihood when observing a longform again

        Update when a previously observed longform is seen again
        """
        self.count += increment
        self.score += increment

    def update_likelihood(self, count, increment=1):
        """Update likelihood when observing a child of associated longform

        Update when observing a candidate longform which can be obtained
        by prepending one token to the associated longform. This must
        always be ran after incrementing the count

        Parameters
        ----------
        count : int
            Current co-occurence frequency of child longform with shortform
        """
        self.score += self.sum_ft2/self.sum_ft if self.sum_ft else 0
        self.sum_ft += increment
        # When this is ran, count will already have been incremented.
        self.sum_ft2 += 2*count*increment - increment**2
        self.score -= self.sum_ft2/self.sum_ft

    def to_dict(self):
        """Returns a dictionary representation of trie
        """
        if not self.children:
            return {}
        out = {}
        for token, child in self.children.items():
            out[token] = {'count': child.count, 'score': child.score,
                          'sum_ft': child.sum_ft, 'sum_ft2': child.sum_ft2,
                          'longform': child.longform,
                          'children': child.to_dict()}
        return out


def load_trie(trie_dict, shortform):
    """Load a Trie from dictionary representation

    Parameters
    ---------
    trie_dict : dict
        Dictionary representation of trie as returned by to_dict method of
        py:class`adeft.discover._TrieNode`

    Returns
    -------
    py:class:`adeft.discover._TrieNode`
        root of trie built from input dictionary
    """
    root = _TrieNode(shortform=shortform)
    for key, value in trie_dict.items():
        root.children[key] = _load_trie_helper(value, root)
    return root


def _load_trie_helper(entry, parent):
    node = _TrieNode(longform=tuple(entry['longform']), parent=parent)
    node.count = entry['count']
    node.score = entry['score']
    node.sum_ft = entry['sum_ft']
    node.sum_ft2 = entry['sum_ft2']
    for key, value in entry['children'].items():
        node.children[key] = _load_trie_helper(value, parent=node)
    return node


class AdeftMiner(object):
    """Finds possible longforms corresponding to an abbreviation in a text
    corpus.

    Makes use of the `Acromine <http://www.chokkan.org/research/acromine/>`_
    algorithm developed by Okazaki and Ananiadou, combined with an alignment
    based longform scoring method we've developed.

    Acromine based likelihood scores are scaled to the range [0, 1] using the
    formula (likelihood_score - 1)/(M + smoothing_param - 1).  Where
    smoothing_param is a positive number and M is the maximum likelihood score
    between a candidate node and all of its ancestors and descendents
    (technically this score can be less than 0 for some poor candidate
    longforms).

    Scaled likelihood scores are combined with alignment based scores through
    a weighted average where the weight associated to the alignment based score
    decays exponentially with the value of M defined above. This gives more
    weight to the alignment based score for rarer longform expansions which the
    Acromine algorithm has difficulty handling.

    [Okazaki06] Naoaki Okazaki and Sophia Ananiadou. "Building an abbreviation
    dicationary using a term recognition approach".
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

    Attributes
    ----------
    _internal_trie : :py:class:`adeft.discover._TrieNode`
        Stores trie data-structure used to implement the algorithm

    _stemmer : :py:class:`adeft.nlp.stem.SnowCounter`
        English stemmer that keeps track of counts of the number of times a
        given word has been mapped to a given stem. Wraps the class
        EnglishStemmer from nltk.stem.snowball

    _alignment_scores_computed : bool
        Will be True if alignment scores have been computed for the current
        state of the candidate trie. It is reset to False any time the
        process_texts method is run

    _scores_propagated : bool
        Will be True if best ancestor and best descendent likelihood scores
        have been propagated to each node for the current state of the
        candidate trie. It is reset to False any time the process_texts method
        is run.
    """
    def __init__(self, shortform, window=100):
        self.shortform = shortform
        self._internal_trie = _TrieNode(shortform=shortform)
        self._stemmer = WatchfulStemmer()
        self.window = window
        self._alignment_scores_computed = False
        self._scores_propagated = False

    def process_texts(self, texts):
        """Update longform candidate scores from a corpus of texts

        Runs co-occurence statistics in a corpus of texts to compute
        likelihood scores for candidate longforms associated to the shortform.
        This is an online method, it can be run multiple times to process_texts
        multiple batches of text. This allows previously trained AdeftMiners to
        be updated when new content becomes available.

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
                    candidate, _ = get_candidate(fragment)
                    self._add(candidate)
        self._alignment_scores_computed = False
        self._scores_propagated = False

    def top(self, limit=None, smoothing_param=4, max_length='auto',
            use_alignment_based_scoring=True, weight_decay_param=0.001):
        """Return top scoring candidates.

        Parameters
        ----------
        limit : Optional[int]
            Limit for the number of candidates to return. Default: None

        smoothing_param : Optional[float]
            Smoothing parameter for the scaled likelihood score.  Likelihood
            scores are scaled using the formula
            (likelihood_score - 1)/(M + smoothing_param - 1)
            where M is the maximum likelihood score between a candidate node
            and all of its ancestors and descendents (technically this score
            can be less than 0 for some poor candidate longforms).
            Larger values of smoothing_param lead to more penalization of
            candidates with small count. Default: 4

        use_alignment_based_scoring : Optional[bool]
            If true use combined acromine/alignment scoring. Alignment
            scores will be computed with default parameters if they have
            not been computed previously using the compute_alignment_scores
            method. The combined score is a weighted average of the acromine
            score and the alignment based score, with weight for the alignment
            based score decaying exponentially with M, the maximum likelihood
            score between a candidate node and all of its ancestors and
            descendents.

        weight_decay_param : Optional[float]
            Adjusts rate of decay for alignment score with respect to the
            Value of M (maximum likelihood score between a candidate node and
            all of its ancestors and descendents.)

            score = weight*alignment_score + (1-weight)*likelihood_score

            where weight = e^{-weight_decay_param*max(M, 0)}

        max_length : Optional[str|int|None]
            Maximum number of tokens in an accepted longform. If None, accepted
            longforms can be arbitrarily long. If 'auto', max_length is set
            to 2*len(self.shortform)+1

        Returns
        ------
        candidates : list of tuple
            List of tuples, each containing a candidate string, it's associated
            score, and it's count. Sorted first in descending order by
            likelihood score, then by count, and length from shortest
            to longest candidate measured in number of tokens, and finally by
            lexicographic order.
        """
        if max_length == 'auto':
            max_length = 2*len(self.shortform) + 1
        score_func = self._get_score_function(smoothing_param,
                                              use_alignment_based_scoring,
                                              weight_decay_param)
        root = self._internal_trie
        stack = [(root, 0)]
        result = []
        while stack:
            current, depth = stack.pop()
            if max_length is not None and depth + 1 > max_length:
                continue
            for child in current.children.values():
                score, count = score_func(child)
                result.append([child.longform,
                               self._make_readable(child.longform),
                               count, score])
                stack.append((child, depth+1))
        result.sort(key=lambda x: (-x[3], -x[2], len(x[0]), x[1]))
        return [(longform, score, count)
                for _, longform, score, count in result[:limit]]

    def get_longforms(self, cutoff=0.1, smoothing_param=4,
                      max_length='auto', use_alignment_based_scoring=True,
                      weight_decay_param=0.001):
        """Return a list of extracted longforms with their scores

        Traverse the candidates trie to search for nodes with score
        greater than or equal to the scores of all children and strictly
        greater than the scores of all ancestors.

        Parameters
        ----------
        cutoff : Optional[int]
            Return only longforms with a score greater than the cutoff.
            Default: 0.1

        smoothing_param : Optional[float]
            Smoothing parameter for the scaled likelihood score.  Likelihood
            scores are scaled using the formula
            (likelihood_score - 1)/(M + smoothing_param - 1)
            where M is the maximum likelihood score between a candidate node
            and all of its ancestors and descendents (technically this score
            can be less than 0 for some poor candidate longforms).
            Larger values of smoothing_param lead to more penalization of
            candidates with small count. Default: 4

        use_alignment_based_scoring : Optional[bool]
            If true use combined acromine/alignment scoring. Alignment
            scores will be computed with default parameters if they have
            not been computed previously using the compute_alignment_scores
            method. The combined score is a weighted average of the acromine
            score and the alignment based score, with weight for the alignment
            based score decaying exponentially with M, the maximum likelihood
            score between a candidate node and all of its ancestors and
            descendents.

        weight_decay_param : Optional[float]
            Adjusts rate of decay for alignment score with respect to the
            Value of M (maximum likelihood score between a candidate node and
            all of its ancestors and descendents.)

            score = weight*alignment_score + (1-weight)*likelihood_score

            where weight = e^{-weight_decay_param*max(M, 0)}

        max_length : Optional[str|int|None]
            Maximum number of tokens in an accepted longform. If None, accepted
            longforms can be arbitrarily long. If 'auto', max_length is set
            to 2*len(self.shortform)+1

        Returns
        -------
        longforms : list of tuple
            list of triples of the form (longform, count, score)
            It is sorted in descending order by count and then score.
            Ties are resolved through lexicographic order.
        """
        if max_length == 'auto':
            max_length = 2*len(self.shortform)+1

        def _get_longform_helper(node, score_func, depth):
            if not node.children or (max_length is not None and
                                     depth == max_length):
                score, count = score_func(node)
                return [(node.longform, score, count)]
            result = []
            for child in node.children.values():
                child_longforms = _get_longform_helper(child, score_func,
                                                       depth + 1)
                result.extend([(longform, score, count)
                               for longform, score, count in
                               child_longforms if node.is_root() or
                               score > score_func(node)[0]])
            if not result:
                score, count = score_func(node)
                result = [(node.longform, score, count)]
            return result

        score_func = self._get_score_function(smoothing_param,
                                              use_alignment_based_scoring,
                                              weight_decay_param)
        root = self._internal_trie
        longforms = _get_longform_helper(root, score_func, 0)
        # Convert longforms as tuples in reverse order into reader strings
        # mapping stems back to the most frequent token that had been mapped
        longforms = [(longform, score, count)
                     for longform, score, count in longforms
                     if score > cutoff]

        # Map stems to the most frequent word that had been mapped to them.
        # Convert longforms as tuples in reverse order into reader strings
        # mapping stems back to the most frequent token that had been
        # mapped to them. tuple of stemmed tokens can be recovered by
        # tokenizing, stemming, and reversing
        longforms = [(self._make_readable(longform), score, count)
                     for longform, score, count in longforms]

        # Sort in preferred order
        longforms = sorted(longforms, key=lambda x: (-x[2], -x[1], x[0]))

        return [(longform, count, score)
                for longform, score, count in longforms]

    def _propagate_scores(self):
        """Add best descendent and best ancestor likelihood scores for nodes
        """
        root = self._internal_trie
        stack = [root]
        while stack:
            current = stack.pop()
            for _, child in current.children.items():
                if child.score > current.best_ancestor_score:
                    child.best_ancestor_score = child.score
                else:
                    child.best_ancestor_score = current.best_ancestor_score
                stack.append(child)
            if not current.children:
                current.best_descendent_score = current.score
                while (current.parent is not None and
                       (current.best_descendent_score > current.parent.score
                        or not current.parent.best_descendent_score)):
                    parent = current.parent
                    if parent.score > current.best_descendent_score:
                        parent.best_descendent_score = parent.score
                    else:
                        parent.best_descendent_score = \
                            current.best_descendent_score
                    current = parent

    def compute_alignment_scores(self, **params):
        """Compute and add alignment scores to candidate nodes in trie

        Parameters
        ----------
        **params
            Parameters for py:class`AlignmentBasedScorer`
        """
        abs_ = AlignmentBasedScorer(self.shortform, **params)
        root = self._internal_trie
        stack = [root]
        # Perform depth first search calculating scores for each candidate in
        # trie. Alignment score of best ancestor is used to decide how current
        # node is processed (No computation is performed if score cannot be
        # improved. No computation for permutations with inversion count that
        # makes improving on best score impossible.
        while stack:
            current = stack.pop()
            for token, child in current.children.items():
                data = [current.alignment_score, current.encoded_tokens,
                        current.word_prizes, current.best_ancestor_align_score,
                        current.best_char_scores,
                        current.sum_parent_word_scores,
                        current.stop_count]
                new_data = abs_._next_score(token, *data)
                child.alignment_score = new_data[0]
                child.encoded_tokens = new_data[1]
                child.word_prizes = new_data[2]
                child.best_ancestor_align_score = new_data[3]
                child.best_char_scores = new_data[4]
                child.sum_parent_word_scores = new_data[5]
                child.stop_count = new_data[6]
                stack.append(child)
        self._abs_fit = True

    def prune(self, max_depth):
        """Prune away all nodes with depth greater than max_depth

        Parameters
        ----------
        max_depth : int
            Positive integer. Maximum depth for nodes to keep in the candidate
            trie. Corresponds to maximum number of tokens in longforms.
        """
        root = self._internal_trie
        stack = [(root, 0)]
        while stack:
            current, depth = stack.pop()
            if depth + 1 > max_depth:
                for child in current.children.values():
                    child = None
                current.children = {}
                continue
            for child in current.children.values():
                stack.append((child, depth + 1))

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
                if not current.is_root():
                    # we are not at the top of the trie. observed candidate
                    # has a parent
                    # update likelihood of candidate's parent
                    count = current.children[token].count
                    current.update_likelihood(count)
                current = current.children[token]

    def _get_score_function(self, smoothing_param,
                            use_alignment_based_scoring,
                            weight_decay_param):
        """Returns scoring function for determining longforms

        Also computes alignment scores and propagates acromine score
        information for ancestors and descendents in the tree of candidates
        if necessary.
        """
        if not self._scores_propagated:
            self._propagate_scores()
            self._scores_propagated = True

        def scaled_score(node):
            numerator = node.score-1
            denominator = max(node.best_ancestor_score,
                              node.best_descendent_score)
            denominator += smoothing_param - 1
            score = 0 if denominator <= 0 else numerator/denominator
            return score
        if not use_alignment_based_scoring:
            def score_func(node):
                return scaled_score(node), node.count
        else:
            if not self._alignment_scores_computed:
                self.compute_alignment_scores()
                self._alignment_scores_computed = True

            def score_func(node):
                acro_score = scaled_score(node)
                phi = np.exp(-weight_decay_param *
                             max(0, node.best_ancestor_score - 1,
                                 node.best_descendent_score - 1))
                score = phi*node.alignment_score + (1-phi)*acro_score
                return score, node.count
        return score_func

    def _make_readable(self, tokens):
        """Convert longform from internal representation to a human readable one
        """
        return ' '.join(self._stemmer.most_frequent(token)
                        for token in tokens[::-1])

    def to_dict(self):
        """Returns dictionary serialization of AdeftMiner
        """
        out = {}
        out['shortform'] = self.shortform
        out['internal_trie'] = self._internal_trie.to_dict()
        out['stemmer'] = self._stemmer.dump()
        out['window'] = self.window
        return out

    def dump(self, f):
        """Serialize AdeftMiner to json into file f"""
        json.dump(self.to_dict(), f)

    def update(self, adeft_miner):
        """Compose two adeft miners trained on separate texts"""
        self._stemmer.counts.update(adeft_miner._stemmer.counts)
        stack = [(self._internal_trie,
                  deepcopy(adeft_miner._internal_trie))]
        while stack:
            left, right = stack.pop()
            for token, child in right.children.items():
                if token not in left.children:
                    left.children[token] = child
                    if not left.is_root():
                        left.update_likelihood(child.count, child.count)
                else:
                    current = left.children[token]
                    current.increment_count(child.count)
                    if not left.is_root():
                        count1, count2 = current.count, child.count
                        left.update_likelihood(count1, count2)
                    stack.append((current, child))


def load_adeft_miner_from_dict(dictionary):
    """Loads an AdeftMiner from dictionary serialization

    Parameters
    ---------
    dictionary : dict
        Dictionary representation of AdeftMiner as returned by its
        dump method

    Returns
    -------
    py:class`AdeftMiner`
    """
    shortform = dictionary['shortform']
    out = AdeftMiner(shortform, window=dictionary['window'])
    out._internal_trie = load_trie(dictionary['internal_trie'], shortform)
    out._stemmer = WatchfulStemmer(dictionary['stemmer'])
    return out


def compose(*adeft_miners):
    output = deepcopy(adeft_miners[0])
    for miner in adeft_miners[1:]:
        output.update(miner)
    return output


def load_adeft_miner(f):
    """Load AdeftMiner from file f"""
    return load_adeft_miner_from_dict(json.load(f))
