"""Discover candidate longforms from a given corpus using the Acromine
algorithm."""
import json
import math
import logging
from copy import deepcopy
from ast import literal_eval
from collections import deque

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
    """
    __slots__ = ['longform', 'count', 'sum_ft', 'sum_ft2', 'score',
                 'parent', 'children', 'encoded_tokens', 'word_prizes',
                 'best_ancestor_align_score', 'sum_ancestor_word_scores',
                 'best_ancestor_char_scores', 'best_char_scores',
                 'alignment_score']

    def __init__(self, longform=(), parent=None):
        self.longform = longform
        if longform:
            self.count = 1
            self.sum_ft = self.sum_ft2 = 0
            self.score = 1
        self.parent = parent
        self.children = {}
        self.encoded_tokens = []
        self.word_prizes = []
        self.sum_ancestor_word_scores = 0
        self.best_ancestor_align_score = -1
        self.alignment_score = 0

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


def load_trie(trie_dict):
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
    root = _TrieNode()
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
    def __init__(self, shortform, window=100, **params):
        self.shortform = shortform
        self._internal_trie = _TrieNode()
        self._internal_trie.best_char_scores = [-1e20]*len(shortform)
        self._longforms = {}
        self._stemmer = WatchfulStemmer()
        self._abs = AlignmentBasedScorer(shortform)
        self.window = window

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
                    candidate, _ = get_candidate(fragment)
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

    def get_longforms(self, cutoff=1, scale=False, smoothing_param=4,
                      max_length=None):
        """Return a list of extracted longforms with their scores

        Traverse the candidates trie to search for nodes with score
        greater than or equal to the scores of all children and strictly
        greater than the scores of all ancestors.

        Parameters
        ----------
        cutoff : Optional[int]
            Return only longforms with a score greater than the cutoff.
            Default: 1

        scale : Optional[bool]
            Whether or not to scale likelihood scores. If True, a likelihood
            score is transformed to (score-1)/(count + smoothing_param-1) where
            smoothing_param can be supplied by the user. Default: False

        smoothing_param : Optional[float]
            Value of smoothing parameter to use in scaling transformation. This
            is ignored if scale is set to False. Default: 4

        max_length : Optional[str|int|None]
            Maximum number of tokens in an accepted longform. If None, accepted
            longforms can be arbitrarily long. If 'auto', max_length is set
            to 2*len(self.shortform)+1

        Returns
        -------
        longforms : list of tuple
            list of longforms along with their scores. It is sorted first in
            descending order by score, then by the length of the longform from
            shortest to longest, and finally by lexicographic order.
        """
        if max_length == 'auto':
            max_length = 2*len(self.shortform)+1
        if scale:
            def score_func(score, count):
                numerator = score-1
                denominator = count+smoothing_param-1
                return 0 if denominator <= 0 else numerator/denominator

            cutoff = 0
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
                     for longform, score in longforms
                     if max_length is None or len(longform) <= max_length]

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

    def compute_alignment_scores(self, **params):
        self._abs = AlignmentBasedScorer(self.shortform, **params)
        root = self._internal_trie
        queue = deque([root])
        while queue:
            current = queue.pop()
            for token, child in current.children.items():
                stopcount = self._abs.count_leading_stopwords(child.longform,
                                                              reverse=True)
                leading_stop_penalty = self._abs.zeta**stopcount
                best_score = current.best_ancestor_align_score
                w = self._abs.get_word_score(token)
                child.word_prizes = current.word_prizes + [w]
                child.sum_ancestor_word_scores = \
                    current.sum_ancestor_word_scores + w
                W = child.sum_ancestor_word_scores
                if not (set(token) & set(self._abs.char_map)):
                    multiplier = ((W - w)/W)**(1 - self._abs.lambda_)
                    child.alignment_score = current.alignment_score * \
                        multiplier * leading_stop_penalty
                    child.best_ancestor_align_score = best_score
                encoded_token = self._abs.encode_token(token)
                child.encoded_tokens = current.encoded_tokens + [encoded_token]
                token_char_scores = self._abs.probe(encoded_token)
                char_score_upper_bound = sum(max(a, b, 0) for a, b in
                                             zip(current.best_char_scores,
                                                 token_char_scores))
                char_score_upper_bound /= len(self._abs.encoded_shortform)
                word_score_upper_bound = \
                    self._abs.opt_selection(current.word_prizes,
                                            len(self.shortform)-1)
                word_score_upper_bound += w
                word_score_upper_bound /= W
                upper_bound = (char_score_upper_bound**self._abs.lambda_ *
                               word_score_upper_bound**(1-self._abs.lambda_))
                if upper_bound <= best_score:
                    multiplier = ((W - w)/W)**(1 - self._abs.lambda_)
                    child.alignment_score = current.alignment_score * \
                        multiplier * leading_stop_penalty
                    child.best_ancestor_align_score = best_score
                    continue
                max_inversions = self._abs.inversions_cap if best_score <= 0 \
                    else math.floor(math.log(best_score/upper_bound,
                                             self._abs.rho))
                max_inversions = min(self._abs.inversions_cap, max_inversions)
                current_score, char_scores = \
                    self._abs.score(child.encoded_tokens[::-1],
                                    child.word_prizes[::-1], W,
                                    max_inversions)
                current_score *= leading_stop_penalty
                child.alignment_score = current_score
                child.best_char_scores = char_scores
                child.best_ancestor_align_score = max(current_score,
                                                      best_score)
                queue.appendleft(child)

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

    def to_dict(self):
        """Returns dictionary serialization of AdeftMiner
        """
        out = {}
        out['shortform'] = self.shortform
        out['internal_trie'] = self._internal_trie.to_dict()
        out['longforms'] = {str(key): value
                            for key, value in self._longforms.items()}

        out['stemmer'] = self._stemmer.dump()
        out['window'] = self.window
        return out

    def dump(self, f):
        """Serialize AdeftMiner to json into file f"""
        json.dump(self.to_dict(), f)

    def update(self, adeft_miner):
        """Compose two adeft miners trained on separate texts"""
        self._stemmer.counts.update(adeft_miner._stemmer.counts)
        queue = deque([(self._internal_trie,
                        deepcopy(adeft_miner._internal_trie))])
        self._longforms.update({key: value for key, value in
                                adeft_miner._longforms.items()
                                if key not in self._longforms})
        while queue:
            left, right = queue.pop()
            for token, child in right.children.items():
                if token not in left.children:
                    left.children[token] = child
                    if not left.is_root():
                        left.update_likelihood(child.count, child.count)
                        self._longforms[left.longform[::-1]] = left.score
                else:
                    current = left.children[token]
                    current.increment_count(child.count)
                    self._longforms[current.longform[::-1]] = current.score
                    if not left.is_root():
                        count1, count2 = current.count, child.count
                        left.update_likelihood(count1, count2)
                        self._longforms[left.longform[::-1]] = left.score
                    queue.appendleft((current, child))


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
    out = AdeftMiner(dictionary['shortform'], window=dictionary['window'])
    out._internal_trie = load_trie(dictionary['internal_trie'])
    out._longforms = {literal_eval(key): value
                      for key, value in dictionary['longforms'].items()}
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
