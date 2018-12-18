from collections import deque

from deft.extraction import Processor
from deft.nlp.stem import SnowCounter


class TrieNode(object):
    __slots__ = ['longform', 'count', 'sum_ft', 'sum_ft2', 'score',
                 'parent', 'children']
    """ Node in Trie associated to a candidate longform

    The children of a node associated to a candidate longform c are all
    observed candidates t that can be obtained by prepending a single token
    to c.

    Contains the current likelihood for the candidate longform as well
    as all information needed to calculate it. The likelihood's are
    updated as candidates are added to the Trie.


    Parameters
    ----------
    longform: list of str
        List of tokens within a candidate longform in reverse order.
        For the candidate longform "in the endoplasmic reticulum" the
        list will take the form ["reticulum", "endoplasmic", "the", "in"],
        ignoring that the tokens should actually be stemmed.

    Attributes
    ----------

    longform: list of str
        Explained above

    count: int
        Current co-occurence frequency of candidate longform with shortform

    sum_ft: int
        Sum of the co-occurence frequencies of all previously observed
        candidate longforms that are children of the associated candidate
        longform (longforms that can be obtained by prepending
        one token to the associated candidate longform).

    sum_ft2: int
        Sum of the squares of the co-occurence freqencies of all previously
        observed candidate longforms that are children of the associated
        longform.
    score: float
        Likelihood score of the associated candidate longform.
        It is given by count - sum_ft**2/sum_ft
        See

        [Okazaki06] Naoaki Okazaki and Sophia Ananiadou. "Building an
        abbreviation dicationary using a term recognition approach".
        Bioinformatics. 2006. Oxford University Press.

        for more information

    parent: :py:class:`deft.mine.TrieNode`
        link to node's parent

    children: dict of :py:class:`deft.mine.ContinuousMiner.TrieNode`
        dictionary of child nodes
    """
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
        """Update count and likelihood when observing a longform again"""
        self.count += 1
        self.score += 1

    def update_likelihood(self, count):
        """Update likelihood when observing a child of associated longform

        Update when observing a candidate longform which can be obtained
        by prepending one token to the associated longform.

        Parameters
        ----------
        count: int
            Current co-occurence frequency of child longform with shortform
        """
        self.score += self.sum_ft2/self.sum_ft if self.sum_ft else 0
        self.sum_ft += 1
        self.sum_ft2 += 2*count - 1
        self.score -= self.sum_ft2/self.sum_ft


class ContinuousMiner(object):
    __slots__ = ['shortform', '_internal_trie',
                 '_longforms', '_snow', 'processor']
    """Finds possible longforms corresponding to an abbreviation in a text corpus

    An online method. Updates likelihoods of terms as it consumes
    additional texts. Based on a trie datastructure. Likelihoods for longforms
    are updated at the time of insertions into the trie.

    Parameters
    ----------
    shortform: str
        Search for candidate longforms associated to this shortform

    exclude: Optional[set of str]
        Terms that are to be excluded from candidate longforms.
        Default: None

    Attributes
    ----------
    _internal_trie: :py:class:`deft.mine.TrieNode`
        Stores trie datastructure that algorithm is based upon

    _longforms: dict
        Dictionary mapping candidate longforms to their likelihoods as
        produced by the acromine algorithm

    _snow: :py:class:`deft.nlp.stem.SnowCounter`
        English stemmer that keeps track of counts of the number of times a
        given word has been mapped to a given stem. Wraps the class
        EnglishStemmer from nltk.stem.snowball
    """
    def __init__(self, shortform, exclude=None):
        self.shortform = shortform
        self._internal_trie = TrieNode()
        self._longforms = {}
        self._snow = SnowCounter()
        self.processor = Processor(shortform, exclude)

    def consume(self, texts):
        """Consume a corpus of texts and use them to train the miner

        Every suffix is then considered as a candidate longform and
        added to the internal trie, which updates the likelihoods for each of
        the candidates.

        If there are no excluded stop words, for the sentence "The growth of
        estrogen receptor (ER)-positive breast cancer cells is inhibited by
        all-trans-retinoic acid (RA)." The maximal candidate longform would be
        "the growth of estrogen receptor". It and the candidates
        "growth of estrogen receptor", "of estrogen receptor",
        "estrogen receptor", "receptor" would then be added to the internal
        trie.

        Parameters
        ----------
        texts: list of str
            A list of texts
        """
        # split each text into a list of sentences
        for text in texts:
            candidates, _ = self.processor.extract(text)
            for candidate in candidates:
                self._add(candidate)

    def top(self, limit=None):
        """Return top scoring candidates from the mine.

        Parameters
        ----------
        limit: Optional[int]
            Limit for the number of candidates to return. Default: None

        Returns
        ------
        candidates: list of tuple
            List of tuples, each containing a candidate string and its
            likelihood score. Sorted first in descending order by
        likelihood score, then by length from shortest to longest, and finally
        by lexicographic order.
        """
        if not self._longforms:
            return []

        candidates = sorted(self._longforms.items(), key=lambda x:
                            (-x[1], len(x[0]), x[0]))
        if limit is not None and limit < len(candidates):
            candidates = candidates[0:limit]
        # Map stems back to the most frequent word that had been mapped to them
        # and convert longforms in tuple format into readable strings.
        candidates = [(' '.join(self._snow.most_frequent(token)
                                for token in longform),
                       score)
                      for longform, score in candidates]
        return candidates

    def get_longforms(self, cutoff=1):
        """Return a list of longforms extracted from the mine with their scores

        The extracted longforms are found by taking the first local maximum
        along each path from root to leaf. This works because the score
        function first increases and then decreases (not necessarily strictly).
        This is done with a breadth-first tree traversal. In an initial
        forward pass, the tree is traversed until every node with a greater
        than equal score to its parent and a greater score than all of its
        children is found. These nodes are placed in a list.

        In a second backward pass, for each node in the list, its parent link
        is followed for as long as the parent has an equal score. In this way,
        we find longforms of maximal score and minimal length.

        Parameters
        ----------
        cutoff: Optional[int]
            Return only longforms with a score greater than the cutoff.
            Default: 1

        Returns
        -------
        longforms: list of tuple
        list of longforms along with their scores. It is sorted first in
        descending order by score, then by the length of the longform from
        shortest to longest, and finally by lexicographic order.
        """
        # Forward pass
        leaves = []
        # The root contains no longform. Initialize queue with all of its
        # children
        queue = deque(self._internal_trie.children.values())
        while queue:
            node = queue.popleft()
            # count the number of worthy children with greater than or equal
            # score to their parent.
            worthy = 0
            for child in node.children.values():
                # only place a node in the queue if it is worthy
                if child.score >= node.score:
                    queue.append(child)
                    worthy += 1
            # If there are no worthy children, add the node to list of leaves.
            if worthy == 0:
                leaves.append(node)
        # Backward pass
        # to contain tuple of longforms and their scores
        longforms = set([])
        # loop through all leaves found in the forward pass
        for leaf in leaves:
            current = leaf
            # Follow the parent links until the root is reached or until the
            # parent has a lower score than its child
            while (not current.parent.is_root() and
                   current.score == current.parent.score):
                current = current.parent
            # Add the found longform and its score to the set. The set allows
            # us to ignore that there will be duplicates
            longforms.add((current.longform, current.score))
        # Map stems to the most frequent word that had been mapped to them.
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
        return longforms

    def _add(self, tokens):
        """Add a list of tokens to the internal trie and update likelihoods.

        Parameters
        ----------
        tokens: str
            A list of tokens to add to the internal trie.

        """
        # start at top of trie
        current = self._internal_trie
        # apply snowball stemmer to each token and put them in reverse order
        tokens = tuple(self._snow.stem(token) for token in tokens)[::-1]
        for token in tokens:
            if token not in current.children:
                # candidate longform is observed for the first time
                # add a new entry for it in the trie
                longform = current.longform + (token, )
                new = TrieNode(longform, parent=current)
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
        return ' '.join(self._snow.most_frequent(token)
                        for token in tokens[::-1])
