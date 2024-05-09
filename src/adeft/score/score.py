import math
from unicodedata import category

from adeft.nlp import stopwords_min
from adeft.score._score import score, optimize_alignment


class AlignmentBasedScorer(object):
    def __init__(self, shortform, penalties=None,
                 alpha=0.2, beta=0.95, gamma=0.95, delta=1.0,
                 epsilon=0.4, lambda_=0.6, rho=0.95, zeta=0.9,
                 word_scores=None, inversions_cap=16):
        self.shortform = shortform.lower()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon
        self.rho = rho
        self.lambda_ = lambda_
        self.zeta = zeta
        self.inversions_cap = inversions_cap
        char_map = {}
        encoded_shortform = []
        j = 0
        for char in self.shortform:
            if char not in char_map:
                char_map[char] = j
                j += 1
            encoded_shortform.append(char_map[char])
        self.char_map = char_map
        self.encoded_shortform = encoded_shortform
        if penalties is not None:
            self.penalties = penalties
        else:
            # Punctuation and space characters have no penalty
            self.penalties = [delta*epsilon**i
                              if not (category(char).startswith('Z') or
                                      category(char).startswith('C'))
                              else 0.0
                              for i, char in enumerate(shortform)]
        if word_scores is None:
            self.word_scores = {word: 0.2 for word in stopwords_min}
        else:
            self.word_scores = word_scores

    def _next_score(self, token, previous_score, previous_encoded_tokens,
                    previous_word_scores, previous_best_score,
                    previous_best_char_scores, previous_sum_word_scores,
                    previous_stop_count):
        encoded_shortform = self.encoded_shortform
        if token in stopwords_min:
            stop_count = previous_stop_count + 1
        else:
            stop_count = 0
        leading_stop_penalty = self.zeta**stop_count
        word_score = self.get_word_score(token)
        sum_word_scores = previous_sum_word_scores + word_score
        word_scores = previous_word_scores + [word_score]
        if not (set(token) & set(self.char_map)):
            multiplier = ((sum_word_scores - word_score) /
                          sum_word_scores)**(1 - self.lambda_)
            score = previous_score*multiplier*leading_stop_penalty
            return (score, previous_encoded_tokens, word_scores,
                    previous_best_score, previous_best_char_scores,
                    sum_word_scores, stop_count)
        encoded_token = self.encode_token(token)
        encoded_tokens = previous_encoded_tokens + [encoded_token]
        token_char_scores = self.probe(encoded_token)
        char_score_upper_bound = sum(max(a, b, 0) for a, b in
                                     zip(previous_best_char_scores,
                                         token_char_scores))
        char_score_upper_bound /= len(encoded_shortform)
        word_score_upper_bound = \
            self.opt_selection(word_scores[:-1],
                               len(encoded_shortform)-1)
        word_score_upper_bound += word_score
        word_score_upper_bound /= sum_word_scores
        upper_bound = (char_score_upper_bound**self.lambda_ *
                       word_score_upper_bound**(1-self.lambda_))
        if upper_bound < previous_best_score:
            multiplier = ((sum_word_scores - word_score) /
                          sum_word_scores)**(1 - self.lambda_)
            score = previous_score*multiplier*leading_stop_penalty
            return (score, previous_encoded_tokens, word_scores,
                    previous_best_score, previous_best_char_scores,
                    sum_word_scores, stop_count)
        max_inversions = self.inversions_cap if previous_best_score <= 0 else \
            math.floor(math.log(previous_best_score/upper_bound, self.rho))
        max_inversions = min(self.inversions_cap, max_inversions)
        current_score, char_scores = \
            self.score(encoded_tokens[::-1], word_scores[::-1],
                       sum_word_scores, max_inversions)
        score = current_score * leading_stop_penalty
        if current_score >= previous_best_score:
            best_score = current_score
            best_char_scores = char_scores
        else:
            best_score = previous_best_score
            best_char_scores = previous_best_char_scores
        return (score, encoded_tokens, word_scores, best_score,
                best_char_scores, sum_word_scores, stop_count)

    def expanding_score(self, tokens):
        scores = []
        if not tokens:
            return []
        data = [0, [], [], -1, [-1e20]*len(self.shortform), 0, 0]
        for token in tokens[::-1]:
            data = self._next_score(token, *data)
            scores.append(data[0])
        return scores

    def encode_token(self, token):
        """Convert list of tokens to info needed to solve optimization problem

        Parameters
        ----------
        candidates : list
            List of tokens that appear in a defining pattern (DP)
            ['that', 'appear', 'in', 'a', 'defining', 'pattern']

        Returns
        -------
        encoded_candidates : list of list
            Characters in shortform are encoded with natural numbers
            Each element of encoded_candidates corresponds to a token
            in candidates that contains one of the characters in the
            shortform. These elements contain the natural number encodings
            for all characters in the token that are also in the shortform,
            in the order in which the appear in the longform. For example,
            in DP, D would be encoded as 0 and P as 1. The encoded_candidates
            corresponding to ['that', 'appear', 'in', 'a', 'defining',
                              'pattern']
            are [[1, 1], [0], [1]].
        indices : list of list
            List of lists of the same shape as encoded_candidates. For each
            token, the associated list contains the indices of the characters
            in the token that are also in the shortform. The indices list for
            the above example is [[1, 2], [0], [0]].
        word_prizes : list
            Token prizes for each token in candidates that contains a character
            overlapping with the shortform. For the above example this will be
            [1.0, 1.0, 1.0] if 'appear', 'defining', and 'pattern' do not
            appear in self.word_scores
         W_array : list of double
            The kth element contains the sum of all prizes for the last k+1
            tokens in candidates, regardless of whether they have a character
            in common with the shortform. These are used for calculating word
            scores for a match. The word score is the sum of prizes for all
            captured tokens divided by the sum of all prizes for tokens in
            a candidate.
        """
        encoded_token = [(self.char_map[c], i) for i, c in enumerate(token)
                         if c in self.char_map]
        return encoded_token

    def probe(self, encoded_token):
        encoded_shortform = self.encoded_shortform
        if not encoded_token:
            return [0.0]*len(encoded_shortform)
        woven_token = [-1]*(len(encoded_shortform)-1)
        woven_indices = [-1]*(len(encoded_shortform)-1)
        for value, index in encoded_token:
            woven_token.append(value)
            woven_indices.append(index)
            woven_token.extend([-1]*(len(encoded_shortform) - 1))
            woven_indices.extend([-1]*(len(encoded_shortform) - 1))
        penalties = [0.0]*len(encoded_shortform)
        word_prizes = [0.0]
        W = 1.0
        score, char_scores = \
            optimize_alignment(woven_token, woven_indices,
                               encoded_shortform,
                               word_prizes, W, penalties, self.alpha,
                               self.beta, self.gamma, 1.0)
        return char_scores

    def score(self, encoded_tokens, word_prizes, max_word_score,
              max_inversions, max_perm_length=9):
        encoded_shortform = self.encoded_shortform
        if not encoded_tokens:
            return (0, [0.0]*len(encoded_shortform))
        return score(encoded_tokens, encoded_shortform, word_prizes,
                     max_word_score, self.penalties, max_inversions,
                     max_perm_length, self.alpha, self.beta, self.gamma,
                     self.lambda_, self.rho)

    def count_leading_stopwords(self, tokens, stopwords=stopwords_min,
                                reverse=False):
        count = 0
        n = len(tokens)
        for i in range(n):
            index = i if not reverse else n-i-1
            if tokens[index] in stopwords:
                count += 1
            else:
                break
        return count

    def opt_selection(self, word_prizes, k):
        """Find the sum of the largest k elements in list"""
        if k >= len(word_prizes):
            return sum(word_prizes)
        for i in range(k):
            max_value = word_prizes[i]
            for j in range(i+1, len(word_prizes)):
                if word_prizes[j] > max_value:
                    max_value = word_prizes[j]
                    word_prizes[i], word_prizes[j] = \
                        word_prizes[j], word_prizes[i]
        return sum(word_prizes[:k])

    def get_word_score(self, token):
        """Calculate scores for tokens in longform"""
        if token in self.word_scores:
            return self.word_scores[token]
        else:
            return 1.0
