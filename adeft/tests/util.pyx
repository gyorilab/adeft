import logging

from adeft.score.score cimport int_array, opt_results, candidates_array, \
    opt_input, opt_params, opt_shortform
from adeft.score.score cimport make_int_array, free_int_array, \
    make_opt_results, free_opt_results, make_opt_input, free_opt_input, \
    make_opt_params, free_opt_params, create_shortform, free_opt_shortform, \
    make_candidates_array, free_candidates_array
from adeft.score.score cimport optimize, stitch, opt_search

logger = logging.getLogger(__name__)


cdef class StitchTestCase:
    """Test construction of candidates array and stitching"""
    cdef:
        list candidates, indices, word_prizes,
        list permutation, result_x, result_word_prizes, result_indices
        list result_word_boundaries, W_array
    def __init__(self, candidates=None, indices=None,
                 word_prizes=None,
                 permutation=None, W_array=None,
                 result_x=None, result_indices=None, result_word_prizes=None,
                 result_word_boundaries=None):
        self.candidates = candidates
        self.indices = indices
        self.word_prizes = word_prizes
        self.W_array = W_array
        self.permutation = permutation
        self.result_x = result_x
        self.result_indices = result_indices
        self.result_word_prizes = result_word_prizes
        self.result_word_boundaries = result_word_boundaries

    def run_test(self):
        cdef:
            opt_input *input_
            int_array *perm
            candidates_array *candidates

        candidates = make_candidates_array(self.candidates,
                                           self.indices,
                                           self.word_prizes,
                                           self.W_array)
        n = len(self.permutation)
        perm = make_int_array(n)
        for i in range(n):
            perm.array[i] = self.permutation[i]
        total_length = candidates.cum_lengths[n - 1]
        input_ = make_opt_input(2*total_length + 1, n)
        stitch(candidates, perm.array, n, input_)
        x, ind, wp, wb = [], [], [], []
        length = input_.x.length
        for i in range(length):
            x.append(input_.x.array[i])
            ind.append(input_.indices.array[i])
        for j in range(input_.word_prizes.length):
            wp.append(input_.word_prizes.array[j])
            wb.append(input_.word_boundaries[j])
        free_candidates_array(candidates)
        free_opt_input(input_)
        free_int_array(perm)
        assert x == self.result_x
        assert ind == self.result_indices
        assert wp == self.result_word_prizes
        assert wb == self.result_word_boundaries


cdef class PermSearchTestCase:
    cdef:
        list shortform, candidates, indices, penalties, word_prizes
        list word_penalties
        double alpha, beta, gamma, lambda_, rho, result_score
        int len_perm
    def __init__(self, shortform=None, candidates=None, indices=None,
                 penalties=None, word_prizes=None, word_penalties=None,
                 alpha=None, beta=None, gamma=None, lambda_=None, rho=None,
                 len_perm=None,
                 result_score=None):
        self.shortform = shortform
        self.candidates = candidates
        self.indices = indices
        self.penalties = penalties
        self.word_prizes = word_prizes
        self.word_penalties = word_penalties
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lambda_ = lambda_
        self.rho = rho
        self.len_perm = len_perm
        self.result_score = result_score

    def run_test(self):
        cdef:
            candidates_array *candidates
            opt_shortform *shortform
            opt_params *params
            opt_results *results
        candidates = make_candidates_array(self.candidates,
                                           self.indices,
                                           self.word_prizes,
                                           self.word_penalties)
        shortform = create_shortform(self.shortform, self.penalties)
        params = make_opt_params(self.alpha, self.beta, self.gamma,
                                 self.lambda_)
        results = make_opt_results(len(self.shortform))
        opt_search(candidates, shortform, params, self.rho,
                   self.len_perm, 100, results)
        assert abs(results.score - self.result_score) < 1e-7


cdef class OptimizationTestCase:
    cdef:
        list x, y, indices, penalties, word_boundaries, word_prizes
        list result_char_scores
        double alpha, beta, gamma, lambda_, C, W, result_score
        int n, m, num_words
    def __init__(self, x=None, y=None, indices=None, penalties=None,
                 word_boundaries=None, word_prizes=None, alpha=None,
                 beta=None, gamma=None, lambda_=None, W=None,
                 result_score=None, result_char_scores=None):
        self.x = x
        self.y = y
        self.indices = indices
        self.penalties = penalties
        self.word_boundaries = word_boundaries
        self.word_prizes = word_prizes
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lambda_ = lambda_
        self.W = W
        self.n = len(x)
        self.m = len(y)
        self.num_words = len(word_boundaries)
        self.result_score = result_score
        self.result_char_scores = result_char_scores

    def check_assertions(self):
        assert len(self.penalties) == self.m
        assert len(self.word_prizes) == self.num_words
        assert self.word_boundaries[-1] == len(self.x) - 1
        assert self.word_boundaries == sorted(self.word_boundaries)

    def run_test(self):
        cdef:
            opt_input *input_
            opt_shortform *shortform
            opt_params *opt_params
            opt_results *output

        input_ = make_opt_input(self.n, self.num_words)
        shortform = create_shortform(self.y, self.penalties)
        params = make_opt_params(self.alpha, self.beta, self.gamma,
                                 self.lambda_)
        output = make_opt_results(self.m)

        input_.W = self.W
        for i in range(self.n):
            input_.x.array[i] = self.x[i]
            input_.indices.array[i] = self.indices[i]
        for i in range(self.num_words):
            input_.word_boundaries[i] = self.word_boundaries[i]
            input_.word_prizes.array[i] = self.word_prizes[i]
        for i in range(self.m):
            shortform.y.array[i] = self.y[i]
            shortform.penalties.array[i] = self.penalties[i]

        optimize(input_, shortform, params, output)
        score = output.score
        cs = output.char_prizes
        char_scores = []
        for i in range(self.m):
            char_scores.append(cs[i])
        free_opt_results(output)
        free_opt_shortform(shortform)
        free_opt_params(params)
        free_opt_input(input_)
        assert abs(score - self.result_score) < 1e-7
        assert all([abs(expected - observed) < 1e-7
                   for observed, expected in
                   zip(char_scores, self.result_char_scores)])
