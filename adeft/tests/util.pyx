from adeft.score.score cimport int_array, opt_results, candidates_array, \
    opt_input, opt_params, opt_shortform
from adeft.score.score cimport make_int_array, free_int_array, \
    make_opt_results, free_opt_results, make_opt_input, free_opt_input, \
    make_opt_params, free_opt_params, make_opt_shortform, free_opt_shortform, \
    make_candidates_array, free_candidates_array
from adeft.score.score cimport optimize, stitch, perm_search

cdef class StitchTestCase:
    """Test construction of candidates array and stitching"""
    cdef:
        list candidates, prizes, word_prizes
        list permutation, result_x, result_prizes, result_word_prizes
        list result_word_boundaries, W_array
    def __init__(self, candidates=None, prizes=None, word_prizes=None,
                 permutation=None, W_array=None,
                 result_x=None, result_prizes=None, result_word_prizes=None,
                 result_word_boundaries=None):
        self.candidates = candidates
        self.prizes = prizes
        self.word_prizes = word_prizes
        self.W_array = W_array
        self.permutation = permutation
        self.result_x = result_x
        self.result_prizes = result_prizes
        self.result_word_prizes = result_word_prizes
        self.result_word_boundaries = result_word_boundaries

    def run_test(self):
        cdef:
            opt_input *input_
            int_array *perm
            candidates_array *candidates

        candidates = make_candidates_array(self.candidates,
                                           self.prizes,
                                           self.word_prizes,
                                           self.W_array)
        n = len(self.permutation)
        perm = make_int_array(n)
        for i in range(n):
            perm.array[i] = self.permutation[i]
        total_length = candidates.cum_lengths[n - 1]
        input_ = make_opt_input(2*total_length + 1, n)
        stitch(candidates, perm.array, n, input_)
        x, p, wp, wb = [], [], [], []
        length = input_.x.length
        for i in range(length):
            x.append(input_.x.array[i])
            p.append(input_.prizes.array[i])
        for j in range(input_.word_prizes.length):
            wp.append(input_.word_prizes.array[j])
            wb.append(input_.word_boundaries[j])
        free_candidates_array(candidates)
        free_opt_input(input_)
        free_int_array(perm)
        assert x == self.result_x
        assert p == self.result_prizes
        assert wp == self.result_word_prizes
        assert wb == self.result_word_boundaries


cdef class PermSearchTestCase:
    cdef:
        list shortform, candidates, prizes, penalties, word_prizes
        list word_penalties
        double beta, rho, inv_penalty, result_score
        int len_perm
    def __init__(self, shortform=None, candidates=None, prizes=None,
                 penalties=None, word_prizes=None, word_penalties=None,
                 beta=None, rho=None, inv_penalty=None, len_perm=None,
                 result_score=None):
        self.shortform = shortform
        self.candidates = candidates
        self.prizes = prizes
        self.penalties = penalties
        self.word_prizes = word_prizes
        self.word_penalties = word_penalties
        self.beta = beta
        self.rho = rho
        self.inv_penalty = inv_penalty
        self.len_perm = len_perm
        self.result_score = result_score

    def run_test(self):
        candidates = make_candidates_array(self.candidates, self.prizes,
                                           self.word_prizes,
                                           self.word_penalties)
        shortform = make_opt_shortform(self.shortform, self.penalties)
        params = make_opt_params(self.beta, self.rho)
        score = perm_search(candidates, shortform, params, self.inv_penalty,
                            self.len_perm)
        assert abs(score - self.result_score) < 1e-7


cdef class OptimizationTestCase:
    cdef:
        list x, y, prizes, penalties, word_boundaries, word_prizes
        list result_char_scores
        double beta, rho, C, W, result_score
        int n, m, num_words
    def __init__(self, x=None, y=None,
                 prizes=None, penalties=None,
                 word_boundaries=None, word_prizes=None, beta=None,
                 rho=None, W=None, result_score=None,
                 result_char_scores=None):
        self.x = x
        self.y = y
        self.prizes = prizes
        self.penalties = penalties
        self.word_boundaries = word_boundaries
        self.word_prizes = word_prizes
        self.beta = beta
        self.rho = rho
        self.W = W
        self.n = len(x)
        self.m = len(y)
        self.num_words = len(word_boundaries)
        self.result_score = result_score
        self.result_char_scores = result_char_scores

    def check_assertions(self):
        assert len(self.prizes) == self.n
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
        shortform = make_opt_shortform(self.y, self.penalties)
        params = make_opt_params(self.beta, self.rho)
        output = make_opt_results(self.m)

        input_.W = self.W
        for i in range(self.n):
            input_.x.array[i] = self.x[i]
            input_.prizes.array[i] = self.prizes[i]
        for i in range(self.num_words):
            input_.word_boundaries[i] = self.word_boundaries[i]
            input_.word_prizes.array[i] = self.word_prizes[i]
        for i in range(self.m):
            shortform.y.array[i] = self.y[i]
            shortform.penalties.array[i] = self.penalties[i]

        optimize(input_, shortform, params, output)
        score = output.score
        cs = output.char_scores
        char_scores = []
        for i in range(self.m):
            char_scores.append(cs[i])
        free_opt_results(output)
        free_opt_shortform(shortform)
        free_opt_params(params)
        free_opt_input(input_)
        assert abs(score - self.result_score) < 1e-7
        assert char_scores == self.result_char_scores
