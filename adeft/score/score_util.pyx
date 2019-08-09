import numpy as np
from libc.math cimport pow as cpow
from cython cimport boundscheck, wraparound
from cpython.mem cimport PyMem_Malloc, PyMem_Free

from adeft.score.permutations cimport permuter, make_permuter, \
    free_permuter, update_permuter


cdef struct opt_results:
    double score
    int *indices
    int chars_matched


cdef opt_results *make_opt_results(int len_y):
    cdef opt_results *results
    results = <opt_results *> PyMem_Malloc(sizeof(opt_results))
    results.indices = <int *> PyMem_Malloc(len_y * sizeof(int))
    return results


cdef void free_opt_results(opt_results *results):
    PyMem_Free(results.indices)
    PyMem_Free(results)
    return

cdef struct int_array:
    int *array
    int length


cdef int_array *make_int_array(int length):
    cdef int_array *output
    output = <int_array *> PyMem_Malloc(sizeof(int_array))
    output.array = <int *> PyMem_Malloc(length * sizeof(int))
    output.length = length
    return output


cdef void free_int_array(int_array *x):
    PyMem_Free(x.array)
    PyMem_Free(x)
    return


cdef struct double_array:
    double *array
    int length

    
cdef double_array *make_double_array(int length):
    cdef double_array *output
    output = <double_array *> PyMem_Malloc(sizeof(int_array))
    output.array = <double *> PyMem_Malloc(length * sizeof(double))
    output.length = length
    return output


cdef void free_double_array(double_array *x):
    PyMem_Free(x.array)
    PyMem_Free(x)
    return


cdef struct candidates_array:
    int_array **array
    int_array *y
    double_array **prizes
    double_array *penalties
    double *word_prizes
    int *cum_lengths
    int length
    double inv_penalty
    double alpha


@boundscheck(False)
@wraparound(False)
cdef candidates_array *make_candidates_array(list encoded_shortform,
                                             list encoded_candidates,
                                             list prizes,
                                             list penalties,
                                             list word_prizes,
                                             double inv_penalty,
                                             double alpha):
    cdef:
        int i, j, num_candidates, m1,m2, n, cum_length, k
        candidates_array *candidates
    n = len(encoded_candidates)
    k = len(encoded_shortform)
    candidates = <candidates_array *> PyMem_Malloc(sizeof(candidates_array))
    candidates.array = <int_array **> PyMem_Malloc(n * sizeof(int_array*))
    candidates.prizes = <double_array **> \
        PyMem_Malloc(n * sizeof(double_array*))
    candidates.penalties = make_double_array(k)
    candidates.word_prizes = <double *> PyMem_Malloc(n * sizeof(double))
    candidates.cum_lengths = <int *> PyMem_Malloc(n * sizeof(int))
    candidates.length = n
    candidates.inv_penalty = inv_penalty
    candidates.alpha = alpha
    candidates.y = make_int_array(k)
    for i in range(k):
        candidates.penalties.array[i] = penalties[i]
        candidates.y.array[i] = encoded_shortform[i]
    cum_length = 0
    for i in range(n):
        m1 = len(encoded_candidates[i])
        m2 = len(encoded_candidates[n-i-1])
        candidates.array[i] = make_int_array(m1)
        candidates.prizes[i] = make_double_array(m1)
        cum_length += m2
        candidates.cum_lengths[i] = cum_length
        candidates.word_prizes[i] = word_prizes[i]
        for j in range(m1):
            candidates.array[i].array[j] = encoded_candidates[i][j]
            candidates.prizes[i].array[j] = prizes[i][j]
    return candidates


cdef free_candidates_array(candidates_array *candidates):
    cdef:
        int i, j
    for i in range(candidates.length):
         free_int_array(candidates.array[i])
         free_double_array(candidates.prizes[i])
    free_int_array(candidates.y)
    free_double_array(candidates.penalties)
    PyMem_Free(candidates.word_prizes)
    PyMem_Free(candidates.prizes)
    PyMem_Free(candidates.array)
    PyMem_Free(candidates.cum_lengths)
    PyMem_Free(candidates)


cdef struct opt_input:
    int_array *x
    double_array *prizes
    unsigned int *word_boundaries
    double_array *word_prizes


cdef opt_input *make_opt_input(int n, int num_words):
    cdef opt_input *input_
    input_ = <opt_input *> PyMem_Malloc(sizeof(opt_input))
    input_.x = make_int_array(n)
    input_.prizes = make_double_array(n)
    input_.word_boundaries = <unsigned int *> \
        PyMem_Malloc(num_words * sizeof(unsigned int))
    input_.word_prizes = make_double_array(num_words)
    return input_


cdef void free_opt_input(opt_input *input_):
    free_int_array(input_.x)
    free_double_array(input_.prizes)
    PyMem_Free(input_.word_boundaries)
    PyMem_Free(input_.word_prizes)
    PyMem_Free(input_)
    

@boundscheck(False)
@wraparound(False)
cdef double perm_search(candidates_array *candidates, int n):
    cdef:
        double best, current_score
        permuter *perms
        opt_input *current
        opt_results *results
    results = make_opt_results(candidates.y.length)
    total_length = candidates.cum_lengths[n - 1]
    current = make_opt_input(2*total_length + 1, n)
    perms = make_permuter(n)
    stitch(candidates, perms.P, n, current)
    optimize(current.x, candidates.y, current.prizes,
             candidates.penalties, current.word_boundaries,
             current.word_prizes.array, candidates.alpha, results)
    best = results.score
    while perms.m != 0:
        update_permuter(perms)
        stitch(candidates, perms.P, n, current)
        optimize(current.x, candidates.y, current.prizes,
                 candidates.penalties, current.word_boundaries,
                 current.word_prizes.array, candidates.alpha, results)
        current_score = results.score * cpow(candidates.inv_penalty,
                                             perms.inversions)
        if current_score > best:
            best = current_score
    free_permuter(perms)
    free_opt_results(results)
    free_opt_input(current)
    return best


@boundscheck(False)
@wraparound(False)
cdef void *stitch(candidates_array *candidates, int *permutation,
                  int len_perm, opt_input *result):
    cdef int i, j, k, current_length, n, p
    n = candidates.length
    # stitched output begins with wildcard represented by -1
    result.x.array[0] = -1
    result.prizes.array[0] = 0
    j = 1
    for i in range(len_perm):
        p = permutation[i]
        current_length = candidates.array[n-len_perm+p].length
        for k in range(current_length):
            result.x.array[j] = \
                candidates.array[n-len_perm+p].array[k]
            # insert wildcard after each element from input
            result.x.array[j+1] = -1
            result.prizes.array[j] = \
                candidates.prizes[n-len_perm+p].array[k]
            result.prizes.array[j+1] = 0
            j += 2
        result.word_prizes.array[i] = candidates.word_prizes[n-len_perm+p]
        result.word_boundaries[i] = j - 1
    return result


@boundscheck(False)
@wraparound(False)
cdef void *optimize(int_array *x, int_array *y,
                    double_array *prizes, double_array *penalties,
                    unsigned int *word_boundaries, double *word_prizes,
                    double alpha,
                    opt_results *output):
    """Subsequence match optimization algorithm for longform scoring

    Uses a dynamic programming algorithm to find optimal instance of
    y as a subsequence in x where elements of x each have a corresponding
    prize. Wildcard characters are allowed in x that match any element of y
    and penalties may be given for when an element of y matches a wildcard
    instead of a regular element of x. Results are updated in place.

    Paramters
    ---------
    x : C array of int
        Contains nonnegative long ints for supersequence in which we seek
        a subsequence match. May also contain the value -1 which corresponds to a
        wildcard that matches any nonnegative int.

    y : C array of int
        Sequence we seek an optimal subsequence match of in x

    prizes : C array of double
        Must be the same length as x. Prize gained for matching an element
        of y to the corresponding element of x

    penalties : C array of double
        Must the the same length as y. Penalty lost if the corresponding
        element of y matches a wildcard.

    results : struct opt_results
        opt_results structure to where output is to be placed
    """
    cdef:
        unsigned int n = x.length
        unsigned int m = y.length
        double possibility1, possibility, capture_prize
        unsigned int i, j, k

    # Dynamic initialization of score_lookup array and traceback pointer
    # array
    cdef:
        double **score_lookup = (<double **>
                                 PyMem_Malloc((n+1) * sizeof(double *)))
        int **pointers = <int **> PyMem_Malloc(n * sizeof(int *))

        int **word_use = <int **> PyMem_Malloc((n+1) * sizeof(int *))
    for i in range(n+1):
        score_lookup[i] = <double *> PyMem_Malloc((m+1) * sizeof(double))
        word_use[i] = <int *> PyMem_Malloc((m+1) * sizeof(int))
        if i != n:
            pointers[i] = <int *> PyMem_Malloc(m * sizeof(int))
    # Initialize lookup array
    score_lookup[0][0] = 0
    word_use[0][0] = 0
    for j in range(1, m+1):
        score_lookup[0][j] = -1e20
    for i in range(1, n+1):
        for j in range(0, m+1):
            score_lookup[i][j] = 0
            word_use[i][j] = 0
    # Main loop
    k = 0
    for i in range(1, n+1):
        for j in range(1, m+1):
            # Case where element of x in current position matches
            # element of y in current position. Algorithm considers
            # either accepting or rejecting this match
            if x.array[i-1] == y.array[j-1]:
                if word_use[i-1][j-1] == 0:
                    capture_prize = word_prizes[k]
                else:
                    capture_prize = 0
                possibility1 = score_lookup[i-1][j]
                possibility2 = (score_lookup[i-1][j-1] +
                                prizes.array[i-1]/cpow(alpha,
                                                       word_use[i-1][j-1]) +
                                capture_prize)
                if possibility2 > possibility1:
                    score_lookup[i][j] = possibility2
                    word_use[i][j] = word_use[i-1][j-1] + 1
                    pointers[i-1][j-1] = 1
                else:
                    score_lookup[i][j] = possibility1
                    word_use[i][j] = word_use[i-1][j]
                    pointers[i-1][j-1] = 0
            # Case where element of x in current position is a wildcard.
            # May either accept or reject this match
            elif x.array[i-1] == -1:
                possibility1 = score_lookup[i-1][j]
                possibility2 = score_lookup[i-1][j-1] - penalties.array[j-1]
                if possibility2 > possibility1:
                    score_lookup[i][j] = possibility2
                    word_use[i][j] = word_use[i-1][j-1]
                    pointers[i-1][j-1] = 1
                else:
                    score_lookup[i][j] = possibility1
                    pointers[i-1][j-1] = 0
                    word_use[i][j] = word_use[i-1][j]
            # No match is possible. There is only one option to fill
            # current entry of dynamic programming lookup array.
            else:
                score_lookup[i][j] = score_lookup[i-1][j]
                word_use[i][j] = word_use[i-1][j]
                pointers[i-1][j-1] = 0
            if i == word_boundaries[k] + 1:
                word_use[i][j] = 0
                k += 1
    # Optimal score is in bottom right corner of lookup array
    score = score_lookup[n][m]
    # Free the memory used by the lookup array
    for i in range(n+1):
        PyMem_Free(score_lookup[i])
        PyMem_Free(word_use[i])
    PyMem_Free(score_lookup)
    PyMem_Free(word_use)
    # Set score in output
    output.score = score
    # Trace backwards through pointer array to discover which elements of x
    # were matched and add the corresponding indices to the index array in
    # reverse order
    i, j, k = n, m, 0
    while j > 0:
        if pointers[i - 1][j - 1]:
            i -= 1
            j -= 1
            output.indices[k] = i
            k += 1
        else:
            i -= 1
    # Free pointer array
    for i in range(n):
        PyMem_Free(pointers[i])
    PyMem_Free(pointers)
    # Set the number of chars in y that were matched
    output.chars_matched = k
    return output


cdef struct perm_out:
    double score


cdef class StitchTestCase:
    """Test construction of candidates array and stitching"""
    cdef:
        list shortform, candidates, prizes, penalties, word_prizes
        list permutation, result_x, result_prizes, result_word_prizes
        list result_word_boundaries
        double inv_penalty, alpha
    def __init__(self, shortform=None, candidates=None,
                 prizes=None, penalties=None, word_prizes=None,
                 permutation=None, result_x=None, result_prizes=None,
                 result_word_prizes=None, result_word_boundaries=None,
                 inv_penalty=0.9, alpha=0.5):
        self.shortform = shortform
        self.candidates = candidates
        self.prizes = prizes
        self.penalties = penalties
        self.word_prizes = word_prizes
        self.permutation = permutation
        self.result_x = result_x
        self.result_prizes = result_prizes
        self.result_word_prizes = result_word_prizes
        self.result_word_boundaries = result_word_boundaries
        self.inv_penalty = inv_penalty
        self.alpha = alpha

    def run_test(self):
        cdef:
            int_array *perm
            opt_input *input_
            candidates_array *candidates

        candidates = make_candidates_array(self.shortform,
                                           self.candidates,
                                           self.prizes,
                                           self.penalties,
                                           self.word_prizes,
                                           self.inv_penalty,
                                           self.alpha)
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


def check_perm_search():
    cdef:
        list sf = [1, 0]
        list ca = [[0], [0, 1], [1, 1, 0], [0, 0], [1]]
        list prizes = [[1.0], [0.5, 1.0], [0.25, 0.5, 1.0],
                       [0.5, 1.0], [1.0]]
        list penalties = [0.2, 0.4]
        list word_prizes = [1.0, 1.0, 1.0, 1.0, 1.0]
        candidates_array *candidates
    candidates = make_candidates_array(sf, ca,  prizes, penalties,
                                       word_prizes, 0.9, 0.5)
    score = perm_search(candidates, 5)
    free_candidates_array(candidates)
    return score


cdef class OptimizationTestCase:
    cdef:
        list x, y, prizes, penalties, word_boundaries, word_prizes
        list result_indices
        double alpha, result_score
        int n, m, num_words
    def __init__(self, x=None, y=None,
                 prizes=None, penalties=None,
                 word_boundaries=None, word_prizes=None, alpha=None,
                 result_score=None, result_indices=None):
        self.x = x
        self.y = y
        self.prizes = prizes
        self.penalties = penalties
        self.word_boundaries = word_boundaries
        self.word_prizes = word_prizes
        self.alpha = alpha
        self.n = len(x)
        self.m = len(y)
        self.num_words = len(word_boundaries)
        self.result_score = result_score
        self.result_indices = result_indices

    def check_assertions(self):
        assert len(self.prizes) == self.n
        assert len(self.penalties) == self.m
        assert len(self.word_prizes) == self.num_words
        assert self.word_boundaries[-1] == len(self.x) - 1
        assert self.word_boundaries == sorted(self.word_boundaries)

    def run_test(self):
        cdef:
            int_array *x = make_int_array(self.n)
            int_array *y = make_int_array(self.m)
            double_array *prizes = make_double_array(self.n)
            double_array *penalties = make_double_array(self.m)
            unsigned int *word_boundaries
            double *word_prizes
            opt_results *output = make_opt_results(self.m)

        word_boundaries = <unsigned int*> \
                        PyMem_Malloc(self.num_words*sizeof(unsigned int))
        word_prizes = <double *> PyMem_Malloc(self.num_words*sizeof(double))
        for i in range(self.n):
            x.array[i] = self.x[i]
            prizes.array[i] = self.prizes[i]
        for i in range(self.m):
            y.array[i] = self.y[i]
            penalties.array[i] = self.penalties[i]
        for i in range(self.num_words):
            word_boundaries[i] = self.word_boundaries[i]
            word_prizes[i] = self.word_prizes[i]

        optimize(x, y, prizes, penalties, word_boundaries,
                 word_prizes, self.alpha, output)
        score = output.score
        indices = output.indices
        chars_matched = output.chars_matched
        ind = []
        for i in range(chars_matched):
            ind.append(indices[i])
        free_opt_results(output)
        free_int_array(x)
        free_int_array(y)
        free_double_array(prizes)
        free_double_array(penalties)
        PyMem_Free(word_boundaries)
        PyMem_Free(word_prizes)
        assert score == self.result_score
        assert ind == self.result_indices



