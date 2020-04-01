import logging


from libc.math cimport pow as cpow
cdef extern from 'limits.h':
    int INT_MAX
from cython cimport boundscheck, wraparound
from cpython.mem cimport PyMem_Malloc, PyMem_Free

from adeft.nlp import stopwords_min

include 'permutations.pyx'

logger = logging.getLogger(__name__)


cdef struct int_array:
    int *array
    int length


cdef struct double_array:
    double *array
    int length


cdef struct opt_results:
    double score
    double *char_prizes


cdef struct candidates_array:
    int_array **array
    int_array **indices
    double *word_prizes
    int *cum_lengths
    double W
    int length


cdef struct opt_input:
    int_array *x
    int_array *indices
    double_array *word_prizes
    double W


cdef struct opt_params:
    double alpha, beta, gamma, lambda_


cdef struct opt_shortform:
    int_array *y
    double_array *penalties


cdef struct perm_out:
    double score


cdef opt_results *make_opt_results(int len_y):
    """Create opt_results data-structure

    Contains the optimal score for optimization problem as well as prizes
    or penalties associated to each character in the shortform
    """
    cdef opt_results *results
    results = <opt_results *> PyMem_Malloc(sizeof(opt_results))
    results.char_prizes = <double *> PyMem_Malloc(len_y * sizeof(double))
    return results


cdef void free_opt_results(opt_results *results):
    """Destroy opt_results data-structure"""
    PyMem_Free(results.char_prizes)
    PyMem_Free(results)
    return


cdef int_array *make_int_array(int length):
    """Create int_array data-structure

    An array of integers along with its length
    """
    cdef int_array *output
    output = <int_array *> PyMem_Malloc(sizeof(int_array))
    output.array = <int *> PyMem_Malloc(length * sizeof(int))
    output.length = length
    return output


cdef void free_int_array(int_array *x):
    """Destroy int_array data-structure"""
    PyMem_Free(x.array)
    PyMem_Free(x)
    return


cdef double_array *make_double_array(int length):
    """Create double_array data-structure

    An array of doubles along with its length
    """
    cdef double_array *output
    output = <double_array *> PyMem_Malloc(sizeof(int_array))
    output.array = <double *> PyMem_Malloc(length * sizeof(double))
    output.length = length
    return output


cdef void free_double_array(double_array *x):
    """Destroy double_array data-structure"""
    PyMem_Free(x.array)
    PyMem_Free(x)
    return


cdef candidates_array *make_candidates_array(list encoded_tokens,
                                             list word_prizes,
                                             float W):
    """Create candidates_array data-structure

    This is the core data-structure that contains all information needed
    to set up the core optimization problem. The list of encoded candidates
    is stored as an array of arrays, as is the list of lists of indices.
    Word prizes and the W_array described in the DocString for
    AdeftLongformScorer.process_candidates are also stored. The number of
    characters overlapping with the shortform in each candidate is also
    stored.
    """
    cdef:
        int i, j, num_candidates, m1, m2, n, cum_length
        candidates_array *candidates
    n = len(encoded_tokens)
    candidates = <candidates_array *> PyMem_Malloc(sizeof(candidates_array))
    candidates.array = <int_array **> PyMem_Malloc(n * sizeof(int_array*))
    candidates.indices = <int_array **> \
        PyMem_Malloc(n * sizeof(double_array*))
    candidates.word_prizes = <double *> PyMem_Malloc(n * sizeof(double))
    candidates.cum_lengths = <int *> PyMem_Malloc(n * sizeof(int))
    candidates.W = W
    candidates.length = n
    cum_length = 0
    for i in range(n):
        m1 = len(encoded_tokens[i])
        m2 = len(encoded_tokens[n-i-1])
        candidates.array[i] = make_int_array(m1)
        candidates.indices[i] = make_int_array(m1)
        cum_length += m2
        candidates.cum_lengths[i] = cum_length
        candidates.word_prizes[i] = word_prizes[i]
        for j in range(m1):
            candidates.array[i].array[j] = encoded_tokens[i][j][0]
            candidates.indices[i].array[j] = encoded_tokens[i][j][1]
    return candidates


cdef void free_candidates_array(candidates_array *candidates):
    """"Destroy candidates array data-structure"""
    cdef:
        int i, j
    for i in range(candidates.length):
        free_int_array(candidates.array[i])
        free_int_array(candidates.indices[i])
    PyMem_Free(candidates.word_prizes)
    PyMem_Free(candidates.array)
    PyMem_Free(candidates.indices)
    PyMem_Free(candidates.cum_lengths)
    PyMem_Free(candidates)


cdef opt_input *make_opt_input(int n, int num_words):
    """Create opt_input data-structure

    Stores information in longform candidate needed to set up
    optimization problem
    """
    cdef opt_input *input_
    input_ = <opt_input *> PyMem_Malloc(sizeof(opt_input))
    input_.x = make_int_array(n)
    input_.indices = make_int_array(n)
    input_.word_prizes = make_double_array(num_words)
    return input_


cdef void free_opt_input(opt_input *input_):
    """Destroy opt_input data-structure"""
    free_int_array(input_.x)
    PyMem_Free(input_.word_prizes)
    PyMem_Free(input_)


cdef opt_params *make_opt_params(double alpha, double beta,
                                 double gamma, double lambda_):
    """Create opt_params data-structure

    Stores scoring parameters
    """
    cdef opt_params *params
    params = <opt_params *> PyMem_Malloc(sizeof(opt_params))
    params.alpha = alpha
    params.beta = beta
    params.gamma = gamma
    params.lambda_ = lambda_
    return params


cdef void free_opt_params(opt_params *params):
    """Destroy opt_params data-structure"""
    PyMem_Free(params)


cdef opt_shortform *make_opt_shortform(int m):
    """Allocate opt_shortform data-structure

    Stores information about shortform needed to set up optimization problem
    """
    cdef opt_shortform *shortform
    shortform = <opt_shortform *> PyMem_Malloc(sizeof(opt_shortform))
    shortform.y = make_int_array(m)
    shortform.penalties = make_double_array(m)
    return shortform


cdef opt_shortform *create_shortform(list encoded_shortform,
                                     list penalties):
    """Initialize opt_shortform data-structure
    """
    cdef:
        opt_shortform *shortform
        int i = 0
        int m = len(encoded_shortform)
    shortform = make_opt_shortform(m)
    for i in range(m):
        shortform.y.array[i] = encoded_shortform[i]
        shortform.penalties.array[i] = penalties[i]
    return shortform


cdef void free_opt_shortform(opt_shortform *shortform):
    """Destroy opt_shortform data-structure"""
    free_int_array(shortform.y)
    free_double_array(shortform.penalties)
    PyMem_Free(shortform)


def score(encoded_tokens, encoded_shortform, word_prizes, W, penalties,
          max_inversions, max_perm_length, alpha, beta, gamma, lambda_, rho):
    cdef:
        candidates_array *candidates
        opt_shortform *shortform
        opt_params *params
        opt_results *result
    candidates = make_candidates_array(encoded_tokens, word_prizes, W)
    shortform = create_shortform(encoded_shortform, penalties)
    params = make_opt_params(alpha, beta, gamma, lambda_)
    result = make_opt_results(len(encoded_shortform))
    opt_search(candidates, shortform, params, rho, max_inversions,
               max_perm_length, result)
    score = result.score
    char_prizes = [result.char_prizes[i]
                   for i in range(len(encoded_shortform))]
    free_candidates_array(candidates)
    free_opt_params(params)
    free_opt_shortform(shortform)
    free_opt_results(result)
    return score, char_prizes


cdef void opt_search(candidates_array *candidates,
                     opt_shortform *shortform,
                     opt_params *params,
                     float rho,
                     int max_inversions,
                     int max_perm_length,
                     opt_results *output):
    """Calculates score for all permutations of tokens in candidate longform

    A multiplicate penalty rho is applied for each inversion in the permutation.
    So if a permutation P of the tokens with k inversions has score S based on
    the core optimization problem, the associated score for these tokens is
    S*rho**k. The scoring algorithm calculates an upper bound for the score
    that can be made by including the next token from right to left. Based on
    this upper bound, the maximum number of inversions in a permutation P that
    can still lead to an improvement in score can be calculated. If this number
    is zero, does not permute at all. If it is one, only checks permutations
    with one inversion. If it is greater than one, runs through all
    permutations, but only solves the core optimization problem for
    permutations with fewer than or equal to max_inversions inversions.
    """
    cdef:
        int input_size, i, j, temp
        double current_score, inv_penalty
        opt_input *current
        opt_results *results
        permuter *perms
    n = candidates.length
    perms = make_permuter(n)
    results = make_opt_results(shortform.y.length)
    total_length = candidates.cum_lengths[n - 1]
    if shortform.y.length > total_length + 1:
        input_size = total_length + shortform.y.length
    else:
        input_size = 2*total_length + 1
    current = make_opt_input(input_size, n)
    stitch(candidates, perms.P, shortform.y.length, current)
    optimize(current, shortform, params, results)
    output.score = results.score
    for i in range(shortform.y.length):
        output.char_prizes[i] = results.char_prizes[i]
    if max_inversions > 1 and n <= max_perm_length:
        while perms.m != 0:
            update_permuter(perms)
            if perms.inversions > max_inversions:
                continue
            stitch(candidates, perms.P, shortform.y.length, current)
            optimize(current, shortform, params, results)
            inv_penalty = cpow(rho, perms.inversions)
            current_score = results.score * inv_penalty
            if current_score > output.score:
                output.score = current_score
                for j in range(shortform.y.length):
                    output.char_prizes[j] = results.char_prizes[j]
    elif max_inversions == 1:
        for i in range(n-2):
            temp = perms.P[i]
            perms.P[i] = perms.P[i+1]
            perms.P[i+1] = temp
            stitch(candidates, perms.P, shortform.y.length, current)
            optimize(current, shortform, params, results)
            inv_penalty = cpow(rho, perms.inversions)
            current_score = results.score * inv_penalty
            if current_score > output.score:
                output.score = current_score
                for j in range(shortform.y.length):
                    output.char_prizes[j] = results.char_prizes[j]
            temp = perms.P[i]
            perms.P[i] = perms.P[i+1]
            perms.P[i+1] = temp
    free_permuter(perms)
    free_opt_results(results)
    free_opt_input(current)


@boundscheck(False)
@wraparound(False)
cdef void stitch(candidates_array *candidates, int *permutation,
                 int len_shortform, opt_input *result):
    """Stitch together information in candidates array into opt_input

    Forms opt_input for a permutation of the tokens in a candidate
    """
    cdef int i, j, k, h, current_length, n, m, p
    m = len_shortform
    n = candidates.length
    j = 0
    # stitched output begins with wildcard represented by -1
    while j < m - 1:
        result.x.array[j] = -1
        result.indices.array[j] = -1
        j += 1
    for i in range(n):
        p = permutation[i]
        current_length = candidates.array[p].length
        for k in range(current_length):
            result.x.array[j] = \
                candidates.array[p].array[k]
            result.indices.array[j] = \
                candidates.indices[p].array[k]
            # insert wildcard after each element from input
            for h in range(m-1):
                j += 1
                result.x.array[j] = -1
                result.indices.array[j] = -1
            j += 1
        if j < result.x.length:
            result.x.array[j] = -2
            result.indices.array[j] = -1
            result.word_prizes.array[i] = candidates.word_prizes[p]
            j += 1
    result.W = candidates.W


def optimize_alignment(woven_encoded_tokens, woven_indices, encoded_shortform,
                       word_prizes, W, penalties, alpha, beta, gamma, lambda_):
    cdef:
        int i
        double score
        list char_prizes
        opt_input *input_
        opt_shortform *shortform
        opt_params *params
        opt_results *result

    input_ = make_opt_input(len(woven_encoded_tokens), 1)
    result = make_opt_results(len(encoded_shortform))
    params = make_opt_params(alpha, beta, gamma, lambda_)
    shortform = make_opt_shortform(len(encoded_shortform))
    for i in range(len(encoded_shortform)):
        shortform.y.array[i] = encoded_shortform[i]
        shortform.penalties.array[i] = penalties[i]
    for i in range(len(woven_encoded_tokens)):
        input_.x.array[i] = woven_encoded_tokens[i]
        input_.indices.array[i] = woven_indices[i]
    for i in range(len(word_prizes)):
        input_.word_prizes.array[i] = word_prizes[i]
    input_.W = W
    optimize(input_, shortform, params, result)
    score = result.score
    char_prizes = [result.char_prizes[i]
                   for i in range(len(encoded_shortform))]
    free_opt_input(input_)
    free_opt_params(params)
    free_opt_shortform(shortform)
    free_opt_results(result)
    return score, char_prizes


@boundscheck(False)
@wraparound(False)
cdef void optimize(opt_input *input_, opt_shortform *shortform,
                   opt_params *params, opt_results *output):
    """Subsequence match optimization algorithm for longform scoring

    Uses a dynamic programming algorithm to find optimal instance of
    y as a subsequence in x where elements of x each have a corresponding
    prize. Wildcard characters are allowed in x that match any element of y
    and penalties may be given for when an element of y matches a wildcard
    instead of a regular element of x.

    Paramters
    ---------
    input_ : opt_input *
    shortform : opt_shortform *
    params : opt_params *
    results : opt_results *
        opt_results structure to where output is to be placed
    """
    cdef:
        int n = input_.x.length
        int m = shortform.y.length
        double possibility1, possibility2, word_score, char_score, c, w
        double first_capture, prize
        int i, j, k, current_fci

    # Dynamic initialization of score_lookup array and traceback pointer
    # array
    cdef:
        double **score_lookup = (<double **>
                                 PyMem_Malloc((n+1) * sizeof(double *)))
        double **char_scores = (<double **>
                                PyMem_Malloc((n+1) * sizeof(double *)))
        double **char_prizes = (<double **>
                                PyMem_Malloc((n+1) * sizeof(double *)))
        double **word_scores = (<double **>
                                PyMem_Malloc((n+1) * sizeof(double *)))
        int **word_use = <int **> PyMem_Malloc((n+1) * sizeof(int *))
        int **first_capture_index = <int **> \
            PyMem_Malloc((n+1) * sizeof(int *))
        int **pointers = <int **> PyMem_Malloc(n * sizeof(int *))
    for i in range(n+1):
        score_lookup[i] = <double *> PyMem_Malloc((m+1) * sizeof(double))
        char_scores[i] = <double *> PyMem_Malloc((m+1) * sizeof(double))
        char_prizes[i] = <double *> PyMem_Malloc((m+1) * sizeof(double))
        word_scores[i] = <double *> PyMem_Malloc((m+1) * sizeof(double))
        word_use[i] = <int *> PyMem_Malloc((m+1) * sizeof(int))
        first_capture_index[i] = <int *> PyMem_Malloc((m+1) * sizeof(int))
        if i != n:
            pointers[i] = <int *> PyMem_Malloc(m * sizeof(int))
    # Initialize lookup array
    score_lookup[0][0] = 0
    char_scores[0][0] = 0
    word_scores[0][0] = 0
    word_use[0][0] = 0
    first_capture_index[0][0] = -1
    for j in range(1, m+1):
        score_lookup[0][j] = -1e20
        char_scores[0][j] = 0
        word_scores[0][j] = 0
        word_use[0][j] = 0
        first_capture_index[0][j] = -1
    for i in range(1, n+1):
        for j in range(0, m+1):
            score_lookup[i][j] = 0
            char_scores[i][j] = 0
            word_scores[i][j] = 0
            word_use[i][j] = 0
            first_capture_index[i][j] = -1
    # Main loop
    k = 0
    first_capture = 0.0
    for i in range(1, n+1):
        for j in range(1, m+1):
            # Case where element of x in current position matches
            # element of y in current position. Algorithm considers
            # either accepting or rejecting this match
            if input_.x.array[i-1] == shortform.y.array[j-1]:
                if word_use[i-1][j-1] == 0:
                    w = input_.word_prizes.array[k]
                    current_fci = input_.indices.array[i-1]
                else:
                    current_fci = first_capture_index[i-1][j-1]
                    w = 0
                first_capture = cpow(params.alpha, current_fci)
                possibility1 = score_lookup[i-1][j]
                # Calculate score if match is accepted
                char_score = char_scores[i-1][j-1]
                prize = first_capture
                if input_.indices.array[i-1] > current_fci:
                    prize *= cpow(params.beta,
                                  input_.indices.array[i-1] -
                                  current_fci - 1)
                if word_use[i-1][j-1] > 1:
                    prize /= cpow(params.gamma, word_use[i-1][j-1] - 1)
                char_score += prize
                if char_score < 0.0:
                    c = 0.0
                else:
                    c = char_score
                word_score = word_scores[i-1][j-1] + w
                possibility2 = (cpow(c/m, params.lambda_) *
                                cpow(word_score/input_.W, (1-params.lambda_)))
                if score_lookup[i-1][j-1] > -1e19 and \
                   possibility2 > possibility1:
                    score_lookup[i][j] = possibility2
                    char_scores[i][j] = char_score
                    word_scores[i][j] = word_score
                    char_prizes[i][j] = prize
                    word_use[i][j] = word_use[i-1][j-1] + 1
                    first_capture_index[i][j] = current_fci
                    pointers[i-1][j-1] = 1
                else:
                    score_lookup[i][j] = possibility1
                    char_scores[i][j] = char_scores[i-1][j]
                    word_scores[i][j] = word_scores[i-1][j]
                    word_use[i][j] = word_use[i-1][j]
                    first_capture_index[i][j] = \
                        first_capture_index[i-1][j]
                    pointers[i-1][j-1] = 0
            # Case where element of x in current position is a wildcard.
            # May either accept or reject this match
            elif input_.x.array[i-1] == -1:
                possibility1 = score_lookup[i-1][j]
                # Score if wildcard match is accepted, skipping current
                # char in shortform
                char_score = char_scores[i-1][j-1]
                char_score -= shortform.penalties.array[j-1]
                # Take min with zero to ensure char_score doesn't become
                # negative
                if char_score < 0.0:
                    c = 0
                else:
                    c = char_score
                word_score = word_scores[i-1][j-1]
                possibility2 = (cpow(c/m, params.lambda_) *
                                cpow(word_score/input_.W, 1-params.lambda_))
                if score_lookup[i-1][j-1] > -1e19 and \
                   possibility2 > possibility1:
                    score_lookup[i][j] = possibility2
                    char_scores[i][j] = char_score
                    word_scores[i][j] = word_score
                    char_prizes[i][j] = -shortform.penalties.array[j-1]
                    word_use[i][j] = word_use[i-1][j-1]
                    first_capture_index[i][j] = \
                        first_capture_index[i-1][j-1]
                    pointers[i-1][j-1] = 1
                else:
                    score_lookup[i][j] = possibility1
                    char_scores[i][j] = char_scores[i-1][j]
                    word_scores[i][j] = word_scores[i-1][j]
                    word_use[i][j] = word_use[i-1][j]
                    first_capture_index[i][j] = \
                        first_capture_index[i-1][j]
                    pointers[i-1][j-1] = 0
            else:
                score_lookup[i][j] = score_lookup[i-1][j]
                char_scores[i][j] = char_scores[i-1][j]
                word_scores[i][j] = word_scores[i-1][j]
                word_use[i][j] = word_use[i-1][j]
                first_capture_index[i][j] = \
                    first_capture_index[i-1][j]
                pointers[i-1][j-1] = 0
            if input_.x.array[i-1] == -2:
                word_use[i][j] = 0
                first_capture_index[i][j] = -1
        if input_.x.array[i-1] == -2:
            k += 1
    # Optimal score is in bottom right corner of lookup array
    score = score_lookup[n][m]
    # Free the memory used by the lookup array

    # Set score in output
    output.score = score
    # Trace backwards through pointer array to discover which elements of x
    # were matched and add the score associated to each character in the
    # shortform into the char scores array.
    i, j, k = n, m, 0
    while j > 0:
        if pointers[i - 1][j - 1]:
            output.char_prizes[m-k-1] = char_prizes[i][j]
            i -= 1
            j -= 1
            k += 1
        else:
            i -= 1
    # Free lookup arrays used by algorithm
    for i in range(n+1):
        PyMem_Free(score_lookup[i])
        PyMem_Free(word_use[i])
        PyMem_Free(char_scores[i])
        PyMem_Free(word_scores[i])
        PyMem_Free(char_prizes[i])
        if i < n:
            PyMem_Free(pointers[i])
    PyMem_Free(score_lookup)
    PyMem_Free(char_scores)
    PyMem_Free(word_scores)
    PyMem_Free(char_prizes)
    PyMem_Free(word_use)
    PyMem_Free(pointers)


cdef class StitchTestCase:
    """Test construction of candidates array and stitching"""
    cdef:
        double W
        int shortform_length
        list encoded_tokens, word_prizes,
        list permutation, result_x, result_word_prizes, result_indices
        list result_word_boundaries
    def __init__(self, encoded_tokens=None,
                 word_prizes=None,
                 permutation=None, W=None,
                 result_x=None, result_indices=None, result_word_prizes=None,
                 shortform_length=None):
        self.encoded_tokens = encoded_tokens
        self.word_prizes = word_prizes
        self.W = W
        self.permutation = permutation
        self.result_x = result_x
        self.result_indices = result_indices
        self.result_word_prizes = result_word_prizes
        self.shortform_length = shortform_length

    def run_test(self):
        cdef:
            opt_input *input_
            int_array *perm
            candidates_array *candidates
        candidates = make_candidates_array(self.encoded_tokens,
                                           self.word_prizes,
                                           self.W)
        n = len(self.permutation)
        perm = make_int_array(n)
        for i in range(n):
            perm.array[i] = self.permutation[i]
        total_length = candidates.cum_lengths[n - 1]
        m = self.shortform_length
        input_size = m*total_length + m + n - 1
        input_ = make_opt_input(input_size, n)
        stitch(candidates, perm.array, self.shortform_length, input_)
        x, ind, wp, wb = [], [], [], []
        length = input_.x.length
        for i in range(length):
            x.append(input_.x.array[i])
            ind.append(input_.indices.array[i])
        for j in range(input_.word_prizes.length):
            wp.append(input_.word_prizes.array[j])
        free_candidates_array(candidates)
        free_opt_input(input_)
        free_int_array(perm)
        assert x == self.result_x
        assert ind == self.result_indices
        assert wp == self.result_word_prizes


cdef class PermSearchTestCase:
    cdef:
        list shortform, encoded_tokens, penalties, word_prizes
        double alpha, beta, gamma, lambda_, rho, result_score, W
        int len_perm
    def __init__(self, shortform=None, encoded_tokens=None,
                 penalties=None, word_prizes=None, W=None,
                 alpha=None, beta=None, gamma=None, lambda_=None, rho=None,
                 len_perm=None,
                 result_score=None):
        self.shortform = shortform
        self.encoded_tokens = encoded_tokens
        self.penalties = penalties
        self.word_prizes = word_prizes
        self.W = W
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lambda_ = lambda_
        self.rho = rho
        self.result_score = result_score

    def run_test(self):
        cdef:
            candidates_array *candidates
            opt_shortform *shortform
            opt_params *params
            opt_results *results
        candidates = make_candidates_array(self.encoded_tokens,
                                           self.word_prizes,
                                           self.W)
        shortform = create_shortform(self.shortform, self.penalties)
        params = make_opt_params(self.alpha, self.beta, self.gamma,
                                 self.lambda_)
        results = make_opt_results(len(self.shortform))
        opt_search(candidates, shortform, params, self.rho, 100, 8, results)
        assert abs(results.score - self.result_score) < 1e-7


cdef class OptimizationTestCase:
    cdef:
        list x, y, indices, penalties, word_prizes
        list result_char_scores
        double alpha, beta, gamma, lambda_, C, W, result_score
        int n, m, num_words
    def __init__(self, x=None, y=None, indices=None, penalties=None,
                 word_prizes=None, alpha=None,
                 beta=None, gamma=None, lambda_=None, W=None,
                 result_score=None, result_char_scores=None):
        self.x = x
        self.y = y
        self.indices = indices
        self.penalties = penalties
        self.word_prizes = word_prizes
        self.num_words = len(self.word_prizes)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lambda_ = lambda_
        self.W = W
        self.n = len(x)
        self.m = len(y)
        self.result_score = result_score
        self.result_char_scores = result_char_scores

    def check_assertions(self):
        assert len(self.penalties) == self.m

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
