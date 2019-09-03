import numpy as np
from libc.math cimport pow as cpow
from cython cimport boundscheck, wraparound
from cpython.mem cimport PyMem_Malloc, PyMem_Free

from adeft.score.permutations cimport permuter, make_permuter, \
    free_permuter, update_permuter


cdef opt_results *make_opt_results(int len_y):
    cdef opt_results *results
    results = <opt_results *> PyMem_Malloc(sizeof(opt_results))
    results.char_scores = <double *> PyMem_Malloc(len_y * sizeof(double))
    return results


cdef void free_opt_results(opt_results *results):
    PyMem_Free(results.char_scores)
    PyMem_Free(results)
    return


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


@boundscheck(False)
@wraparound(False)
cdef candidates_array *make_candidates_array(list encoded_candidates,
                                             list prizes,
                                             list word_prizes,
                                             list W):
    cdef:
        int i, j, num_candidates, m1,m2, n, cum_length
        candidates_array *candidates
    n = len(encoded_candidates)
    candidates = <candidates_array *> PyMem_Malloc(sizeof(candidates_array))
    candidates.array = <int_array **> PyMem_Malloc(n * sizeof(int_array*))
    candidates.prizes = <double_array **> \
        PyMem_Malloc(n * sizeof(double_array*))
    candidates.word_prizes = <double *> PyMem_Malloc(n * sizeof(double))
    candidates.cum_lengths = <int *> PyMem_Malloc(n * sizeof(int))
    candidates.W_array = <double *> PyMem_Malloc(n * sizeof(double))
    candidates.length = n
    cum_length = 0
    for i in range(n):
        m1 = len(encoded_candidates[i])
        m2 = len(encoded_candidates[n-i-1])
        candidates.array[i] = make_int_array(m1)
        candidates.prizes[i] = make_double_array(m1)
        cum_length += m2
        candidates.cum_lengths[i] = cum_length
        candidates.word_prizes[i] = word_prizes[i]
        candidates.W_array[i] = W[i]
        for j in range(m1):
            candidates.array[i].array[j] = encoded_candidates[i][j]
            candidates.prizes[i].array[j] = prizes[i][j]
    return candidates


cdef void free_candidates_array(candidates_array *candidates):
    cdef:
        int i, j
    for i in range(candidates.length):
         free_int_array(candidates.array[i])
         free_double_array(candidates.prizes[i])
    PyMem_Free(candidates.word_prizes)
    PyMem_Free(candidates.W_array)
    PyMem_Free(candidates.prizes)
    PyMem_Free(candidates.array)
    PyMem_Free(candidates.cum_lengths)
    PyMem_Free(candidates)


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


cdef opt_params *make_opt_params(double beta, double rho):
    cdef opt_params *params
    params = <opt_params *> PyMem_Malloc(sizeof(opt_params))
    params.beta = beta
    params.rho = rho
    return params


cdef void free_opt_params(opt_params *params):
    PyMem_Free(params)


cdef opt_shortform *make_opt_shortform(list encoded_shortform,
                                       list penalties):
    cdef:
        opt_shortform *shortform
        int i = 0
        int m = len(encoded_shortform)
    shortform = <opt_shortform *> PyMem_Malloc(sizeof(opt_shortform))
    shortform.y = make_int_array(m)
    shortform.penalties = make_double_array(m)
    for i in range(m):
        shortform.y.array[i] = encoded_shortform[i]
        shortform.penalties.array[i] = penalties[i]
    return shortform


cdef void free_opt_shortform(opt_shortform *shortform):
    free_int_array(shortform.y)
    free_double_array(shortform.penalties)
    PyMem_Free(shortform)


cdef class LongformScorer:
    def __init__(self, shortform, penalties=None, alpha=0.5, beta=0.45,
                 inv_penalty=0.9, rho=0.6, word_scores=None):
        pass


@wraparound(False)
cdef double perm_search(candidates_array *candidates,
                        opt_shortform *shortform,
                        opt_params *params,
                        float inv_penalty,
                        int n):
    cdef:
        double best, current_score
        permuter *perms
        opt_input *current
        opt_results *results
    results = make_opt_results(shortform.y.length)
    total_length = candidates.cum_lengths[n - 1]
    current = make_opt_input(2*total_length + 1, n)
    perms = make_permuter(n)
    stitch(candidates, perms.P, n, current)
    optimize(current, shortform, params, results)
    best = results.score
    while perms.m != 0:
        update_permuter(perms)
        stitch(candidates, perms.P, n, current)
        optimize(current, shortform, params, results)
        current_score = results.score * cpow(inv_penalty,
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
    result.W = candidates.W_array[len_perm - 1]
    return result


@boundscheck(False)
@wraparound(False)
cdef void *optimize(opt_input *input_, opt_shortform *shortform,
                    opt_params *params, opt_results *output):
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
 : C array of double
        Must the the same length as y. Penalty lost if the corresponding
        element of y matches a wildcard.

    results : struct opt_results
        opt_results structure to where output is to be placed
    """
    cdef:
        unsigned int n = input_.x.length
        unsigned int m = shortform.y.length
        double possibility1, possibility2, c, w, delta
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
            if input_.x.array[i-1] == shortform.y.array[j-1]:
                if word_use[i-1][j-1] == 0:
                    w = input_.word_prizes.array[k]
                else:
                    w = 0
                possibility1 = score_lookup[i-1][j]
                # Calculate score if match is accepted
                previous_score = score_lookup[i-1][j-1]
                c = input_.prizes.array[i-1]/cpow(params.beta,
                                                  word_use[i-1][j-1])           
                delta = params.rho * c/m + \
                    (1 - params.rho) * w/input_.W                  
                possibility2 = previous_score + delta
                if possibility2 > possibility1:
                    score_lookup[ i][j] = possibility2
                    word_use[i][j] = word_use[i-1][j-1] + 1
                    pointers[i-1][j-1] = 1
                else:
                    score_lookup[i][j] = possibility1
                    word_use[i][j] = word_use[i-1][j]
                    pointers[i-1][j-1] = 0
            # Case where element of x in current position is a wildcard.
            # May either accept or reject this match
            elif input_.x.array[i-1] == -1:
                possibility1 = score_lookup[i-1][j]
                # Score if wildcard match is accepted, skipping current
                # char in shortform
                possibility2 = score_lookup[i-1][j-1] - \
                    shortform.penalties.array[j-1]/m
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
            if i == input_.word_boundaries[k] + 1:
                word_use[i][j] = 0
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
            i -= 1
            j -= 1
            if input_.x.array[i] == -1:
                output.char_scores[m-k-1] = -shortform.penalties.array[k+1]
            else:
                output.char_scores[m-k-1] = \
                    input_.prizes.array[i]/cpow(params.beta, word_use[i][j])
            k += 1
        else:
            i -= 1
    # Free lookup arrays used by algorithm
    for i in range(n+1):
        PyMem_Free(score_lookup[i])
        PyMem_Free(word_use[i])
        if i < n:
            PyMem_Free(pointers[i])
    PyMem_Free(score_lookup)
    PyMem_Free(word_use)
    PyMem_Free(pointers)
    return output
