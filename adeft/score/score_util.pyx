import numpy as np
from libc.math cimport pow as cpow
from cython cimport boundscheck, wraparound
from cpython.mem cimport PyMem_Malloc, PyMem_Free

from adeft.score.permutations cimport permuter, make_permuter, \
    free_permuter, update_permuter


def check_optimize():
    cdef:
        int x_0[5]
        int y_0[2]
        double prizes_0[5]
        double penalties_0[2]
        int_array x, y
        double_array prizes, penalties
        opt_results *output

    penalties.length = 2

    a = np.array([-1, 1, -1, 0, -1], dtype=np.int)
    b = np.array([1, 0], dtype=np.int)
    c = np.array([0.0, 1.0, 0.0, 1.0, 0.0],
                 dtype=np.double)
    d = np.array([0.2, 0.4], dtype=np.double)

    for i in range(5):
        x_0[i] = a[i]
        prizes_0[i] = c[i]

    for i in range(2):
        y_0[i] = b[i]
        penalties_0[i] = d[i]

    x.array = x_0
    x.length = 5
    y.array = y_0
    y.length = 2
    prizes.array = prizes_0
    prizes.length = 5
    penalties.array = penalties_0

    output = make_opt_results(2)
    optimize(&x, &y, &prizes, &penalties, output)
    score = output.score
    indices = output.indices
    chars_matched = output.chars_matched
    ind = np.empty(chars_matched, dtype=np.int)
    for i in range(chars_matched):
        ind[i] = indices[i]
    free_opt_results(output)
    return score, ind


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
    int *cum_lengths
    int length
    double inv_penalty


cdef candidates_array *make_candidates_array(list encoded_shortform,
                                             list encoded_candidates,
                                             list prizes,
                                             list penalties,
                                             double inv_penalty):
    cdef:
        int i, j, num_candidates, m, n, cum_length, k
        candidates_array *candidates
    n = len(encoded_candidates)
    k = len(encoded_shortform)
    candidates = <candidates_array *> PyMem_Malloc(sizeof(candidates_array))
    candidates.array = <int_array **> PyMem_Malloc(n * sizeof(int_array*))
    candidates.prizes = <double_array **> \
        PyMem_Malloc(n * sizeof(double_array*))
    candidates.penalties = make_double_array(k)
    candidates.cum_lengths = <int *> PyMem_Malloc(n * sizeof(int))
    candidates.length = n
    candidates.inv_penalty = inv_penalty
    candidates.y = make_int_array(k)
    for i in range(k):
        candidates.penalties.array[i] = penalties[i]
        candidates.y.array[i] = encoded_shortform[i]
    cum_length = 0
    for i in range(n):
        m = len(encoded_candidates[i])
        candidates.array[i] = make_int_array(m)
        candidates.prizes[i] = make_double_array(m)
        cum_length += m
        candidates.cum_lengths[i] = cum_length
        for j in range(m):
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
    PyMem_Free(candidates.prizes)
    PyMem_Free(candidates.array)
    PyMem_Free(candidates.cum_lengths)
    PyMem_Free(candidates)


cdef struct opt_input:
    int_array *x
    double_array *prizes


cdef opt_input *make_opt_input(int n):
    cdef opt_input *input_
    input_ = <opt_input *> PyMem_Malloc(sizeof(opt_input))
    input_.x = make_int_array(n)
    input_.prizes = make_double_array(n)
    return input_


cdef void free_opt_input(opt_input *input_):
    free_int_array(input_.x)
    free_double_array(input_.prizes)
    PyMem_Free(input_)
    

cdef double perm_search(candidates_array *candidates, int n):
    cdef:
        double best, current_score
        permuter *perms
        opt_input *current
        opt_results *results

    results = make_opt_results(candidates.y.length)
    total_length = candidates.cum_lengths[n - 1]
    current = make_opt_input(2*total_length + 1)
    perms = make_permuter(n)
    stitch(candidates, perms.P, n, current)
    optimize(current.x, candidates.y, current.prizes,
             candidates.penalties, results)
    best = results.score
    while perms.m != 0:
        update_permuter(perms)
        stitch(candidates, perms.P, n, current)
        optimize(current.x, candidates.y, current.prizes,
                 candidates.penalties, results)
        current_score = results.score * cpow(candidates.inv_penalty,
                                             perms.inversions)
        if current_score > best:
            best = current_score
    free_permuter(perms)
    free_opt_results(results)
    free_opt_input(current)
    return best


cdef void *stitch(candidates_array *candidates, int *permutation,
                  int len_perm, opt_input *result):
    cdef int i, j, k, total_length, current_length
    
    

    # stitched output begins with wildcard represented by -1
    result.x.array[0] = -1
    result.prizes.array[0] = 0
    j = 1
    for i in range(len_perm):
        current_length = candidates.array[permutation[i]].length
        for k in range(current_length):
            result.x.array[j] = \
                candidates.array[permutation[i]].array[k]
            # insert wildcard after each element from input
            result.x.array[j+1] = -1
            result.prizes.array[j] = \
                candidates.prizes[permutation[i]].array[k]
            result.prizes.array[j+1] = 0
            j += 2
    return result

    
cdef void *optimize(int_array *x, int_array *y,
                           double_array *prizes, double_array *penalties,
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
        double possibility1, possibility
        unsigned int i, j, k

    # Dynamic initialization of score_lookup array and traceback pointer
    # array
    cdef:
        double **score_lookup = (<double **>
                                 PyMem_Malloc((n+1) * sizeof(double *)))
        int **pointers = <int **> PyMem_Malloc(n * sizeof(int *))

    for i in range(n+1):
        score_lookup[i] = <double *> PyMem_Malloc((m+1) * sizeof(double))
        if i != n:
            pointers[i] = <int *> PyMem_Malloc(m * sizeof(int))

    # Hold on to your butts
    with boundscheck(False), wraparound(False):
        score_lookup[0][0] = 0
        for j in range(1, m+1):
            score_lookup[0][j] = -1e20
        for i in range(1, n+1):
            for j in range(0, m+1):
                score_lookup[i][j] = 0
        for i in range(1, n+1):
            for j in range(1, m+1):
                # Case where element of x in current position matches
                # element of y in current position. Algorithm considers
                # either accepting or rejecting this match
                if x.array[i-1] == y.array[j-1]:
                    possibility1 = score_lookup[i-1][j]
                    possibility2 = score_lookup[i-1][j-1] + prizes.array[i-1]
                    if possibility2 >= possibility1:
                        score_lookup[i][j] = possibility2
                        pointers[i-1][j-1] = 1
                    else:
                        score_lookup[i][j] = possibility1
                        pointers[i-1][j-1] = 0
                # Case where element of x in current position is a wildcard.
                # May either accept or reject this match
                elif x.array[i-1] == -1:
                    possibility1 = score_lookup[i-1][j]
                    possibility2 = score_lookup[i-1][j-1] - penalties.array[j-1]
                    if possibility2 >= possibility1:
                        score_lookup[i][j] = possibility2
                        pointers[i-1][j-1] = 1
                    else:
                        score_lookup[i][j] = possibility1
                        pointers[i-1][j-1] = 0
                # No match is possible. There is only one option to fill
                # current entry of dynamic programming lookup array.
                else:
                    score_lookup[i][j] = score_lookup[i-1][j]
                    pointers[i-1][j-1] = 0
    # Optimal score is in bottom right corner of lookup array
    score = score_lookup[n][m]
    # Free the memory used by the lookup array
    for i in range(n+1):
        PyMem_Free(score_lookup[i])
    PyMem_Free(score_lookup)

    # Set score in output
    output.score = score

    # Trace backwards through pointer array to discover which elements of x
    # were matched and add the corresponding indices to the index array in
    # reverse order
    i, j, k = m-1, n-1, 0
    while i > 0:
        if pointers[i][j]:
            i -= 1
            j -= 1
            output.indices[k] = i
            k += 1
        else:
            j -= 1
    # Free pointer array
    for i in range(n):
        PyMem_Free(pointers[i])
    PyMem_Free(pointers)
    # Set the number of chars in y that were matched
    output.chars_matched = k
    return output


cdef struct perm_out:
    double score

    
def check_convert():
    cdef:
        list sf = [1, 0]
        list ca = [[0], [1]]
        list prizes = [[1.0], [1.0]]
        list penalties = [0.2, 0.4]
        int perm[2]
        opt_input *input_
        candidates_array *candidates
        list x = []
        list p = []

    perm[0], perm[1] = 1, 0

    candidates = make_candidates_array(sf, ca,  prizes, penalties, 0.9)
    input_ = make_opt_input(candidates.y.length)
    stitch(candidates, perm, 2, input_)

    length = input_.x.length
    
    for i in range(length):
        x.append(input_.x.array[i])
        p.append(input_.prizes.array[i])
    free_candidates_array(candidates)
    free_opt_input(input_)
    return (x, p)


def check_perm_search():
    cdef:
        list sf = [1, 0]
        list ca = [[1], [0, 1], [1, 1, 0]]
        list prizes = [[1.0], [0.5, 1.0], [0.25, 0.5, 1.0]]
        list penalties = [0.2, 0.4]
        candidates_array *candidates

    candidates = make_candidates_array(sf, ca,  prizes, penalties, 0.9)
    score = perm_search(candidates, 3)
    free_candidates_array(candidates)
    return score



