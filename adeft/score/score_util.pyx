import numpy as np
from cython cimport boundscheck, wraparound
from cpython.mem cimport PyMem_Malloc, PyMem_Free


def check_optimize():
    a = np.array([-1, 0, -1, 1, -1, 0, -1, 1, -1, 0], dtype=np.int)
    b = np.array([0, 1], dtype=np.int)
    c = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 0.8, 0.0, 0.6, 0.0, 0.7],
                 dtype=np.double)
    d = np.array([0.4, 0.2], dtype=np.double)
    output = optimize(a, b, c, d)
    score = output.score
    indices = output.indices
    chars_matched = output.chars_matched
    ind = np.empty(chars_matched, dtype=np.int)
    for i in range(chars_matched):
        ind[i] = indices[i]
    return score, ind


cdef struct results:
    double score
    int *indices
    int chars_matched


cdef results optimize(long[:] x, long[:] y,
                    double[:] prizes, double[:] penalties):
    """Subsequence match optimization algorithm for longform scoring

    Uses a dynamic programming algorithm to find optimal instance of
    y as a subsequence in x where elements of x each have a corresponding
    prize. Wildcard characters are allowed in x that match any element of y
    and penalties may be given for when an element of y matches a wildcard
    instead of a regular element of x.

    Paramters
    ---------
    x : TypedMemoryView of long
        Contains nonnegative long ints for supersequence in which we seek
        a subsequence match. May also contain the value -1 which corresponds to a
        wildcard that matches any nonnegative int.

    y : TypedMemoryView of long
        Sequence we seek an optimal subsequence match of in x

    prizes : TypedMemoryView of double
        Must be the same length as x. Prize gained for matching an element
        of y to the corresponding element of x

    penalties : TypedMemoryView of double
        Must the the same length as y. Penalty lost if the corresponding
        element of y matches a wildcard.

    Returns
    -------
    output : struct results
        Contains three entries. The score of optimal match, a c array of
        indices matched in x in reverse order, and the number of characters
        in y that were matched in x. 
    """
    # Check once that input shapes are valid.
    if x.shape[0] != prizes.shape[0] or y.shape[0] != penalties.shape[0]:
        raise ValueError
    cdef:
        unsigned int n = x.shape[0]
        unsigned int m = y.shape[0]
        double possibility1, possibility
        unsigned int i, j, k
        results output

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
            score_lookup[0][j] = np.double('-inf')
        for i in range(1, n+1):
            for j in range(0, m+1):
                score_lookup[i][j] = 0
        for i in range(1, n+1):
            for j in range(1, m+1):
                # Case where element of x in current position matches
                # element of y in current position. Algorithm considers
                # either accepting or rejecting this match
                if x[i-1] == y[j-1]:
                    possibility1 = score_lookup[i-1][j]
                    possibility2 = score_lookup[i-1][j-1] + prizes[i-1]
                    if possibility2 >= possibility1:
                        score_lookup[i][j] = possibility2
                        pointers[i-1][j-1] = 1
                    else:
                        score_lookup[i][j] = possibility1
                        pointers[i-1][j-1] = 0
                # Case where element of x in current position is a wildcard.
                # May either accept or reject this match
                elif x[i-1] == -1:
                    possibility1 = score_lookup[i-1][j]
                    possibility2 = score_lookup[i-1][j-1] - penalties[j-1]
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
    # Initialize indices array in output. Max possible length is the length
    # of y
    output.indices = <int *> PyMem_Malloc(m * sizeof(int))

    # Trace backwards through pointer array to discover which elements of x
    # were matched and add the corresponding indices to the index array in
    # reverse order
    i, j, k = n-1, m-1, 0
    while i > 0:
        if pointers[i][j]:
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

cdef struct int_array:
    int *array
    int length

cdef struct candidates_array:
    int_array *array
    int *cum_lengths
    int length
    
cdef candidates_array convert_input(list encoded_candidates):
    cdef:
        int i, j, num_candidates, m, n, cum_length
        candidates_array candidates
    n = len(encoded_candidates)
    candidates.array = <int_array *> PyMem_Malloc(n * sizeof(int_array))
    candidates.cum_lengths = <int *> PyMem_Malloc(n * sizeof(int))
    candidates.length = n
    num_candidates = len(encoded_candidates)
    cum_length = 0
    for i in range(num_candidates):
        m = len(encoded_candidates[i])
        candidates.array[i].array = <int *> PyMem_Malloc(m * sizeof(int))
        candidates.array[i].length = m
        cum_length += m
        candidates.cum_lengths[i] = cum_length
        for j in range(m):
            candidates.array[i].array[j] = encoded_candidates[i][j]
    return candidates


cdef free_candidates_array(candidates_array *candidates):
    cdef:
        int i, j
    for i in range(candidates.length):
        PyMem_Free(candidates.array[i].array)
    PyMem_Free(candidates.array)


cdef int_array *stitch(candidates_array *candidates,
                       int *permutation,
                       int len_perm):
    cdef:
        int i, j, k, total_length, current_length
        int_array output
        int *temp
    total_length = candidates.cum_lengths[len_perm - 1]
    output.array = <int *> PyMem_Malloc((2*total_length + 1) * sizeof(int))
    output.length = 2*total_length + 1
    # stitched output begins with wildcard represented by -1
    output.array[0] = -1
    j = 1
    for i in range(len_perm):
        temp = candidates.array[permutation[i]].array
        current_length = candidates.array[permutation[i]].length
        for k in range(current_length):
            output.array[j] = temp[k]
            # insert wildcard after each element from input
            output.array[j+1] = -1
            j += 2
    return &output
        

def check_convert():
    cdef:
        list b = [[0, 1, 0], [0, 1], [1]]
        int perm[3]
        int_array *output

    perm[0], perm[1], perm[2] = 2, 0, 1

    candidates = convert_input(b)
    output = stitch(&candidates, perm, 3)

    length = output.length
    for i in range(length):
        print(output.array[i])

    free_candidates_array(&candidates)
    PyMem_Free(output.array)



