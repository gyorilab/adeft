import numpy as np
from libc.math cimport pow as cpow
from cython cimport boundscheck, wraparound
from cpython.mem cimport PyMem_Malloc, PyMem_Free


def check_optimize():
    cdef:
        int x_0[5]
        int y_0[2]
        double prizes_0[5]
        double penalties_0[2]
        int_array x, y
        double_array prizes, penalties

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

    output = optimize(&x, &y, &prizes, &penalties)
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


cdef results *optimize(int_array *x, int_array *y,
                       double_array *prizes, double_array *penalties):
    """Subsequence match optimization algorithm for longform scoring

    Uses a dynamic programming algorithm to find optimal instance of
    y as a subsequence in x where elements of x each have a corresponding
    prize. Wildcard characters are allowed in x that match any element of y
    and penalties may be given for when an element of y matches a wildcard
    instead of a regular element of x.

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

    Returns
    -------
    output : pointer to struct results
        Contains three entries. The score of optimal match, a c array of
        indices matched in x in reverse order, and the number of characters
        in y that were matched in x. 
    """
    cdef:
        unsigned int n = x.length
        unsigned int m = y.length
        double possibility1, possibility
        unsigned int i, j, k
        results *output

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

    output = <results *> PyMem_Malloc(sizeof(results))
    # Set score in output
    output.score = score
    # Initialize indices array in output. Max possible length is the length
    # of y
    output.indices = <int *> PyMem_Malloc(m * sizeof(int))

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

cdef struct int_array:
    int *array
    int length

cdef struct double_array:
    double *array
    int length

cdef struct candidates_array:
    int_array *array
    int_array *y
    double_array *prizes
    double_array *penalties
    int *cum_lengths
    int length
    double inv_penalty


cdef candidates_array *convert_input(list encoded_shortform,
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
    candidates.array = <int_array *> PyMem_Malloc(n * sizeof(int_array))
    candidates.prizes = <double_array *> PyMem_Malloc(n * sizeof(double_array))
    candidates.penalties = <double_array *> PyMem_Malloc(sizeof(double_array))
    candidates.cum_lengths = <int *> PyMem_Malloc(n * sizeof(int))
    candidates.length = n
    candidates.inv_penalty = inv_penalty

    candidates.y = <int_array *> PyMem_Malloc(sizeof(int_array))
    candidates.y.array = <int *> PyMem_Malloc(k * sizeof(int))
    candidates.y.length = k

    candidates.penalties.length = k
    candidates.penalties.array = <double *> \
        PyMem_Malloc(k * sizeof(double))
    for i in range(k):
        candidates.penalties.array[i] = penalties[i]
        candidates.y.array[i] = encoded_shortform[i]
    cum_length = 0
    for i in range(n):
        m = len(encoded_candidates[i])
        candidates.array[i].array = <int *> PyMem_Malloc(m * sizeof(int))
        candidates.array[i].length = m
        candidates.prizes[i].array = <double *> \
            PyMem_Malloc(m * sizeof(double))
        candidates.prizes[i].length = m
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
         PyMem_Free(candidates.array[i].array)
         PyMem_Free(candidates.prizes[i].array)
    PyMem_Free(candidates.prizes)
    PyMem_Free(candidates.array)
    PyMem_Free(candidates.cum_lengths)
    PyMem_Free(candidates.y.array)
    PyMem_Free(candidates.y)
    PyMem_Free(candidates.penalties.array)
    PyMem_Free(candidates.penalties)
    PyMem_Free(candidates)


cdef struct opt_input:
    int_array *x
    double_array *prizes
    

cdef opt_input *stitch(candidates_array *candidates,
                       int *permutation,
                       int len_perm):
    cdef:
        int i, j, k, total_length, current_length
        opt_input *output
        int *temp
    output = <opt_input *> PyMem_Malloc(sizeof(opt_input))
    total_length = candidates.cum_lengths[len_perm - 1]
    output.x = <int_array *> PyMem_Malloc(sizeof(int_array))
    output.prizes = <double_array *> PyMem_Malloc(sizeof(double_array))
    output.x.array = \
        <int *> PyMem_Malloc((2*total_length + 1) * sizeof(int))
    output.x.length = 2*total_length + 1
    # stitched output begins with wildcard represented by -1
    output.x.array[0] = -1
    output.prizes.array = \
        <double *> PyMem_Malloc((2*total_length + 1) * sizeof(double))
    output.prizes.length = 2*total_length + 1
    output.prizes.array[0] = 0
    j = 1
    for i in range(len_perm):
        current_length = candidates.array[permutation[i]].length
        for k in range(current_length):
            output.x.array[j] = \
                candidates.array[permutation[i]].array[k]
            # insert wildcard after each element from input
            output.x.array[j+1] = -1
            output.prizes.array[j] = \
                candidates.prizes[permutation[i]].array[k]
            output.prizes.array[j+1] = 0
            j += 2
    return output


cdef struct perm_out:
    double score

    
cdef double perm_search(candidates_array *candidates, int n):
    cdef:
        int m = n - 1
        int inversions = 0
        int *P
        int *Pinv
        int *D
        int *T
        int X, Y, Z, W
        double best, current_score
        opt_input *current
        results *opt_results

    P = <int *> PyMem_Malloc(n * sizeof(int))
    Pinv = <int *> PyMem_Malloc(n * sizeof(int))
    D = <int *> PyMem_Malloc(n * sizeof(int))
    T = <int *> PyMem_Malloc(n * sizeof(int))

    cdef int i = 0
    for i in range(n):
        P[i] = i
        Pinv[i] = i
        D[i] = -1
        T[i] = -1
    current = stitch(candidates, P, n)
    opt_results = optimize(current.x, candidates.y, current.prizes,
                           candidates.penalties)
    best = opt_results.score
    while m != 0:
        X = Pinv[m]
        Y = X + D[m]
        Z = P[Y]
        P[Y] = m
        P[X] = Z
        Pinv[Z] = X
        Pinv[m] = Y
        if D[m] < 0:
            inversions += 1
        else:
            inversions -= 1
        W = Y + D[m]
        if W == -1 or W == n or P[W] > m:
            D[m] = -D[m]
            if m == n - 1:
                if T[n-1] < 0:
                    m = n - 2
                    if -T[n-1] != n - 1:
                        T[n-2] = T[n-1]
                else:
                    m = T[n-1] - 1
            else:
                T[n-1] = -(m+2)
                if T[m] > 0:
                    T[m+1] = T[m]
                else:
                    T[m+1] = m
                    if -T[m] != m:
                        T[m-1] = T[m]
                m = n - 1
        else:
            if m != n-1:
                T[n-1] = -m - 1
                m = n-1
        current = stitch(candidates, P, n)
        opt_results = optimize(current.x, candidates.y, current.prizes,
                               candidates.penalties)
        current_score = opt_results.score * cpow(candidates.inv_penalty,
                                                 inversions)
        if current_score > best:
            best = current_score
    PyMem_Free(P)
    PyMem_Free(Pinv)
    PyMem_Free(D)
    PyMem_Free(T)
    return best


# def longform_score(encoded_shortform, encoded_candidates, prizes):
#     cdef:
#         candidates_array *candidates

#     candidates = convert_input(encoded_candidates, prizes)
    
#     return



def check_convert():
    cdef:
        list sf = [1, 0]
        list ca = [[0], [1]]
        list prizes = [[1.0], [1.0]]
        list penalties = [0.2, 0.4]
        int perm[2]
        opt_input *output
        candidates_array *candidates
        list x = []
        list p = []

    perm[0], perm[1] = 1, 0

    candidates = convert_input(sf, ca,  prizes, penalties, 0.9)
    output = stitch(candidates, perm, 2)

    length = output.x.length
    
    for i in range(length):
        x.append(output.x.array[i])
        p.append(output.prizes.array[i])
    free_candidates_array(candidates)
    PyMem_Free(output.x.array)
    PyMem_Free(output.prizes.array)
    PyMem_Free(output)
    return (x, p)


def check_perm_search():
    cdef:
        list sf = [1, 0]
        list ca = [[1], [0, 1], [1, 1, 0]]
        list prizes = [[1.0], [0.5, 1.0], [0.25, 0.5, 1.0]]
        list penalties = [0.2, 0.4]
        candidates_array *candidates

    candidates = convert_input(sf, ca,  prizes, penalties, 0.9)
    score = perm_search(candidates, 3)
    free_candidates_array(candidates)
    return score


@boundscheck(False)
@wraparound(False)
def permutations(x):
    cdef:
        int n = x
        int m = n - 1
        int *P
        int *Pinv
        int *D
        int *T
        int X, Y, Z, W
        list out

    P = <int *> PyMem_Malloc(n * sizeof(int))
    Pinv = <int *> PyMem_Malloc(n * sizeof(int))
    D = <int *> PyMem_Malloc(n * sizeof(int))
    T = <int *> PyMem_Malloc(n * sizeof(int))

    cdef int i = 0
    for i in range(n):
        P[i] = i
        Pinv[i] = i
        D[i] = -1
        T[i] = -1

    out = [None]*n
    for i in range(n):
        out[i] = P[i]
    yield out
    while m != 0:
        X = Pinv[m]
        Y = X + D[m]
        Z = P[Y]
        P[Y] = m
        P[X] = Z
        Pinv[Z] = X
        Pinv[m] = Y
        W = Y + D[m]
        if W == -1 or W == n or P[W] > m:
            D[m] = -D[m]
            if m == n - 1:
                if T[n-1] < 0:
                    m = n - 2
                    if -T[n-1] != n - 1:
                        T[n-2] = T[n-1]
                else:
                    m = T[n-1] - 1
            else:
                T[n-1] = -(m+2)
                if T[m] > 0:
                    T[m+1] = T[m]
                else:
                    T[m+1] = m
                    if -T[m] != m:
                        T[m-1] = T[m]
                m = n - 1
        else:
            if m != n-1:
                T[n-1] = -m - 1
                m = n-1
        out = [None]*n
        for i in range(n):
            out[i] = P[i]
        yield out




