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
    if x.shape[0] != prizes.shape[0] or y.shape[0] != penalties.shape[0]:
        raise ValueError
    cdef:
        unsigned int n = x.shape[0]
        unsigned int m = y.shape[0]
        double possibility1, possibility
        unsigned int i, j, k
        results output

    if n != prizes.shape[0] or m != penalties.shape[0]:
        raise ValueError

    cdef:
        double **score_lookup = (<double **>
                                 PyMem_Malloc((n+1) * sizeof(double *)))
        int **pointers = <int **> PyMem_Malloc(n * sizeof(int *))

    for i in range(n+1):
        score_lookup[i] = <double *> PyMem_Malloc((m+1) * sizeof(double))
        if i != n:
            pointers[i] = <int *> PyMem_Malloc(m * sizeof(int))

    with boundscheck(False), wraparound(False):
        score_lookup[0][0] = 0
        for j in range(1, m+1):
            score_lookup[0][j] = np.double('-inf')
        for i in range(1, n+1):
            for j in range(0, m+1):
                score_lookup[i][j] = 0
        for i in range(1, n+1):
            for j in range(1, m+1):
                if x[i-1] == y[j-1]:
                    possibility1 = score_lookup[i-1][j]
                    possibility2 = score_lookup[i-1][j-1] + prizes[i-1]
                    if possibility2 >= possibility1:
                        score_lookup[i][j] = possibility2
                        pointers[i-1][j-1] = 1
                    else:
                        score_lookup[i][j] = possibility1
                        pointers[i-1][j-1] = 0
                elif x[i-1] == -1:
                    possibility1 = score_lookup[i-1][j]
                    possibility2 = score_lookup[i-1][j-1] - penalties[j-1]
                    if possibility2 >= possibility1:
                        score_lookup[i][j] = possibility2
                        pointers[i-1][j-1] = 1
                    else:
                        score_lookup[i][j] = possibility1
                        pointers[i-1][j-1] = 0
                else:
                    score_lookup[i][j] = score_lookup[i-1][j]
                    pointers[i-1][j-1] = 0
    score = score_lookup[n][m]
    PyMem_Free(score_lookup)

    output.score = score
    output.indices = <int *> PyMem_Malloc(m * sizeof(int))

    i, j, k = n-1, m-1, 0
    while i > 0:
        if pointers[i][j]:
            i -= 1
            j -= 1
            output.indices[k] = i
            k += 1
        else:
            i -= 1
    PyMem_Free(pointers)
    output.chars_matched = k
    return output
