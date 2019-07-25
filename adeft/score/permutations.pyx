from cython cimport boundscheck, wraparound
from cpython.mem cimport PyMem_Malloc, PyMem_Free


cdef permuter *make_permuter(int n):
    cdef permuter *perms
    perms = <permuter *> PyMem_Malloc(sizeof(permuter))

    perms.n = n
    perms.m = n - 1
    perms.P = <int *> PyMem_Malloc(n * sizeof(int))
    perms.Pinv = <int *> PyMem_Malloc(n * sizeof(int))
    perms.D = <int *> PyMem_Malloc(n * sizeof(int))
    perms.T = <int *> PyMem_Malloc(n * sizeof(int))

    cdef int i = 0
    for i in range(n):
        perms.P[i] = i
        perms.Pinv[i] = i
        perms.D[i] = perms.T[i] = -1
    return perms


cdef void free_permuter(permuter *perms):
    PyMem_Free(perms.P)
    PyMem_Free(perms.Pinv)
    PyMem_Free(perms.D)
    PyMem_Free(perms.T)
    PyMem_Free(perms)
    return


@boundscheck(False)
@wraparound(False)
cdef void update_permuter(permuter *perms):
    cdef int X, Y, Z, W
    X = perms.Pinv[perms.m]
    Y = X + perms.D[perms.m]
    Z = perms.P[Y]
    perms.P[Y] = perms.m
    perms.P[X] = Z
    perms.Pinv[Z] = X
    perms.Pinv[perms.m] = Y
    if perms.D[perms.m] < 0:
        perms.inversions += 1
    else:
        perms.inversions -= 1
    W = Y + perms.D[perms.m]
    if W == -1 or W == perms.n or perms.P[W] > perms.m:
        perms.D[perms.m] = -perms.D[perms.m]
        if perms.m == perms.n - 1:
            if perms.T[perms.n-1] < 0:
                perms.m = perms.n - 2
                if -perms.T[perms.n - 1] != perms.n - 1:
                    perms.T[perms.n - 2] = perms.T[perms.n - 1]
            else:
                perms.m = perms.T[perms.n - 1] - 1
        else:
            perms.T[perms.n - 1] = -(perms.m + 2)
            if perms.T[perms.m] > 0:
                perms.T[perms.m + 1] = perms.T[perms.m]
            else:
                perms.T[perms.m + 1] = perms.m
                if -perms.T[perms.m] != perms.m:
                    perms.T[perms.m-1] = perms.T[perms.m]
            perms.m = perms.n - 1
    else:
        if perms.m != perms.n - 1:
            perms.T[perms.n - 1] = -perms.m - 1
            perms.m = perms.n - 1
        
     
     
         
     
        
