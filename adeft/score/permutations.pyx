"""This extension implements the Steinhaus-Johnson-Trotter permutation
generation Algorithm with Even's speedup. The permuter works by successively
swapping elements of an array, with state variables kept so that at all steps
it is easy to calculate which elements should be swapped. These are C functions
that can be cimported within Cython files but cannot be imported in Python files
"""


from cython cimport boundscheck, wraparound
from cpython.mem cimport PyMem_Malloc, PyMem_Free


cdef permuter *make_permuter(int n):
    """Initialize a permuter. 

    Parameters
    ----------
    n : C int
        Length of permutations to generate. Permutes an array of the first
        n integers starting with 0

    Returns
    -------
    permuter : perms
        A permuter has the following attributes
    
        m : int
            Tracks which element is active and will be switched at the next
            update This is not the index, but the value of the element that
            is to be switched

        P : array of int, shape (n, )
            Current permutation state. An array of length n

        Pinv : array of int, shape (n, )
            Inverse of the permutation P. Used for finding the location of the
            element m in P when it is time to switch

        D : array of int,  shape (n, )
            Current directions for each element. When an element is switched it
            will move in its current direction. Directions are updated at each
            step

        T : array of int, shape (n, )
             State vector that is used to calculate which element is to be
             switched during the next step.
    """
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
    perms.inversions = 0
    return perms


cdef void free_permuter(permuter *perms):
    """Free the memory used by a permuter"""
    PyMem_Free(perms.P)
    PyMem_Free(perms.Pinv)
    PyMem_Free(perms.D)
    PyMem_Free(perms.T)
    PyMem_Free(perms)
    return


@boundscheck(False)
@wraparound(False)
cdef void update_permuter(permuter *perms):
    """Make one swap to generate the next permutation using SJT algorithm

    See Algorithmic Combinatorics (1973) by Shimon Evens
    """
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
        # m has reached its final step in its current direction
        # value will now change
        perms.D[perms.m] = -perms.D[perms.m]
        if perms.m == perms.n - 1:
            # Case 2: m has reached its final step and equals n - 1
            if perms.T[perms.n-1] < 0:
                perms.m = perms.n - 2
                if -perms.T[perms.n - 1] != perms.n - 1:
                    perms.T[perms.n - 2] = perms.T[perms.n - 1]
            else:
                perms.m = perms.T[perms.n - 1] - 1
        else:
            # Case 4: m has reached its final step and does not equal n - 1
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
            # Case 3: m has not reached its less step and m != n - 1
            perms.T[perms.n - 1] = -perms.m - 1
            perms.m = perms.n - 1
        # Case 1: m = n - 1 and it has not reached the last step in its
        # present direction. No changes are needed.
