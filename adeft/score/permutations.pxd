cdef struct permuter:
    int n
    int m
    int inversions
    int *P
    int *Pinv
    int *D
    int *T


cdef permuter *make_permuter(int n)
cdef void free_permuter(permuter *perms)
cdef void update_permuter(permuter *perms)
