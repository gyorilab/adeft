def encode(shortform, candidates):
    n = len(shortform)
    char_map = {}
    encoded_shortform = [None]*n
    encoded_candidates = []
    char_indices = []
    used_tokens = []
    
    i = j = 0
    for i in range(n):
        if shortform[i] not in char_map:
            char_map[shortform[i]] = j
            j += 1
        encoded_shortform[n-1-i] = char_map[shortform[i]]
    for index, candidate in enumerate(candidates[::-1]):
        if set(shortform) & set(candidate):
            m = len(candidate)
            coded = []
            indices = []
            for j in range(m):
                c = candidate[m-j-1]
                if c in char_map:
                    coded.append(char_map[c])
                    indices.append(m-j-1)
            encoded_candidates.append([coded])
            char_indices.append(indices)
            used_tokens.append(index)
            
    return encoded_shortform, encoded_candidates, char_indices, used_tokens


def evens_speedup(n):
    m = n-1
    P = list(range(n))
    Pinv = list(range(n))
    D = [-1]*n
    T = [-1]*n
    inversions = 0

    yield P.copy(), inversions
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
        yield P.copy(), inversions
