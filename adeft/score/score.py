import numpy as np


def encode(shortform, candidates):
    n = len(shortform)
    char_map = {}
    encoded_shortform = np.empty(n)
    encoded_candidates = []
    
    i = j = 0
    for i in range(n):
        if shortform[i] not in char_map:
            char_map[shortform[i]] = j
            j += 1
        encoded_shortform[n-1-i] = char_map[shortform[i]]
    for candidate in candidates[::-1]:
        if set(shortform) & set(candidate):
            encoded_candidates.append([char_map[c] for c in candidate[::-1]
                                      if c in char_map])
    return encoded_shortform, encoded_candidates


def evens_speedup(n):
    output = []
    m = n-1
    P = list(range(n)) + [n+1]
    Pinv = list(range(n))
    D = [-1]*n
    T = [0]*n
    T[-1] = -1

    i = 0
    while m != 0 and i < 10:
        i += 1
        print('T', T, 'm', m)
        output.append(P[:-1])
        X = Pinv[m]
        Y = X + D[m]
        Z = P[Y]
        P[Y] = m
        P[X] = Z
        Pinv[Z] = X
        Pinv[m] = Y
        if P[Y + D[m]] > m:
            D[m] = -D[m]
            if m == n - 1:
                if T[n-1] < 0:
                    m = n - 2
                    if -T[n-1] == n - 2:
                        continue
                    else:
                        T[n-2] = T[n-1]
                else:
                    m = T[n-1]
            else:
                T[n-1] = -(m+1)
                if T[m] > 0:
                    T[m+1] = T[m]
                    m = n-1
                    continue
        else:
            if m == n-1:
                continue
            else:
                T[n-1] = -m
                m = n-1
                continue
    return output
                    
        
        
    


# def longform_score(shortform, candidates, cutoff=0.4)
#     encoded_shortform, encoded_candidates = encode(shortform, candidates):
