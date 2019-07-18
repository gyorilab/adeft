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


# def longform_score(shortform, candidates, cutoff=0.4)
#     encoded_shortform, encoded_candidates = encode(shortform, candidates):
