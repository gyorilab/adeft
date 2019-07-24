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
