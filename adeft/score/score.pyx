from libc.math cimport pow as cpow
cdef extern from 'limits.h':
    int INT_MAX
from cython cimport boundscheck, wraparound
from cpython.mem cimport PyMem_Malloc, PyMem_Free

from adeft.nlp import stopwords
from adeft.score.permutations cimport permuter, make_permuter, \
    free_permuter, update_permuter


cdef class AdeftLongformScorer:
    """Scorer for longform expansions based on character matching

    Searches for shortform as a subsequence of the characters within
    longform candidates. Longform candidates are given by sequences of tokens
    preceding a defining pattern (DP). Prizes are given for characters matched
    within the longform and penalties are given for characters in the
    shortform that are not matched to any character in the longform. Prizes
    within tokens are context dependent, depending on the pattern of previous
    matches in the current token. The maximum character prize is 1.0. A 
    character score is calculated by taking the sum of prizes for all
    characters matched in the longform subtracted by the sum of penalties
    for all characters in the shortform that are not matched and then dividing
    by the length of the shortform. The character score is then the max of
    this number and zero.
    
    Character prizes are controlled by three parameters, alpha, beta, and
    gamma. Penalties for unmatched characters from the shortform are controlled
    by the parameters delta, and epsilon. More information in the description
    of parameters below.

    The algorithm considers candidate longforms one at a time, preceding
    from right to left from the defining pattern (DP). Tokens that do not
    contain any of the character from the shortform are ignored, since
    they cannot contain a match. Each token has an associated score. If a
    character is matched in a given token, we say that token has been
    captured. A token score is calculated as the sum of scores for all
    captured tokens divided by the sum of scores for all tokens. Tokens
    that do not contain characters in the shortform are still included
    when calculating the sum of all token scores. 

    The character scores and token scores described above each fall between
    0 and 1. The total score for a longform is given by a weighted geometric
    mean of these two values, with the weight being controlled by a user
    supplied parameter lambda_.

    Given a candidate, all permutations of its tokens are considered as
    potential matches to the shortform. This allows the algorithm to identify
    cases such as beta-2 adrenergic receptor (ADRB2). A multiplicative penalty,
    rho, is applied for each inversion of the permutation. A number of
    optimizations and heuristics are performed that allow the algorithm to
    efficiently despite the possibility of super-factorial complexity.

    Attributes
    ----------
    shortform : str
        Shortform for which longforms are sought
    penalties : list of double
        Penalties for characters in the shortform that are not matched.
        If None, then penalties are calculated based on the parameters
        gamma and delta.
        Default: None
    alpha : double
        Real value in [0, 1]
        Within a token, the initial character has prize 1.0. Prizes for
        succeeding characters decay exponentially at rate alpha.
        Default: 0.5
    beta : double
        Real value in (0, 1]
        Character prizes are increased if previous characters within a token
        have already been matched to a character in the shortform. Character
        prize is divided by beta**(num_previous_matches). For useful results,
        make sure beta >= alpha
        Default: 0.4
    gamma : double
        Real value in [0, 1]
        Penalty for not matching first character in shortform
        Default: 1.0
    delta : double
        Real value in [0, 1]
        Penalties for additional characters in shortform decay exponentially
        at rate delta
        Default: 0.9
    lambda_ : double
        Real value in [0, 1]
        Weighting for character based scoring vs token based scoring. Larger
        values of lambda_ correspond to more importance being given to
        character matching
        Default: 0.7
    rho : double
        Multiplicative penalty for number of inversions in permutation of
        tokens If trying a match with a permution P of the tokens in a
        candidate, multiply score by rho**inv where inv is the number
        of inversions in P
        Default: 0.9
    word_scores : dict
        Scores associated to each token. Higher scores correspond to a higher
        penalty for not being included in a match with the shortform. The
        scores for words not in the word_scores dictionary default to 1
    """
    cdef:
        public str shortform
        public list penalties
        public double alpha, beta, gamma, delta, epsilon, lambda_, rho
        public dict word_scores
        int len_shortform
        dict char_map
        opt_shortform *shortform_c
        opt_params *params_c

    def __init__(self, shortform, penalties=None, alpha=0.2, beta=0.85,
                 gamma=0.9, delta=1.0, epsilon=0.4, lambda_=0.6, rho=0.95,
                 word_scores=None):
        self.shortform = shortform.lower()
        self.len_shortform = len(shortform)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon
        self.rho = rho
        self.lambda_ = lambda_
        self.params_c = make_opt_params(alpha, beta, gamma, lambda_)
        # Encode shortform chars as integers and build map for encoding
        # longform candidates
        self.char_map = {}
        cdef list encoded_shortform = []
        cdef int i = 0, j = 0
        for i in range(self.len_shortform):
            if self.shortform[i] not in self.char_map:
                self.char_map[self.shortform[i]] = j
                j += 1
            encoded_shortform.append(self.char_map[self.shortform[i]])
        if penalties is not None:
            self.penalties = penalties
        else:
            self.penalties = [delta*epsilon**i
                              for i in range(self.len_shortform)]
        self.shortform_c = create_shortform(encoded_shortform,
                                            self.penalties)
        self.params_c = make_opt_params(alpha, beta, gamma, lambda_)
        if word_scores is None:
            self.word_scores = {word: 0.2 for word in stopwords}
        else:
            self.word_scores = word_scores

    cdef double get_word_score(self, str token):
        if token in self.word_scores:
            return self.word_scores[token]
        else:
            return 1.0

    cdef tuple process_candidates(self, list candidates):
        cdef:
            double word_score
            str token
            list encoded_candidates, indices, encoded_token
            list token_indices, word_prizes, W
            int n, i, j
        encoded_candidates = []
        indices = []
        word_prizes = []
        n = len(candidates)
        for i in range(n):
            encoded_token = []
            token_indices = []
            token = candidates[i].lower()
            m = len(token)
            for j in range(m):
                if token[j] in self.char_map:
                    encoded_token.append(self.char_map[token[j]])
                    token_indices.append(j)
            if encoded_token:
                encoded_candidates.append(encoded_token)
                indices.append(token_indices)
                word_score = self.get_word_score(token)
                word_prizes.append(word_score)
        if not encoded_candidates:
            return ([], [], [], [])
        W = [word_prizes[-1]]
        for i in range(1, len(word_prizes)):
            W.append(W[i-1] + word_prizes[-i])
        return (encoded_candidates, indices, word_prizes, W)

    cdef tuple get_score_results(self,
                                 list candidates,
                                 list scores,
                                 list W_array):
        """Produce useable output for longform scorer
        """
        n = len(candidates)
        i = j = 0
        shortform_chars = set(self.shortform)
        results = []
        best_score = -1.0
        current_score = 0.0
        current_candidates_list = []
        best_candidate = ''
        while i < n:
            current_candidates_list.append(candidates[n-i-1])
            current_candidate = ' '.join(current_candidates_list[::-1])
            if set(candidates[n-i-1]) & shortform_chars:
                current_score = scores[j]
                j += 1
            else:
                W = W_array[j-1] if j > 0 else 0.0
                w = self.get_word_score(candidates[n-i-1])
                current_score *= (W/(W + w))**(1 - self.lambda_)
            results += (current_candidate, current_score)
            if current_score > best_score:
                best_score = current_score
                best_candidate = current_candidate
            i += 1
        return (best_candidate, best_score, results)

    def score(self, candidates):
        cdef:
            int i, j, k, n, max_inversions
            double current_score, best_score, w, W, ub_char_scores
            double ub_word_scores
            double_array *best_char_scores
            double_array *previous_word_scores
            list scores, encoded_candidates, indices, word_prizes, W_array
            tuple out
            candidates_array *candidates_c
            opt_results *results
            opt_results *probe_results
        results = make_opt_results(self.len_shortform)
        probe_results = make_opt_results(self.len_shortform)
        encoded_candidates, indices, word_prizes, W_array = \
            self.process_candidates(candidates)
        if not encoded_candidates:
            return self.get_score_results(candidates, [0.0]*len(candidates),
                                          [0.0]*len(candidates))
        candidates_c = make_candidates_array(encoded_candidates,
                                             indices,
                                             word_prizes,
                                             W_array)
        n = candidates_c.length
        scores = [None]*n
        best_score = -1.0
        best_char_scores = make_double_array(self.len_shortform)
        for i in range(self.len_shortform):
            best_char_scores.array[i] = -1e20
        for i in range(1, n + 1):
            ub_char_scores = 0.0
            probe(candidates_c.array[n - i], candidates_c.indices[n - i],
                  self.shortform_c.y,
                  self.params_c.alpha, self.params_c.beta, self.params_c.gamma,
                  probe_results)
            for j in range(self.len_shortform):
                if probe_results.char_prizes[j] > best_char_scores.array[j]:
                    ub_char_scores += probe_results.char_prizes[j]
                elif best_char_scores.array[j] > 0:
                    ub_char_scores += best_char_scores.array[j]
            if i > 1:
                previous_word_scores = make_double_array(i)
                for k in range(i):
                    previous_word_scores.array[k] = \
                        candidates_c.word_prizes[n-k-1]
                ub_word_scores = opt_selection(previous_word_scores,
                                               self.len_shortform-1)
                free_double_array(previous_word_scores)
            else:
                ub_word_scores = 0.0
            ub_char_scores = ub_char_scores/self.len_shortform
            W = candidates_c.W_array[i-1]
            w = candidates_c.word_prizes[n-i]
            ub_word_scores = (w + ub_word_scores)/W
            upper_bound = (cpow(ub_char_scores, self.params_c.lambda_) *
                           cpow(ub_word_scores, 1-self.params_c.lambda_))
            if upper_bound < best_score:
                scores[i-1] = scores[i-2]*(W - w)/W
                continue
            if upper_bound * self.rho < best_score:
                opt_search(candidates_c, self.shortform_c,
                           self.params_c, self.rho,
                           i, 0, 0, results)
            else:
                max_inversions = get_max_inversions(best_score, upper_bound,
                                                    self.rho)
                opt_search(candidates_c, self.shortform_c,
                           self.params_c, self.rho,
                           i, 1, max_inversions, results)
            current_score = results.score
            scores[i-1] = current_score
            if current_score >= best_score:
                best_score = current_score
                for j in range(self.len_shortform):
                    best_char_scores.array[j] = results.char_prizes[j]
        out = self.get_score_results(candidates, scores,
                                     [candidates_c.W_array[i]
                                      for i in range(candidates_c.length)])
        free_double_array(best_char_scores)
        free_opt_results(probe_results)
        free_opt_results(results)
        free_candidates_array(candidates_c)
        return out


cdef inline int get_max_inversions(double best_score,
                                   double upper_bound,
                                   double rho):
    cdef int k = 0
    if best_score <= 0.0:
        return INT_MAX
    while best_score <= upper_bound * cpow(rho, k):
        k += 1
    return k - 1


cdef double opt_selection(double_array *word_prizes, int k):
    cdef:
        int max_index, i, j
        double max_value, temp, output
    if k >= word_prizes.length:
        output = 0.0
        for i in range(word_prizes.length):
            output += word_prizes.array[i]
        return output
    for i in range(k):
        max_index = i
        max_value = word_prizes.array[i]
        for j in range(i+i, word_prizes.length):
            if word_prizes.array[j] > max_value:
                max_index = j
                max_value = word_prizes.array[j]
                temp = word_prizes.array[max_index]
                word_prizes.array[max_index] = word_prizes.array[i]
                word_prizes.array[i] = temp
    output = 0.0
    for i in range(k):
        output += word_prizes.array[i]
    return output


cdef opt_results *make_opt_results(int len_y):
    cdef opt_results *results
    results = <opt_results *> PyMem_Malloc(sizeof(opt_results))
    results.char_prizes = <double *> PyMem_Malloc(len_y * sizeof(double))
    return results


cdef void free_opt_results(opt_results *results):
    PyMem_Free(results.char_prizes)
    PyMem_Free(results)
    return


cdef int_array *make_int_array(int length):
    cdef int_array *output
    output = <int_array *> PyMem_Malloc(sizeof(int_array))
    output.array = <int *> PyMem_Malloc(length * sizeof(int))
    output.length = length
    return output


cdef void free_int_array(int_array *x):
    PyMem_Free(x.array)
    PyMem_Free(x)
    return


cdef double_array *make_double_array(int length):
    cdef double_array *output
    output = <double_array *> PyMem_Malloc(sizeof(int_array))
    output.array = <double *> PyMem_Malloc(length * sizeof(double))
    output.length = length
    return output


cdef void free_double_array(double_array *x):
    PyMem_Free(x.array)
    PyMem_Free(x)
    return


cdef candidates_array *make_candidates_array(list encoded_candidates,
                                             list indices,
                                             list word_prizes,
                                             list W):
    cdef:
        int i, j, num_candidates, m1,m2, n, cum_length
        candidates_array *candidates
    n = len(encoded_candidates)
    candidates = <candidates_array *> PyMem_Malloc(sizeof(candidates_array))
    candidates.array = <int_array **> PyMem_Malloc(n * sizeof(int_array*))
    candidates.indices = <int_array **> \
        PyMem_Malloc(n * sizeof(double_array*))
    candidates.word_prizes = <double *> PyMem_Malloc(n * sizeof(double))
    candidates.cum_lengths = <int *> PyMem_Malloc(n * sizeof(int))
    candidates.W_array = <double *> PyMem_Malloc(n * sizeof(double))
    candidates.length = n
    cum_length = 0
    for i in range(n):
        m1 = len(encoded_candidates[i])
        m2 = len(encoded_candidates[n-i-1])
        candidates.array[i] = make_int_array(m1)
        candidates.indices[i] = make_int_array(m1)
        cum_length += m2
        candidates.cum_lengths[i] = cum_length
        candidates.word_prizes[i] = word_prizes[i]
        candidates.W_array[i] = W[i]
        for j in range(m1):
            candidates.array[i].array[j] = encoded_candidates[i][j]
            candidates.indices[i].array[j] = indices[i][j]
    return candidates


cdef void free_candidates_array(candidates_array *candidates):
    cdef:
        int i, j
    for i in range(candidates.length):
        free_int_array(candidates.array[i])
        free_int_array(candidates.indices[i])
    PyMem_Free(candidates.word_prizes)
    PyMem_Free(candidates.W_array)
    PyMem_Free(candidates.array)
    PyMem_Free(candidates.indices)
    PyMem_Free(candidates.cum_lengths)
    PyMem_Free(candidates)


cdef opt_input *make_opt_input(int n, int num_words):
    cdef opt_input *input_
    input_ = <opt_input *> PyMem_Malloc(sizeof(opt_input))
    input_.x = make_int_array(n)
    input_.indices = make_int_array(n)
    input_.word_boundaries = <unsigned int *> \
        PyMem_Malloc(num_words * sizeof(unsigned int))
    input_.word_prizes = make_double_array(num_words)
    return input_


cdef void free_opt_input(opt_input *input_):
    free_int_array(input_.x)
    PyMem_Free(input_.word_boundaries)
    PyMem_Free(input_.word_prizes)
    PyMem_Free(input_)


cdef opt_params *make_opt_params(double alpha, double beta,
                                 double gamma, double lambda_):
    cdef opt_params *params
    params = <opt_params *> PyMem_Malloc(sizeof(opt_params))
    params.alpha = alpha
    params.beta = beta
    params.gamma = gamma
    params.lambda_ = lambda_
    return params


cdef void free_opt_params(opt_params *params):
    PyMem_Free(params)


cdef opt_shortform *make_opt_shortform(int m):
    cdef opt_shortform *shortform
    shortform = <opt_shortform *> PyMem_Malloc(sizeof(opt_shortform))
    shortform.y = make_int_array(m)
    shortform.penalties = make_double_array(m)
    return shortform


cdef opt_shortform *create_shortform(list encoded_shortform,
                                     list penalties):
    cdef:
        opt_shortform *shortform
        int i = 0
        int m = len(encoded_shortform)
    shortform = make_opt_shortform(m)
    for i in range(m):
        shortform.y.array[i] = encoded_shortform[i]
        shortform.penalties.array[i] = penalties[i]
    return shortform


cdef void free_opt_shortform(opt_shortform *shortform):
    free_int_array(shortform.y)
    free_double_array(shortform.penalties)
    PyMem_Free(shortform)


cdef void opt_search(candidates_array *candidates,
                     opt_shortform *shortform,
                     opt_params *params,
                     float rho,
                     int n,
                     int permute,
                     int max_inversions,
                     opt_results *output):
    cdef:
        int input_size, i,
        double current_score, inv_penalty
        permuter *perms
        opt_input *current
        opt_results *results
    results = make_opt_results(shortform.y.length)
    total_length = candidates.cum_lengths[n - 1]
    if shortform.y.length > total_length + 1:
        input_size = total_length + shortform.y.length
    else:
        input_size = 2*total_length + 1
    current = make_opt_input(input_size, n)
    perms = make_permuter(n)
    stitch(candidates, perms.P, n, current)
    optimize(current, shortform, params, results)
    output.score = results.score
    for i in range(shortform.y.length):
        output.char_prizes[i] = results.char_prizes[i]
    if permute:
        while perms.m != 0:
            update_permuter(perms)
            if perms.inversions > max_inversions:
                continue
            stitch(candidates, perms.P, n, current)
            optimize(current, shortform, params, results)
            inv_penalty = cpow(rho, perms.inversions)
            current_score = results.score * inv_penalty
            if current_score > output.score:
                output.score = current_score
                for i in range(shortform.y.length):
                    output.char_prizes[i] = results.char_prizes[i]
    free_permuter(perms)
    free_opt_results(results)
    free_opt_input(current)


@boundscheck(False)
@wraparound(False)
cdef void probe(int_array *next_token, int_array *indices, int_array *y,
                double alpha, double beta, double gamma,
                opt_results *probe_results):
    cdef:
        int i, j, input_size
        opt_input *input_
        opt_params *params
        opt_shortform *shortform
    # First initalize the probe
    if y.length > next_token.length + 1:
        input_size = next_token.length + y.length
    else:
        input_size = 2*next_token.length + 1
    input_ = make_opt_input(input_size, 1)
    result = make_opt_results(y.length)
    params = make_opt_params(alpha, beta, gamma, 1.0)
    shortform = make_opt_shortform(y.length)
    for i in range(y.length):
        shortform.y.array[i] = y.array[i]
        shortform.penalties.array[i] = 0.0
    input_.x.array[0] = -1
    input_.indices.array[0] = -1
    i = 0
    j = 1
    for i in range(next_token.length):
        input_.x.array[j] = next_token.array[i]
        input_.indices.array[j] = indices.array[i]
        input_.x.array[j+1] = -1
        input_.indices.array[j+1] = -1
        j += 2
    while j < input_.x.length:
        input_.x.array[j] = -1
        input_.indices.array[j] = -1
        j += 1
    input_.word_prizes.array[0] = 0
    input_.word_boundaries[0] = j - 1
    input_.W = 1
    optimize(input_, shortform, params, probe_results)
    free_opt_input(input_)
    free_opt_params(params)
    free_opt_shortform(shortform)


@boundscheck(False)
@wraparound(False)
cdef void stitch(candidates_array *candidates, int *permutation,
                  int len_perm, opt_input *result):
    cdef int i, j, k, current_length, n, p
    n = candidates.length
    # stitched output begins with wildcard represented by -1
    result.x.array[0] = -1
    result.indices.array[0] = -1
    i = 0
    j = 1
    for i in range(len_perm):
        p = permutation[i]
        current_length = candidates.array[n-len_perm+p].length
        for k in range(current_length):
            result.x.array[j] = \
                candidates.array[n-len_perm+p].array[k]
            result.indices.array[j] = \
                candidates.indices[n-len_perm+p].array[k]
            # insert wildcard after each element from input
            result.x.array[j+1] = -1
            result.indices.array[j+1] = -1
            j += 2
        result.word_boundaries[i] = j - 1
        result.word_prizes.array[i] = candidates.word_prizes[n-len_perm+p]
    while j < result.x.length:
        result.x.array[j] = -1
        result.indices.array[j] = -1
        j += 1
    result.W = candidates.W_array[len_perm - 1]


@boundscheck(False)
@wraparound(False)
cdef void optimize(opt_input *input_, opt_shortform *shortform,
                   opt_params *params, opt_results *output):
    """Subsequence match optimization algorithm for longform scoring

    Uses a dynamic programming algorithm to find optimal instance of
    y as a subsequence in x where elements of x each have a corresponding
    prize. Wildcard characters are allowed in x that match any element of y
    and penalties may be given for when an element of y matches a wildcard
    instead of a regular element of x. Results are updated in place.

    Paramters
    ---------
    input_ : opt_input *
    shortform : opt_shortform *
    params : opt_params *
    results : opt_results *
        opt_results structure to where output is to be placed
    """
    cdef:
        unsigned int n = input_.x.length
        unsigned int m = shortform.y.length
        double possibility1, possibility2, word_score, char_score, c, w
        double first_capture, prize
        unsigned int i, j, k, first_capture_index

    # Dynamic initialization of score_lookup array and traceback pointer
    # array
    cdef:
        double **score_lookup = (<double **>
                                 PyMem_Malloc((n+1) * sizeof(double *)))
        double **char_scores = (<double **>
                                PyMem_Malloc((n+1) * sizeof(double *)))
        double **char_prizes = (<double **>
                                PyMem_Malloc((n+1) * sizeof(double *)))
        double **word_scores = (<double **>
                                PyMem_Malloc((n+1) * sizeof(double *)))
        int **word_use = <int **> PyMem_Malloc((n+1) * sizeof(int *))
        int **pointers = <int **> PyMem_Malloc(n * sizeof(int *))
    for i in range(n+1):
        score_lookup[i] = <double *> PyMem_Malloc((m+1) * sizeof(double))
        char_scores[i] = <double *> PyMem_Malloc((m+1) * sizeof(double))
        char_prizes[i] = <double *> PyMem_Malloc((m+1) * sizeof(double))
        word_scores[i] = <double *> PyMem_Malloc((m+1) * sizeof(double))
        word_use[i] = <int *> PyMem_Malloc((m+1) * sizeof(int))
        if i != n:
            pointers[i] = <int *> PyMem_Malloc(m * sizeof(int))
    # Initialize lookup array
    score_lookup[0][0] = 0
    char_scores[0][0] = 0
    word_scores[0][0] = 0
    word_use[0][0] = 0
    for j in range(1, m+1):
        score_lookup[0][j] = -1e20
        char_scores[0][j] = 0
        word_scores[0][j] = 0
        word_use[0][j] = 0
    for i in range(1, n+1):
        for j in range(0, m+1):
            score_lookup[i][j] = 0
            char_scores[i][j] = 0
            word_scores[i][j] = 0
            word_use[i][j] = 0
    # Main loop
    k = 0
    first_capture = 0
    first_capture_index = -1
    for i in range(1, n+1):
        for j in range(1, m+1):
            # Case where element of x in current position matches
            # element of y in current position. Algorithm considers
            # either accepting or rejecting this match
            if input_.x.array[i-1] == shortform.y.array[j-1]:
                if word_use[i-1][j-1] == 0:
                    w = input_.word_prizes.array[k]
                    first_capture_index = input_.indices.array[i-1]
                    first_capture = cpow(params.alpha, first_capture_index)
                else:
                    w = 0
                possibility1 = score_lookup[i-1][j]
                # Calculate score if match is accepted
                char_score = char_scores[i-1][j-1]
                prize = first_capture
                prize *= cpow(params.beta,
                              input_.indices.array[i-1] - first_capture_index)
                if word_use[i-1][j-1] > 1:
                    prize /= cpow(params.gamma, word_use[i-1][j-1] - 1)
                char_score += prize
                if char_score < 0.0:
                    c = 0.0
                else:
                    c = char_score
                word_score = word_scores[i-1][j-1] + w
                possibility2 = (cpow(c/m, params.lambda_) *
                                cpow(word_score/input_.W, (1-params.lambda_)))
                if score_lookup[i-1][j-1] > -1e19 and \
                   possibility2 > possibility1:
                    score_lookup[i][j] = possibility2
                    char_scores[i][j] = char_score
                    word_scores[i][j] = word_score
                    char_prizes[i][j] = prize
                    word_use[i][j] = word_use[i-1][j-1] + 1
                    pointers[i-1][j-1] = 1
                else:
                    score_lookup[i][j] = possibility1
                    char_scores[i][j] = char_scores[i-1][j]
                    word_scores[i][j] = word_scores[i-1][j]
                    word_use[i][j] = word_use[i-1][j]
                    pointers[i-1][j-1] = 0
                # update position in current word
            # Case where element of x in current position is a wildcard.
            # May either accept or reject this match
            elif input_.x.array[i-1] == -1:
                possibility1 = score_lookup[i-1][j]
                # Score if wildcard match is accepted, skipping current
                # char in shortform
                char_score = char_scores[i-1][j-1]
                char_score -= shortform.penalties.array[j-1]
                # Take min with zero to ensure char_score doesn't become
                # negative
                if char_score < 0.0:
                    c = 0
                else:
                    c = char_score
                word_score = word_scores[i-1][j-1]
                possibility2 = (cpow(c/m, params.lambda_) *
                                cpow(word_score/input_.W, 1-params.lambda_))
                if score_lookup[i-1][j-1] > -1e19 and \
                   possibility2 > possibility1:
                    score_lookup[i][j] = possibility2
                    char_scores[i][j] = char_score
                    word_scores[i][j] = word_score
                    char_prizes[i][j] = -shortform.penalties.array[j-1]
                    word_use[i][j] = word_use[i-1][j-1]
                    pointers[i-1][j-1] = 1
                else:
                    score_lookup[i][j] = possibility1
                    char_scores[i][j] = char_scores[i-1][j]
                    word_scores[i][j] = word_scores[i-1][j]
                    word_use[i][j] = word_use[i-1][j]
                    pointers[i-1][j-1] = 0
            # No match is possible. There is only one option to fill
            # current entry of dynamic programming lookup array.
            else:
                score_lookup[i][j] = score_lookup[i-1][j]
                char_scores[i][j] = char_scores[i-1][j]
                word_scores[i][j] = word_scores[i-1][j]
                word_use[i][j] = word_use[i-1][j]
                pointers[i-1][j-1] = 0
                # Update position in current word
            # Reset word_use to zero when word_boundary is passed
            if i == input_.word_boundaries[k] + 1:
                word_use[i][j] = 0
        # Increment number of words used when boundary is passed
        # Also reset position in current word
        if i == input_.word_boundaries[k] + 1:
            k += 1
    # Optimal score is in bottom right corner of lookup array
    score = score_lookup[n][m]
    # Free the memory used by the lookup array

    # Set score in output
    output.score = score
    # Trace backwards through pointer array to discover which elements of x
    # were matched and add the score associated to each character in the
    # shortform into the char scores array.
    i, j, k = n, m, 0
    while j > 0:
        if pointers[i - 1][j - 1]:
            output.char_prizes[m-k-1] = char_prizes[i][j]
            i -= 1
            j -= 1
            k += 1
        else:
            i -= 1
    # Free lookup arrays used by algorithm
    for i in range(n+1):
        PyMem_Free(score_lookup[i])
        PyMem_Free(word_use[i])
        PyMem_Free(char_scores[i])
        PyMem_Free(word_scores[i])
        PyMem_Free(char_prizes[i])
        if i < n:
            PyMem_Free(pointers[i])
    PyMem_Free(score_lookup)
    PyMem_Free(char_scores)
    PyMem_Free(word_scores)
    PyMem_Free(char_prizes)
    PyMem_Free(word_use)
    PyMem_Free(pointers)
