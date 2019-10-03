cdef struct int_array:
    int *array
    int length


cdef struct double_array:
    double *array
    int length

    
cdef struct opt_results:
    double score
    double *char_prizes

    
cdef struct candidates_array:
    int_array **array
    int_array **indices
    double *word_prizes
    double *W_array
    int *cum_lengths
    int length


cdef struct opt_input:
    int_array *x
    int_array *indices
    int *word_boundaries
    double_array *word_prizes
    double W


cdef struct opt_params:
    double alpha, beta, gamma, lambda_


cdef struct opt_shortform:
    int_array *y
    double_array *penalties


cdef struct perm_out:
    double score


cdef opt_results *make_opt_results(int len_y)
cdef void free_opt_results(opt_results *results)

cdef int_array *make_int_array(int length)
cdef void free_int_array(int_array *x)

cdef double_array *make_double_array(int length)
cdef void free_double_array(double_array *x)

cdef candidates_array *make_candidates_array(list encoded_candidates,
                                             list indices,
                                             list word_prizes,
                                             list W)
cdef void free_candidates_array(candidates_array *candidates)

cdef opt_input *make_opt_input(int n, int num_words)
cdef void free_opt_input(opt_input *input_)

cdef opt_params *make_opt_params(double alpha, double beta,
                                 double gamma, double lambda_)
cdef void free_opt_params(opt_params *params)

cdef opt_shortform *create_shortform(list encoded_shortform,
                                     list penalties)
cdef opt_shortform *make_opt_shortform(int m)

cdef void free_opt_shortform(opt_shortform *shortform)

cdef void opt_search(candidates_array *candidates,
                     opt_shortform *shortform,
                     opt_params *params,
                     float inv_penalty,
                     int n,
                     int max_inversions,
                     opt_results *output)

cdef void stitch(candidates_array *candidates, int *permutation,
                  int len_parm, opt_input *result)

cdef void optimize(opt_input *input_, opt_shortform *shortform,
                    opt_params *params, opt_results *output)






