from adeft.tests.util import OptimizationTestCase, StitchTestCase, \
    PermSearchTestCase


def test_perm_search():
    case1 = PermSearchTestCase(candidates=[[0], [0, 1], [0, 0, 1], [0, 0],
                                           [1]],
                               shortform=[1, 0],
                               prizes=[[1.0], [1.0, 0.5], [1.0, 0.5, 0.25],
                                       [1.0, 0.5], [1.0]],
                               penalties=[0.4, 0.2],
                               word_prizes=[1.0]*5,
                               word_penalties=[1.0, 2.0, 3.0, 4.0, 5.0],
                               beta=0.5,
                               rho=0.75,
                               inv_penalty=0.9,
                               len_perm=5,
                               result_score=(2/5)**(1/4) * (9/10))
    for case in [case1]:
        case.run_test()


def test_make_candidates_array():
    case1 = StitchTestCase(candidates=[[0], [0, 1], [1, 1, 0], [0, 0], [1]],
                           prizes=[[1.], [1., 0.5], [1., 0.5, 0.25],
                                   [1., 0.5], [1.]],
                           word_prizes=[1., 0.9, 0.8, 0.7, 0.6],
                           W_array=[0.6, 1.3, 2.1, 3., 4.],
                           permutation=[2, 0, 1, 3, 4],
                           result_x=[-1, 1, -1, 1, -1, 0, -1, 0, -1, 0,
                                     -1, 1, -1, 0, -1, 0, -1, 1, -1],
                           result_prizes=[0., 1., 0., 0.5, 0., 0.25,
                                          0., 1., 0., 1., 0., 0.5, 0.,
                                          1., 0., 0.5, 0., 1., 0.],
                           result_word_prizes=[0.8, 1., 0.9, 0.7, 0.6],
                           result_word_boundaries=[6, 8, 12, 16, 18])

    case2 = StitchTestCase(candidates=[[0], [0, 1], [1, 1, 0], [0, 0], [1]],
                           prizes=[[1.], [1., 0.5], [1., 0.5, 0.25],
                                   [1., 0.5], [1.]],
                           word_prizes=[1., 0.9, 0.8, 0.7, 0.6],
                           W_array=[0.6, 1.3, 2.1, 3., 4.],
                           permutation=[2, 0, 1],
                           result_x=[-1, 1, -1, 1, -1, 1, -1, 0, -1, 0,
                                     -1, 0, -1],
                           result_prizes=[0., 1., 0., 1., 0., 0.5,
                                          0., 0.25, 0., 1., 0., 0.5, 0.],
                           result_word_prizes=[0.6, 0.8, 0.7],
                           result_word_boundaries=[2, 8, 12])

    cases = [case1, case2]
    for case in cases:
        case.run_test()


def test_optimize():
    # Two words with beginning characters matching the characters in the
    # shortform
    case1 = OptimizationTestCase(x=[-1, 0, -1, 1, -1, 0, -1],
                                 y=[0, 1],
                                 prizes=[0., 1., 0., 1., 0., 0.5, 0.],
                                 penalties=[0.4, 0.2],
                                 word_boundaries=[2, 6],
                                 word_prizes=[1, 1],
                                 beta=0.5,
                                 rho=0.75,
                                 W=2.,
                                 result_score=1.0,
                                 result_char_scores=[1., 1.])

    # Single word containing consecutive characters from shortform
    case2 = OptimizationTestCase(x=[-1, 0, -1, 0, -1, 1, -1, 1, -1],
                                 y=[0, 1],
                                 prizes=[0., 1., 0., 0.5, 0.,
                                         0.25, 0., 0.125, 0.],
                                 penalties=[0.4, 0.2],
                                 word_boundaries=[8],
                                 word_prizes=[1],
                                 beta=0.5,
                                 rho=0.75,
                                 W=1.,
                                 result_score=(3/4)**(3/4),
                                 result_char_scores=[1.0, 0.5])

    # Three words, shortform matches in two places. Highest scoring match
    # has a larger word captured prize and smaller total letter prizes
    case3 = OptimizationTestCase(x=[-1, 0, -1, 1, -1, 0, -1, 0, -1, 1, -1],
                                 y=[0, 1],
                                 prizes=[0., 1., 0., 1., 0., 1., 0., 0.5,
                                         0., 0.25, 0.],
                                 penalties=[0.4, 0.2],
                                 word_boundaries=[2, 4, 10],
                                 word_prizes=[0.5, 0.5, 2.0],
                                 beta=0.5,
                                 rho=0.5,
                                 W=3.0,
                                 result_score=(3/4)**(1/2)*(2/3)**(1/2),
                                 result_char_scores=[1.0, 0.5])

    # Three words, shortform matches in two places. Highest scoring match
    # has larger total letter prizes and smaller word captured prizes
    case4 = OptimizationTestCase(x=[-1, 0, -1, 1, -1, 0, -1, 0, -1, 1, -1],
                                 y=[0, 1],
                                 prizes=[0., 1., 0., 1., 0., 1., 0., 0.5,
                                         0., 0.25, 0.],
                                 penalties=[0.4, 0.2],
                                 word_boundaries=[2, 4, 10],
                                 word_prizes=[0.5, 0.5, 1.25],
                                 beta=0.5,
                                 rho=0.75,
                                 W=2.25,
                                 result_score=(2/3)**(1/2),
                                 result_char_scores=[1., 1.])

    # Three words, shortform matches in two places with equal scores
    case5 = OptimizationTestCase(x=[-1, 0, -1, 1, -1, 0, -1, 0, -1, 1, -1],
                                 y=[0, 1],
                                 prizes=[0., 1., 0., 1., 0., 1., 0., 0.5,
                                         0., 0.25, 0.],
                                 penalties=[0.4, 0.2],
                                 word_boundaries=[2, 4, 10],
                                 word_prizes=[0.5, 0.5, 1.5],
                                 beta=0.5,
                                 rho=0.75,
                                 W=2.5,
                                 result_score=(2/5)**(1/4),
                                 result_char_scores=[1., 1.])

    # Two words. Only one character in shortform matches
    case6 = OptimizationTestCase(x=[-1, 0, -1, 0, -1],
                                 y=[0, 1],
                                 prizes=[0., 1., 0., 1., 0.],
                                 penalties=[0.4, 0.2],
                                 word_boundaries=[2, 4],
                                 word_prizes=[1., 0.5],
                                 beta=0.5,
                                 rho=0.75,
                                 W=1.5,
                                 result_score=(2/5)**(3/4)*(2/3)**(1/4),
                                 result_char_scores=[1., -0.2])

    test_cases = [case1, case2, case3, case4, case5, case6]
    for case in test_cases:
        case.check_assertions()
        case.run_test()
