from adeft.score.score_util import OptimizationTestCase, StitchTestCase


def test_make_candidates_array():
    case1 = StitchTestCase(shortform=[0, 1],
                           candidates=[[0], [0, 1], [1, 1, 0], [0, 0], [1]],
                           prizes=[[1.], [1., 0.5], [1., 0.5, 0.25],
                                   [1., 0.5], [1.]],
                           penalties=[0.4, 0.2],
                           word_prizes=[1., 0.9, 0.8, 0.7, 0.6],
                           permutation=[2, 0, 1, 3, 4],
                           result_x=[-1, 1, -1, 1, -1, 0, -1, 0, -1, 0,
                                     -1, 1, -1, 0, -1, 0, -1, 1, -1],
                           result_prizes=[0., 1., 0., 0.5, 0., 0.25,
                                          0., 1., 0., 1., 0., 0.5, 0.,
                                          1., 0., 0.5, 0., 1., 0.],
                           result_word_prizes=[0.8, 1., 0.9, 0.7, 0.6],
                           result_word_boundaries=[6, 8, 12, 16, 18])

    case2 = StitchTestCase(shortform=[0, 1],
                           candidates=[[0], [0, 1], [1, 1, 0], [0, 0], [1]],
                           prizes=[[1.], [1., 0.5], [1., 0.5, 0.25],
                                   [1., 0.5], [1.]],
                           penalties=[0.4, 0.2],
                           word_prizes=[1., 0.9, 0.8, 0.7, 0.6],
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
                                 alpha=0.5,
                                 result_score=4.0,
                                 result_char_scores=[1., 1.])

    # Single word containing consecutive characters from shortform
    case2 = OptimizationTestCase(x=[-1, 0, -1, 0, -1, 1, -1, 1, -1],
                                 y=[0, 1],
                                 prizes=[0., 1., 0., 0.5, 0.,
                                         0.25, 0., 0.125, 0.],
                                 penalties=[0.4, 0.2],
                                 word_boundaries=[8],
                                 word_prizes=[1],
                                 alpha=0.5,
                                 result_score=2.5,
                                 result_char_scores=[1.0, 0.5])

    # Three words, shortform matches in two places. Highest scoring match
    # has a larger word captured prize and smaller total letter prizes
    case3 = OptimizationTestCase(x=[-1, 0, -1, 1, -1, 0, -1, 0, -1, 1, -1],
                                 y=[0, 1],
                                 prizes=[0., 1., 0., 1., 0., 1., 0., 0.5,
                                         0., 0.25, 0.],
                                 penalties=[0.4, 0.2],
                                 word_boundaries=[2, 4, 10],
                                 word_prizes=[0.5, 0.5, 1.75],
                                 alpha=0.5,
                                 result_score=3.25,
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
                                 alpha=0.5,
                                 result_score=3.,
                                 result_char_scores=[1., 1.])

    # Three words, shortform matches in two places with equal scores
    case5 = OptimizationTestCase(x=[-1, 0, -1, 1, -1, 0, -1, 0, -1, 1, -1],
                                 y=[0, 1],
                                 prizes=[0., 1., 0., 1., 0., 1., 0., 0.5,
                                         0., 0.25, 0.],
                                 penalties=[0.4, 0.2],
                                 word_boundaries=[2, 4, 10],
                                 word_prizes=[0.5, 0.5, 1.5],
                                 alpha=0.5,
                                 result_score=3.,
                                 result_char_scores=[1., 1.])

    test_cases = [case1, case2, case3, case4, case5]
    for case in test_cases:
        case.check_assertions()
        case.run_test()
