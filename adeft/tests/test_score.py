from adeft.score._score import OptimizationTestCase, StitchTestCase, \
    PermSearchTestCase


def test_perm_search():
    case1 = PermSearchTestCase(encoded_tokens=[[(1, 0)], [(1, 0), (0, 1)],
                                               [(1, 0), (1, 1), (0, 2)],
                                               [(1, 0), (1, 1)], [(0, 0)]],
                               shortform=[0, 1],
                               penalties=[0.4, 0.2],
                               word_prizes=[1.0]*5,
                               W=5.0,
                               alpha=0.5,
                               beta=0.8,
                               gamma=0.85,
                               lambda_=0.75,
                               rho=0.9,
                               result_score=(2/5)**(1/4) * (9/10))
    case2 = PermSearchTestCase(encoded_tokens=[[(0, 0), (1, 3), (0, 5)],
                                               [(1, 0), (0, 1),
                                                (0, 3), (1, 6)]],
                               shortform=[0, 1],
                               penalties=[1.0, 0.4],
                               word_prizes=[1.0, 1.0],
                               W=2.0,
                               alpha=0.5,
                               beta=0.8,
                               gamma=0.85,
                               lambda_=0.6,
                               rho=0.9,
                               result_score=1.0)
    for case in [case1, case2]:
        case.run_test()


def test_make_candidates_array():
    case1 = StitchTestCase(encoded_tokens=[[(0, 0)], [(0, 0), (1, 1)],
                                           [(1, 0), (1, 1), (0, 2)],
                                           [(0, 0), (0, 1)], [(1, 0)]],
                           word_prizes=[1., 0.9, 0.8, 0.7, 0.6],
                           W=4.0,
                           permutation=[2, 0, 1, 3, 4],
                           result_x=[-1, -1, 1, -1, -1, 1, -1, -1, 0,
                                     -1, -1, -2, 0, -1, -1, -2, 0, -1,
                                      -1, 1, -1, -1, - 2, 0, -1, -1,
                                      0, -1, -1, -2, 1, -1, -1, -2],
                           result_indices=[-1, -1, 0, -1, -1, 1, -1,
                                           -1, 2, -1, -1, -1, 0, -1,
                                           -1, -1, 0, -1, -1, 1, -1,
                                           -1, -1, 0, -1, -1, 1, -1,
                                           -1, -1, 0, -1, -1, -1],
                           result_word_prizes=[0.8, 1., 0.9, 0.7, 0.6],
                           shortform_length=3)
    cases = [case1]
    for case in cases:
        case.run_test()


def test_optimize():
    case1 = OptimizationTestCase(x=[-1, 0, -1, -2, 1, -1, 0, -1],
                                 y=[0, 1],
                                 indices=[-1, 0, -1, -1, 0, -1, 1, -1],
                                 penalties=[0.4, 0.2],
                                 word_prizes=[1, 1],
                                 alpha=0.5,
                                 beta=0.8,
                                 gamma=0.85,
                                 lambda_=0.75,
                                 W=2.,
                                 result_score=1.0,
                                 result_char_scores=[1., 1.])

    case2 = OptimizationTestCase(x=[-1, 0, -1, 0, -1, 1, -1, 1, -1],
                                 y=[0, 1],
                                 indices=[-1, 0, -1, 1, -1,
                                          2, -1, 3, -1],
                                 penalties=[0.4, 0.2],
                                 word_prizes=[1],
                                 alpha=0.5,
                                 beta=0.8,
                                 gamma=0.85,
                                 lambda_=0.75,
                                 W=1.,
                                 result_score=(1.8/2)**(3/4),
                                 result_char_scores=[1.0, 0.8])

    case3 = OptimizationTestCase(x=[-1, 0, -1, -2, 1, -1, -2, 0,
                                    -1, 0, -1, 1, -1],
                                 y=[0, 1],
                                 indices=[-1, 0, -1, -1, 0, -1, -1, 0, -1, 1,
                                         -1, 2, -1],
                                 penalties=[0.4, 0.2],
                                 word_prizes=[0.5, 0.5, 2.0],
                                 alpha=0.5,
                                 beta=0.8,
                                 gamma=0.85,
                                 lambda_=0.5,
                                 W=3.0,
                                 result_score=(1.8/2)**(1/2)*(2/3)**(1/2),
                                 result_char_scores=[1.0, 0.8])

    case4 = OptimizationTestCase(x=[-1, 0, -1, -2, 1, -1, -2, 0,
                                    -1, 0, -1, 1, -1],
                                 y=[0, 1],
                                 indices=[-1, 0, -1, -1, 0, -1, -1, 0,
                                          -1, 1,-1, 2, -1],
                                 penalties=[0.4, 0.2],
                                 word_prizes=[0.5, 0.5, 1.25],
                                 alpha=0.5,
                                 beta=0.8,
                                 gamma=0.85,
                                 lambda_=0.75,
                                 W=2.25,
                                 result_score=(2/3)**(1/2),
                                 result_char_scores=[1., 1.])

    case5 = OptimizationTestCase(x=[-1, 0, -1, -2, 1, -1, -2, 0,
                                    -1, 0, -1, 1, -1],
                                 y=[0, 1],
                                 indices=[-1, 0, -1, -1, 0, -1, -1, 0,
                                          -1, 1, -1, 2, -1],
                                 penalties=[0.4, 0.2],
                                 word_prizes=[0.5, 0.5, 1],
                                 alpha=0.5,
                                 beta=0.8,
                                 gamma=0.85,
                                 lambda_=0.75,
                                 W=2.5,
                                 result_score=(2/5)**(1/4),
                                 result_char_scores=[1., 1.])

    # Two words. Only one character in shortform matches
    case6 = OptimizationTestCase(x=[-1, 0, -1, -2, 0, -1],
                                 y=[0, 1],
                                 indices=[-1, 0, -1, -1, 0, -1],
                                 penalties=[0.4, 0.2],
                                 word_prizes=[1., 0.5],
                                 alpha=0.5,
                                 beta=0.8,
                                 gamma=0.85,
                                 lambda_=0.75,
                                 W=1.5,
                                 result_score=(2/5)**(3/4)*(2/3)**(1/4),
                                 result_char_scores=[1., -0.2])

    # INDRA with tokens permuted
    char_score = (3 + 0.8**4 - 0.4**2)/5
    case7 = OptimizationTestCase(x=[-1, 4, -1, 3, -1, -2, 3, -1, 4, -1, 1, -1,
                                    0, -1, 1, -1, -2, 2, -1, 1, -1, 4, -1, 0,
                                    -1, 4, -1, -2, 0, -1, 1, -1, 3, -1, 4, -1,
                                    2, -1, -2, 1, -1, 3, -1, -2, 4, -1, 1, -1,
                                    2, -1],
                                 y=[0, 1, 2, 3, 4],
                                 indices=[-1, 0, -1, 8, -1, -1, 0,
                                          -1, 2, -1, 5, -1,
                                          6, -1, 7, -1, -1, 0,
                                          -1, 2, -1, 3, -1, 5,
                                          -1, 7, -1, -1, 0, -1, 1,
                                          -1, 5, -1, 6, -1,
                                          9, -1, -1, 0, -1, 5,
                                          -1, -1, 0, -1, 1, -1, 2, -1],
                                 penalties=[0.4**i for i in range(5)],
                                 word_prizes=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                                 alpha=0.5,
                                 beta=0.8,
                                 gamma=0.85,
                                 lambda_=0.6,
                                 W=6.0,
                                 result_score=(char_score)**0.6*(1/2)**0.4,
                                 result_char_scores=[1., 1., -0.4**2,
                                                     0.8**4, 1.])

    # Estrogen Receptor (ER)
    case8 = OptimizationTestCase(x=[-1, 0, -1, 1, -1, 0, -1, -2, 1,
                                    -1, 0, -1, 0, -1, 1, -1],
                                 y=[0, 1],
                                 indices=[-1, 0, -1, 3, -1, 5, -1, -1,
                                          0, -1, 1, -1, 4, -1, 6, -1],
                                 penalties=[1.0, 0.4],
                                 word_prizes=[1.0, 1.0],
                                 alpha=0.5,
                                 beta=0.8,
                                 gamma=0.85,
                                 lambda_=0.6,
                                 W=2.0,
                                 result_score=1.0,
                                 result_char_scores=[1.0, 1.0])

    test_cases = [case1, case2, case3, case4, case5, case6, case7, case8]
    for case in test_cases:
        case.check_assertions()
        case.run_test()
