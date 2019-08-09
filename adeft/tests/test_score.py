from adeft.score.score_util import check_optimize, check_perm_search, \
    check_make_candidates_array


def test_make_candidates_array():
    x, p, wp, wb = check_make_candidates_array()
    assert x == [-1, 1, -1, 1, -1, 0, -1, 0, -1, 0, -1, 1, -1, 0,
                 -1, 0, -1, 1, -1]
    assert p == [0., 1., 0., 0.5, 0., 0.25, 0., 1., 0., 1., 0.,
                 0.5, 0., 1., 0., 0.5, 0., 1., 0.]
    assert wp == [1.0, 1.0, 1.0, 1.0, 1.0]
    assert wb == [6, 8, 12, 16, 18]


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
                                 result_indices=[3, 1])

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
                                 result_indices=[5, 1])

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
                                 result_indices=[9, 5])

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
                                 result_indices=[3, 1])

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
                                 result_indices=[3, 1])

    test_cases = [case1, case2, case3, case4, case5]
    for case in test_cases:
        case.check_assertions()
        case.run_test()


def test_perm_search():
    score = check_perm_search()
    assert score == 4


class OptimizationTestCase(object):
    def __init__(self, x=None, y=None,
                 prizes=None, penalties=None,
                 word_boundaries=None, word_prizes=None, alpha=None,
                 result_score=None, result_indices=None):
        self.x = x
        self.y = y
        self.prizes = prizes
        self.penalties = penalties
        self.word_boundaries = word_boundaries
        self.word_prizes = word_prizes
        self.alpha = alpha
        self.n = len(x)
        self.m = len(y)
        self.num_words = len(word_boundaries)
        self.result_score = result_score
        self.result_indices = result_indices

    def check_assertions(self):
        assert len(self.prizes) == self.n
        assert len(self.penalties) == self.m
        assert len(self.word_prizes) == self.num_words
        assert self.word_boundaries[-1] == len(self.x) - 1
        assert self.word_boundaries == sorted(self.word_boundaries)

    def run_test(self):
        score, ind = check_optimize(self)
        assert score == self.result_score
        assert ind == self.result_indices
