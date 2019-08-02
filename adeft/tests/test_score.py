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
    score, ind = check_optimize()
    assert score == 4
    assert ind == [3, 1]


def test_perm_search():
    score = check_perm_search()
    assert score == 4
