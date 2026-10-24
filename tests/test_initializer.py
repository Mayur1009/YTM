import numpy as np
import pytest

from ytm import ConstTAInit, SampleTAInit, UniformTAInit

from .support import discrete, host


def _states(tm):
    return host(tm, tm.dev.ta_states).astype(np.int64)


@pytest.mark.parametrize(
    ("ta_init", "cls", "low", "high"),
    [
        ("middle", ConstTAInit, None, None),
        ("random", UniformTAInit, None, None),
        ("random_include", UniformTAInit, -1, 1),
        ("random:2", UniformTAInit, -3, 2),
    ],
)
def test_string_ta_init_picks_the_matching_initializer(ta_init, cls, low, high):
    cfg = discrete("multi", ta_init=ta_init).config
    inc = cfg._include_state
    assert isinstance(cfg._ta_init, cls)
    if cls is ConstTAInit:
        assert cfg._ta_init.state == inc - 1
    elif low is not None:
        assert (cfg._ta_init.low, cfg._ta_init.high) == (inc + low, inc + high)


def test_const_ta_init_sets_every_state(device):
    tm = discrete("multi", device, ta_init=5)
    assert np.all(_states(tm) == 5)


def test_uniform_ta_init_stays_in_range(device):
    tm = discrete("multi", device, n_clauses=64, ta_init=UniformTAInit(low=10, high=20))
    s = _states(tm)
    assert s.min() >= 10 and s.max() < 20


def test_sample_ta_init_copies_the_sample_literals(device):
    X = np.array([[1, 0, 1, 1]])
    tm = discrete("multi", device, ta_init=SampleTAInit(X))
    inc = tm.config._include_state
    expected = np.array([1, 0, 1, 1, 0, 1, 0, 0], dtype=bool)
    assert np.array_equal(_states(tm) >= inc, np.broadcast_to(expected, (tm.config._total_clauses, 8)))


def test_sample_ta_init_with_y_seeds_positive_clauses_from_their_class(device):
    X = np.array([[1, 1, 0, 0], [0, 0, 1, 1]])
    Y = np.array([0, 1])
    tm = discrete("multi", device, n_clauses=4, coalesced=False, ta_init=SampleTAInit(X, Y))
    cfg = tm.config
    inc = cfg._include_state
    lits = {0: np.array([1, 1, 0, 0, 0, 0, 1, 1], dtype=bool), 1: np.array([0, 0, 1, 1, 1, 1, 0, 0], dtype=bool)}
    weights = host(tm, tm.dev.clause_weights)
    included = _states(tm) >= inc
    for clause in range(cfg._total_clauses):
        c, k = divmod(clause, cfg._n_clauses)
        sample = c if weights[c, k] > 0 else 1 - c
        assert np.array_equal(included[clause], lits[sample])


def test_sample_ta_init_conv_copies_one_patch(device):
    X = np.arange(9).reshape(1, 3, 3, 1) % 2
    tm = discrete("multi", device, n_clauses=16, dim=(3, 3, 1), patch_dim=(2, 2), ta_init=SampleTAInit(X))
    cfg = tm.config
    n_pos, half = cfg._n_position_feats, cfg._n_literals // 2
    patches = []
    for y in range(2):
        for x in range(2):
            v = X[0, y : y + 2, x : x + 2, 0].reshape(-1)
            lit = np.zeros(cfg._n_literals, dtype=bool)
            lit[n_pos + np.flatnonzero(v == 1)] = True
            lit[n_pos + half + np.flatnonzero(v == 0)] = True
            patches.append(lit)
    included = _states(tm) >= cfg._include_state
    for row in included:
        assert any(np.array_equal(row, p) for p in patches)
