import numpy as np
import pytest
from scipy import stats

from .support import (
    FB_NONE,
    FB_T1A,
    FB_T1B,
    FB_T2,
    apply_feedback,
    decide_feedback,
    discrete,
    fill,
    host,
    make_buffers,
    set_clauses,
    set_ta_states,
    set_weights,
)

ROW = [10, 20, 30, 40, 50, 60, 70, 80]  # x0..x3 then not x0..not x3


def _run(tm, X, fb_clause0, row):
    """Clause 0 gets `fb_clause0` on sample X, clause 1 gets nothing; returns clause 0's states."""
    states = np.full((tm.config._total_clauses, tm.config._n_literals), tm.config._include_state - 1)
    states[0] = row
    set_ta_states(tm, states)
    buf = make_buffers(tm, X, np.zeros((len(X), tm.config.n_classes)))
    apply_feedback(tm, buf, [[fb_clause0]] + [[FB_NONE]] * (tm.config._n_clauses - 1))
    return host(tm, tm.dev.ta_states)[0]


def test_type1a_binary_reinforces_true_literals_and_forgets_false_ones(device):
    """Type I on X=[1,0,1,0]: true x_i +1, false x_i -1, negations the other way. A swapped branch flips every entry."""
    tm = discrete("binary", device)
    out = _run(tm, [[1, 0, 1, 0]], FB_T1A, ROW)
    assert list(out) == [11, 19, 31, 39, 49, 61, 69, 81]


def test_type1a_without_negated_literals(device):
    """With negated_literals=False only the four feature literals exist; the N_LITERALS/2 offset must not be used."""
    tm = discrete("binary", device, negated_literals=False)
    out = _run(tm, [[1, 0, 1, 0]], FB_T1A, ROW[:4])
    assert list(out) == [11, 19, 31, 39]


def test_type1b_forgets_every_literal(device):
    """Type I-b with s=1 decrements all eight literals by one."""
    tm = discrete("binary", device)
    assert list(_run(tm, [[1, 0, 1, 0]], FB_T1B, ROW)) == [9, 19, 29, 39, 49, 59, 69, 79]


def test_type2_pushes_false_literals_towards_include(device):
    """Type II on X=[1,0,1,0]: literals false under X (x1, x3, not x0, not x2) +1, nothing else moves."""
    tm = discrete("binary", device)
    assert list(_run(tm, [[1, 0, 1, 0]], FB_T2, ROW)) == [10, 21, 30, 41, 51, 60, 71, 80]


def test_no_feedback_leaves_states_alone(device):
    """FB_NONE must be a true no-op on the states of a clause that is offered the sample."""
    tm = discrete("binary", device)
    assert list(_run(tm, [[1, 0, 1, 0]], FB_NONE, ROW)) == ROW


def test_type1a_thermometer(device):
    """feat_maxs=3, X=[2,0]. Feature 0 (lits 0-2, negations 6-8): k<2 true. Feature 1 (lits 3-5, negations 9-11): all false."""
    tm = discrete("binary", device, dim=(2, 1, 1), feat_maxs=3)
    out = _run(tm, [[2, 0]], FB_T1A, [50] * 12)
    assert list(out) == [51, 51, 49, 49, 49, 49, 49, 49, 51, 51, 51, 51]


def test_type2_thermometer(device):
    """feat_maxs=3, X=[2,0]: false literals are lit 2, all of feature 1, negations 6,7 (x0>0, x0>1 hold)."""
    tm = discrete("binary", device, dim=(2, 1, 1), feat_maxs=3)
    out = _run(tm, [[2, 0]], FB_T2, [50] * 12)
    assert list(out) == [50, 50, 51, 51, 51, 51, 51, 51, 50, 50, 50, 50]


def test_conv_type1a_updates_position_and_feature_literals_of_the_selected_window(device):
    """Type I on a conv clause hits the position and feature literals of the one window it matches (px=2 here)."""
    # dim=(1,4,1), patch (1,2), X=[0,0,1,0], clause "first pixel set" matches only the window at px=2, so it is selected.
    # Position lits 0,1 (`px > k`, k<2) +1; their negations 4,5 -1. Window pixels (1,0): feature lit 2 +1, not-lit 6 -1,
    # lit 3 -1, not-lit 7 +1.
    tm = discrete("binary", device, n_clauses=2, dim=(1, 4, 1), patch_dim=(1, 2))
    states = np.full((2, 8), 50)
    states[0, 2] = tm.config._include_state  # include feature 0
    set_ta_states(tm, states)
    X = np.array([[0, 0, 1, 0]]).reshape(1, 1, 4, 1)
    buf = make_buffers(tm, X, np.zeros((1, 1)))
    apply_feedback(tm, buf, [[FB_T1A], [FB_NONE]])
    assert list(host(tm, tm.dev.ta_states)[0]) == [51, 51, 129, 49, 49, 49, 49, 51]


def _decide_setup(device):
    tm = discrete("binary", device, n_clauses=6, max_includes=1)
    set_clauses(tm, {0: [0], 1: [0], 2: [1], 3: [1], 4: [0, 5], 5: [0, 5]})
    set_weights(tm, [[2, -2, 2, -2, 2, -2]])
    return tm


X_DEC = [[1, 0, 0, 0]]  # fires: c0, c1 (x0), c4, c5 (x0 and not x1). Silent: c2, c3 (x1)


@pytest.mark.parametrize(
    "y, votes, expected",
    [
        (+10.0, -10.0, [FB_T1A, FB_T2, FB_T1B, FB_NONE, FB_T1B, FB_T2]),
        (-10.0, +10.0, [FB_T2, FB_T1A, FB_NONE, FB_T1B, FB_T2, FB_T1B]),
    ],
    ids=["target+", "target-"],
)
def test_decide_feedback_follows_polarity_target_output_and_space(device, y, votes, expected):
    """Positive weight + positive target -> Type I (a: fired with room, b: otherwise); opposite signs -> Type II only if fired."""
    tm = _decide_setup(device)
    buf = make_buffers(tm, X_DEC, [[y]])
    fb = decide_feedback(tm, buf, [votes])
    assert list(fb.ravel()) == expected


def test_dropped_clauses_get_no_feedback(device):
    """The drop mask must veto feedback regardless of votes."""
    tm = _decide_setup(device)
    buf = make_buffers(tm, X_DEC, [[10.0]])
    mask = np.zeros(6, dtype=np.int8)
    mask[0] = 1
    fill(tm, buf.clause_drop_mask, mask)
    assert decide_feedback(tm, buf, [-10.0]).ravel()[0] == FB_NONE


TRIALS = 4000
ALPHA = 1e-4  # ~25 binomtests per device, keep the family-wise false-failure rate low


def _drift(tm, X, fb, key_base=1):
    """Apply `fb` to clause 0 TRIALS times with different keys; return per-literal net change."""
    cfg = tm.config
    start = 30000
    states = np.full((cfg._total_clauses, cfg._n_literals), start)
    set_ta_states(tm, states)
    buf = make_buffers(tm, X, np.zeros((len(X), cfg.n_classes)))
    fbs = [[fb]] + [[FB_NONE]] * (cfg._n_clauses - 1)
    for t in range(TRIALS):
        apply_feedback(tm, buf, fbs, key=key_base + t)
    return host(tm, tm.dev.ta_states)[0].astype(np.int64) - start


@pytest.mark.statistical
@pytest.mark.parametrize("s", [2.0, 4.0])
def test_type1a_forgetting_frequency_matches_one_over_s(device, s):
    """Each false literal (and each true literal's negation) is decremented with probability 1/s per application."""
    tm = discrete("binary", device, s=s, n_states=65536)
    d = _drift(tm, [[1, 0, 1, 0]], FB_T1A)
    # x0, x2 true: inc every time (boost). x1, x3 false: dec w.p. 1/s. not-x0, not-x2: dec w.p. 1/s. not-x1, not-x3: inc every time.
    assert list(d[[0, 2, 5, 7]]) == [TRIALS] * 4
    for i in (1, 3, 4, 6):
        assert stats.binomtest(int(-d[i]), TRIALS, 1 / s).pvalue > ALPHA, i


@pytest.mark.statistical
def test_type1a_without_boost_increments_with_probability_one_minus_one_over_s(device):
    """boost_tp_inc=False: true literals grow with probability 1 - 1/s, not always."""
    tm = discrete("binary", device, s=4.0, n_states=65536, boost_tp_inc=False)
    d = _drift(tm, [[1, 0, 1, 0]], FB_T1A)
    for i in (0, 2, 5, 7):
        assert stats.binomtest(int(d[i]), TRIALS, 0.75).pvalue > ALPHA, i


@pytest.mark.statistical
def test_type1b_decrements_every_literal_with_probability_one_over_s(device):
    """The geometric skipping in type1b must be marginally Bernoulli(1/s) for every literal, including the last ones."""
    tm = discrete("binary", device, s=4.0, n_states=65536)
    d = _drift(tm, [[1, 0, 1, 0]], FB_T1B)
    for i in range(8):
        assert stats.binomtest(int(-d[i]), TRIALS, 0.25).pvalue > ALPHA, i


@pytest.mark.statistical
def test_type1a_thermometer_geometric_skip_frequencies_match_one_over_s(device):
    """The geometric-skip recurrence over thermometer ranges must be marginally Bernoulli(1/s) per literal, not just for binary."""
    s = 4.0
    tm = discrete("binary", device, s=s, n_states=65536, dim=(2, 1, 1), feat_maxs=3)
    d = _drift(tm, [[2, 0]], FB_T1A)
    # X=[2,0]: feature 0 val=2 (lits 0-2, negations 6-8), feature 1 val=0 (lits 3-5, negations 9-11). boost_tp_inc: "inc" ranges always +1.
    assert list(d[[0, 1, 8, 9, 10, 11]]) == [TRIALS] * 6
    # "dec" ranges go through prob_dec_literals_in_range with S_INV: lit 2, lits 3-5, negations 6-7.
    for i in (2, 3, 4, 5, 6, 7):
        assert stats.binomtest(int(-d[i]), TRIALS, 1 / s).pvalue > ALPHA, i
