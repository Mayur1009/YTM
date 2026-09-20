import numpy as np
import pytest

from ytm._core.config import _get_unsinged_type

from .support import FB_NONE, FB_T1A, apply_feedback, discrete, host, make_buffers, set_clauses, set_ta_states, set_weights


@pytest.mark.parametrize(
    "val, dtype",
    [(2, np.uint8), (255, np.uint8), (256, np.uint8), (257, np.uint16), (65535, np.uint16), (65536, np.uint16), (65537, np.uint32)],
)
def test_type_choice_at_the_thresholds(val, dtype):
    """The chosen type must hold every value 0..val-1: 256 fits uint8 (max 255), 257 needs uint16."""
    assert _get_unsinged_type(val)[0] is dtype


@pytest.mark.parametrize("n_states", [256, 257, 65536, 65537])
def test_ta_state_saturates_at_both_ends_for_every_dtype(device, n_states):
    """max+1 wrapping to 0 (or 0-1 to max) at a dtype boundary would flip a literal from strongly included to excluded."""
    tm = discrete("binary", device, n_states=n_states)
    hi = n_states - 1
    states = np.full((4, 8), 0)
    states[0] = [hi, 0, hi, 0, 0, hi, 0, hi]  # true literals at max, false ones at 0, for X=[1,0,1,0]
    set_ta_states(tm, states)
    buf = make_buffers(tm, [[1, 0, 1, 0]], [[0.0]])
    apply_feedback(tm, buf, [[FB_T1A]] + [[FB_NONE]] * 3)
    assert list(host(tm, tm.dev.ta_states)[0]) == [hi, 0, hi, 0, 0, hi, 0, hi]


@pytest.mark.parametrize("feat_max", [255, 256])
def test_feature_values_at_the_top_of_their_dtype_are_not_wrapped(device, feat_max):
    """If X were narrowed one dtype step too far, 256 -> 0 and the clause below would not fire."""
    # feat_max 255 fits uint8 (values 0..255), 256 needs uint16.
    tm = discrete("binary", device, n_clauses=2, dim=(1, 1, 1), feat_maxs=feat_max)
    set_clauses(tm, {0: [feat_max - 1]})  # `x0 > feat_max - 1`, i.e. x0 == feat_max
    set_weights(tm, [[3, 0]])
    X = np.array([[feat_max], [feat_max - 1]])
    assert list(tm.score(X, force_repack=True)[:, 0]) == [3, 0]


def test_patch_id_above_255_reaches_the_update(device):
    """257 windows need a uint16 patch id. Only window 256 matches; if its id wrapped to 0 the feedback would read pixel 0 instead."""
    tm = discrete("binary", device, n_clauses=2, dim=(1, 257, 1), patch_dim=(1, 1))
    n_lits = tm.config._n_literals  # 256 position + 1 feature, doubled
    assert n_lits == 2 * (256 + 1)
    states = np.full((2, n_lits), tm.config._include_state - 1)
    states[0, 256] = tm.config._include_state  # include the single feature literal (index N_POSITION_FEATS)
    set_ta_states(tm, states)
    X = np.zeros((1, 1, 257, 1), dtype=int)
    X[0, 0, 256, 0] = 1
    buf = make_buffers(tm, X, [[0.0]])
    apply_feedback(tm, buf, [[FB_T1A], [FB_NONE]])
    out = host(tm, tm.dev.ta_states)[0]
    assert list(out[:256]) == [tm.config._include_state] * 256  # every position literal `px > k` (k<256) holds at px=256: +1
    assert out[256] == tm.config._include_state + 1  # feature literal: pixel is 1
