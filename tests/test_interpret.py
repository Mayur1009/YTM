import numpy as np

from .support import discrete, set_clauses, set_weights


def _tm(device, **kw):
    kw.setdefault("coalesced", True)
    return discrete("multi", device, **kw)  # 4 clauses, 2 classes, dim (4,1,1)


def test_wic_sums_signed_feature_contributions_by_polarity(device):
    """c0: x0 AND NOT x1 (w=+3), c1: x0 (w=+2), c2: x2 (w=-2). Positive polarity -> [5,-3,0,0]; negative -> [0,0,2,0]."""
    tm = _tm(device)
    set_clauses(tm, {0: [0, 5], 1: [0], 2: [2]})
    set_weights(tm, [[3, 2, -2, 0], [0, 0, 0, 0]])
    assert np.array_equal(tm.wic(0, +1, force_repack=True).ravel(), [5, -3, 0, 0])
    assert np.array_equal(tm.wic(0, -1, force_repack=True).ravel(), [0, 0, 2, 0])


def test_wic_ignores_contradictory_clauses(device):
    """`has_contra` clauses have no meaningful bounds and must contribute nothing."""
    tm = _tm(device)
    set_clauses(tm, {0: [0, 4], 1: [1]})
    set_weights(tm, [[3, 2, 0, 0], [0, 0, 0, 0]])
    assert np.array_equal(tm.wic(0, +1, force_repack=True).ravel(), [0, 2, 0, 0])


def test_wac_counts_only_clauses_that_fire_on_the_sample(device):
    """Same clauses as above: X=[1,0,0,0] fires c0 and c1, X=[0,...] fires neither."""
    tm = _tm(device)
    set_clauses(tm, {0: [0, 5], 1: [0], 2: [2]})
    set_weights(tm, [[3, 2, -2, 0], [0, 0, 0, 0]])
    X = np.array([[1, 0, 0, 0], [0, 0, 0, 0]])
    out = tm.wac(X, np.array([0, 0]), +1, force_repack=True).reshape(2, 4)
    assert np.array_equal(out, [[5, -3, 0, 0], [0, 0, 0, 0]])


def _small_patch_tm(device):
    return discrete("binary", device, n_clauses=2, dim=(10, 10, 1), patch_dim=(8, 8), stride=(5, 5), feat_maxs=1)


# RED, real bug: `wic` writes the raw patch feature index instead of the image offset when N_PATCHES == 1.
# Cause: interpret.c:51 `output[k] += cp * wm;` in the `#else` branch of `wic`; it should use `feature_offset(k, 0, 0)`.
# Reproducer: this geometry, clause 0 includes feature 8, weight +2 -> got 2.0 at [0, 8], expected 2.0 at [1, 0].
def test_wic_puts_a_patch_feature_at_its_image_pixel_when_patch_is_smaller_than_the_image(device):
    """Patch feature 8 is window pixel (1,0) = image flat index 10; `wic` wrote it at the raw index 8."""
    tm = _small_patch_tm(device)
    set_clauses(tm, {0: [8]})
    set_weights(tm, [[2, 0]])
    out = tm.wic(0, +1, force_repack=True).reshape(10, 10)
    expected = np.zeros((10, 10), dtype=np.float32)
    expected[1, 0] = 2
    assert np.array_equal(out, expected)


def test_wac_puts_a_patch_feature_at_its_image_pixel_when_patch_is_smaller_than_the_image(device):
    """PROBE (extra): same geometry as the wic test; wac must also place patch feature 8 at image pixel (1,0).

    Calls `tm.dev.wac` with a hand-prepared X because `tm.wac` -> `_prepare_X` cannot handle this geometry at all
    (`_feat_mins.reshape(_dim)` on a per-patch-feature array raises ValueError), a separate finding."""
    tm = _small_patch_tm(device)
    set_clauses(tm, {0: [8]})
    set_weights(tm, [[2, 0]])
    X = np.zeros((1, 10, 10, 1), dtype=tm.config._fbound_dtype)
    X[0, 1, 0, 0] = 1
    out = tm.dev.wac(X, np.array([0]), +1, force_repack=True).reshape(10, 10)
    expected = np.zeros((10, 10), dtype=np.float32)
    expected[1, 0] = 2
    assert np.array_equal(out, expected)


# RED, real bug: src/ytm/_core/base.py:39 `cfg._feat_mins.reshape(cfg._dim)` in the `n_patches == 1` branch of `_prepare_X`;
# `_feat_mins` has one entry per patch feature (64 here), `_dim` has 100 pixels, so it raises ValueError.
def test_score_runs_when_a_single_window_is_smaller_than_the_image(device):
    """One window smaller than the image made `_prepare_X` reshape 64 bounds into 100 pixels, so score/fit/wac raised ValueError."""
    tm = _small_patch_tm(device)
    X = np.zeros((2, 10, 10, 1), dtype=np.uint8)
    assert tm.score(X).shape == (2, 1)
