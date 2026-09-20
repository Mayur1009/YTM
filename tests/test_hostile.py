import numpy as np
import pytest

from .support import discrete, guided, host, set_clauses, set_weights


def _xy(n=30, f=4, classes=2, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, (n, f), dtype=np.int32), rng.integers(0, classes, n)


# n_states=4 keeps the TA dtype uint8 but leaves 4..255 representable, so an unguarded inc past MAX_TA_STATE (3) or a dec wrapping
# 0 -> 255 is visible to `_assert_sane` (with the default 256 states every uint8 value is "valid" and the bound could never fail).
# 4 states also saturate within a few samples (init 1, include_state 2), so both guards are actually exercised.
NS = 4


def _assert_sane(tm):
    cfg = tm.config
    ta, w = host(tm, tm.dev.ta_states), host(tm, tm.dev.clause_weights)
    assert ta.max() <= cfg.n_states - 1  # the upper bound is what catches a dec that wrapped 0 -> 255 (dtype is unsigned)
    assert np.all(np.isfinite(w)) and np.abs(w).max() <= cfg.max_weight


_OVERFLOW_CAST = pytest.mark.filterwarnings("ignore:overflow encountered in cast:RuntimeWarning")


@pytest.mark.parametrize("T", [2.0**32, 1e30, pytest.param(1e39, marks=_OVERFLOW_CAST), 1e-30])
def test_extreme_T_keeps_probabilities_and_state_in_bounds(device, T):
    """Huge T underflows prob to 0 (dead but legal), 1e39 overflows the T_MAX float literal: no NaN may reach states or weights."""
    tm = discrete("multi", device, n_classes=2, T=T, n_states=NS)
    X, Y = _xy()
    tm.fit(X, Y)
    tm.fit(X, Y)
    _assert_sane(tm)
    # At T=1e39 the run is dead by design (T_MAX -> inf, prob NaN, no feedback): this only proves nothing non-finite leaked out.
    assert np.all(np.isfinite(tm.score(X)))


@pytest.mark.parametrize("s", [1.0, 1e30, 1e-30])
def test_extreme_s_keeps_ta_states_in_bounds(device, s):
    """s=1e30 makes geom_sample return ~1e30: the float->int cast must stay behind the loop guard. s<1 is clamped to 1."""
    tm = discrete("multi", device, n_classes=2, s=s, n_states=NS)
    X, Y = _xy()
    tm.fit(X, Y)
    _assert_sane(tm)


@pytest.mark.parametrize("lr", [0.0, 1e-30, 1e30])
@pytest.mark.parametrize("fb_signal", ["grad", "delta_l"])
def test_extreme_guided_learning_rate_keeps_weights_finite_and_clipped(device, lr, fb_signal):
    """lr=1e30 pushes `w + lr*grad` far past max_weight; it must clip, not overflow to inf/NaN."""
    # max_weight=10 (default 2^30 is unreachable in two 30-sample epochs) so that lr=1e30 must saturate at the bound.
    tm = guided("multi", device, n_classes=2, lr=lr, fb_signal=fb_signal, n_states=NS, max_weight=10.0)
    X, Y = _xy()
    tm.fit(X, Y)
    tm.fit(X, Y)
    _assert_sane(tm)
    if lr == 1e30:
        assert np.abs(host(tm, tm.dev.clause_weights)).max() == tm.config.max_weight


@pytest.mark.parametrize("make", [discrete, guided], ids=["discrete", "guided"])
def test_all_clauses_dropped_changes_nothing(device, make):
    """clause_drop_p=1 drops every clause: TA states and weights must be bit-for-bit unchanged."""
    X, Y = _xy()
    control = make("multi", device, n_classes=2, n_states=NS)
    c0 = (host(control, control.dev.ta_states), host(control, control.dev.clause_weights))
    control.fit(X, Y, clause_drop_p=0.0)
    assert not (
        np.array_equal(host(control, control.dev.ta_states), c0[0]) and np.array_equal(host(control, control.dev.clause_weights), c0[1])
    ), "control fit changed nothing, the p=1.0 assertion below would be vacuous"

    tm = make("multi", device, n_classes=2, n_states=NS)
    before = (host(tm, tm.dev.ta_states), host(tm, tm.dev.clause_weights))
    tm.fit(X, Y, clause_drop_p=1.0)
    assert np.array_equal(host(tm, tm.dev.ta_states), before[0])
    assert np.array_equal(host(tm, tm.dev.clause_weights), before[1])


@pytest.mark.parametrize("batch_size", [1, 3, 7, 100, -1])
def test_scoring_does_not_depend_on_batch_size(device, batch_size):
    """N=7 with batch sizes that do not divide it, exceed it, or equal 1: the tail batch must not be dropped or mis-offset."""
    # On CPU `calc_class_sums` ignores batch_size, so this is only meaningful on CUDA.
    tm = discrete("multi", device, n_classes=2, n_clauses=8)
    set_clauses(tm, {0: [0], 1: [1, 6], 2: [2], 5: [3]})
    set_weights(tm, np.arange(16, dtype=np.float32).reshape(2, 8) - 5)
    X = np.random.default_rng(3).integers(0, 2, (7, 4))
    assert np.array_equal(tm.score(X, batch_size=batch_size, force_repack=True), tm.score(X, force_repack=True))


@pytest.mark.parametrize("n_clauses", [1, 2, 3])
def test_tiny_clause_counts_train_and_score(device, n_clauses):
    """1 clause with negative_clauses (n_neg = 0) and an odd coalesced count must still index every array in range."""
    tm = discrete("multi", device, n_classes=2, n_clauses=n_clauses, n_states=NS)
    X, Y = _xy()
    tm.fit(X, Y)
    _assert_sane(tm)
    assert tm.score(X).shape == (30, 2)


def test_class_without_positive_samples_trains(device):
    """A class that never appears must not divide by zero or index an empty class list."""
    tm = discrete("multi", device, n_classes=3, n_states=NS)
    X, Y = _xy(classes=2)
    tm.fit(X, Y)
    _assert_sane(tm)


def test_more_threads_than_clauses_gives_the_same_scores():
    """omp with 8 threads on 2 clauses: idle threads must not perturb the reduction."""
    a = discrete("multi", "cpu:1", n_clauses=2, n_classes=2)
    b = discrete("multi", "cpu:8", n_clauses=2, n_classes=2)
    for tm in (a, b):
        set_clauses(tm, {0: [0], 1: [1]})
        set_weights(tm, [[1, 2], [3, 4]])
    X = np.random.default_rng(1).integers(0, 2, (9, 4))
    assert np.array_equal(a.score(X, force_repack=True), b.score(X, force_repack=True))
