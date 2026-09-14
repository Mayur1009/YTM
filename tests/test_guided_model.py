"""Python layer checks for failures that are silent, mirroring `tests/test_model.py` for guided.

None of these assert accuracy. They assert that state survives a round trip (including the raw,
pre-activation votes), that the arrays reaching the backend still line up with each other, and that
the per sample rng key actually varies. Each of those degrades a model quietly rather than raising.
"""

import copy
import ctypes
import pickle

import numpy as np
import pytest

from ytm._guided import BinaryTM, MultiClassTM, RegressionTM


@pytest.fixture(scope="module")
def trained():
    """A trained model, because an untrained one hides anything that confuses the initial arrays
    with the loaded ones."""
    rng = np.random.default_rng(0)
    X = rng.integers(0, 2, size=(200, 16), dtype=np.int32)
    Y = ((X[:, :8].sum(1) > X[:, 8:].sum(1)).astype(int) + (X[:, 0] == 1)) % 3

    tm = MultiClassTM(64, 2.0, (4, 4), 3, feat_maxs=1, seed=1)
    for _ in range(6):
        tm.fit(X, Y)

    assert not np.array_equal(tm.get_ta_states(), MultiClassTM(64, 2.0, (4, 4), 3, feat_maxs=1, seed=1).get_ta_states())
    return tm, X


class TestRoundTrip:
    """`load_state_dict` once rebound the arrays while the ctypes pointers still addressed the ones
    `dev_init` allocated. On a fresh model those hold the same values, so only a trained model shows
    it, and the symptom is the C reading a different model than python reports."""

    @pytest.mark.parametrize("clone", [pickle, copy], ids=["pickle", "deepcopy"])
    def test_predictions_survive(self, trained, clone):
        tm, X = trained
        back = pickle.loads(pickle.dumps(tm)) if clone is pickle else copy.deepcopy(tm)
        assert np.array_equal(back.predict(X)[0], tm.predict(X)[0])

    def test_predictions_survive_a_device_move(self, trained):
        tm, X = trained
        before = tm.predict(X)[0]
        moved = pickle.loads(pickle.dumps(tm))  # leave the fixture alone
        moved.to("cpu:2")
        assert moved.device_config.device == "cpu:2"
        assert np.array_equal(moved.predict(X)[0], before)

    def test_the_c_side_sees_the_loaded_arrays(self, trained):
        """Comparing python arrays is not enough: they were correct even when the pointers were stale."""
        tm, _ = trained
        back = pickle.loads(pickle.dumps(tm))
        for name in ("ta_states", "clause_weights", "patch_weights"):
            arr = getattr(back.dev, name)
            ptr = getattr(back.dev, f"p_{name}")
            assert arr.ctypes.data == ctypes.cast(ptr, ctypes.c_void_p).value, name

    def test_a_reloaded_model_continues_its_rng_rather_than_replaying(self, trained):
        """Both generators are stored as `bit_generator.state`. Reseeding instead would make a
        resumed run redraw the shuffles and drop masks it already used."""
        tm, _ = trained
        a, b = pickle.loads(pickle.dumps(tm)), pickle.loads(pickle.dumps(tm))

        # compared before anything draws, since drawing advances the generator
        assert a._rng.bit_generator.state == tm._rng.bit_generator.state
        assert a.dev._rng.bit_generator.state == tm.dev._rng.bit_generator.state

        assert np.array_equal(a._rng.random(5), b._rng.random(5)), "two loads must agree"

        fresh = MultiClassTM(64, 2.0, (4, 4), 3, feat_maxs=1, seed=1)
        assert b._rng.bit_generator.state != fresh._rng.bit_generator.state, "a reload must not reseed"

    def test_raw_votes_survive_round_trip(self, trained):
        """`raw_votes` bypasses activation; a stale pointer here would be invisible to the
        `predict`-based checks above if only the activated path happened to still be correct."""
        tm, X = trained
        back = pickle.loads(pickle.dumps(tm))
        assert np.array_equal(back.raw_votes(X), tm.raw_votes(X))

    def test_raw_votes_are_not_activated(self, trained):
        """The whole point of `raw_votes` is to skip the softmax normalization `predict` applies."""
        tm, X = trained
        raw = tm.raw_votes(X)
        activated = tm.predict(X)[1]
        assert not np.allclose(raw.sum(axis=1), 1.0)
        assert np.allclose(activated.sum(axis=1), 1.0)


class TestFitAlignment:
    def test_x_y_stay_row_aligned_through_the_shuffle(self):
        """`_fit` permutes X and Y together. If they fell out of step the model would train on
        mismatched pairs and simply be worse, with nothing raised."""
        seen = {}

        class Spy(MultiClassTM):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                self.dev.fit_epoch = lambda X, Y, p, bs, lr, lambda_: seen.update(X=X.copy(), Y=Y.copy())

        n, bits = 40, 6
        X = np.zeros((n, 16), dtype=np.int32)
        for i in range(n):  # bits 0..5 spell the row index, so a shuffled row can be identified
            X[i, :bits] = [(i >> b) & 1 for b in range(bits)]
        Y = np.arange(n) % 3

        tm = Spy(8, 2.0, (4, 4), 3, feat_maxs=1, seed=1)
        tm.fit(X, Y, shuffle=True)

        flat = seen["X"].reshape(n, -1)
        recovered = []
        for row in range(n):
            i = sum(int(flat[row, b]) << b for b in range(bits))
            recovered.append(i)
            assert seen["Y"][row, Y[i]] == 1.0, f"row {row} carries sample {i}, but Y does not agree"

        assert sorted(recovered) == list(range(n)), "the shuffle must be a permutation, not a resample"
        assert recovered != list(range(n)), "shuffle=True should actually reorder"

    def test_shuffle_off_preserves_the_original_order(self):
        seen = {}

        class Spy(MultiClassTM):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                self.dev.fit_epoch = lambda X, Y, p, bs, lr, lambda_: seen.update(X=X.copy())

        X = np.random.default_rng(2).integers(0, 2, size=(20, 16), dtype=np.int32)
        tm = Spy(8, 2.0, (4, 4), 3, feat_maxs=1, seed=1)
        tm.fit(X, np.arange(20) % 3, shuffle=False)
        assert np.array_equal(seen["X"].reshape(20, -1), X)


def test_every_sample_gets_a_different_rng_key():
    """One reused key would give every sample the same patch choices and the same feedback coin
    flips. The model would still train, just worse, with nothing raised."""
    tm = MultiClassTM(8, 2.0, (4, 4), 3, feat_maxs=1, seed=1)
    pbar = range(500)
    pairs = list(tm.dev._fit_samples(pbar))

    assert [e for e, _ in pairs] == list(pbar)
    keys = [k for _, k in pairs]
    assert len(set(keys)) == len(keys)
    assert all(k > 0 for k in keys)


class TestGetClauses:
    @pytest.mark.parametrize("coalesced", [True, False])
    def test_reshape_keeps_each_clause_in_its_own_bank(self, coalesced):
        """A wrong reshape gives a plausible ClauseInfo with clauses attributed to the wrong bank."""
        tm = MultiClassTM(6, 2.0, (4, 4), 3, feat_maxs=1, seed=1, coalesced=coalesced)
        cfg = tm.config
        info = tm.get_clauses(force_repack=True)
        flat = tm.dev.get_packed_clauses()

        assert info.clause_density.shape == (cfg._n_clause_banks, cfg._n_clauses)
        assert np.array_equal(info.clause_density.reshape(-1), flat.clause_density)
        assert np.array_equal(
            info.feature_bounds.reshape(cfg._total_clauses, -1), flat.clause_feat_bounds.reshape(cfg._total_clauses, -1)
        )

    def test_position_bounds_are_absent_without_patches(self):
        flat = MultiClassTM(6, 2.0, (4, 4), 3, feat_maxs=1, seed=1, position_literals=False)
        assert flat.config._n_patches == 1
        assert flat.get_clauses(force_repack=True).position_bounds is None

    def test_position_bounds_are_present_for_a_conv_model(self):
        conv = MultiClassTM(6, 2.0, (6, 6), 3, patch_dim=(3, 3), feat_maxs=1, seed=1)
        info = conv.get_clauses(force_repack=True)
        assert info.position_bounds is not None
        assert info.position_bounds.shape == (conv.config._n_clause_banks, conv.config._n_clauses, 4)


def test_binary_and_regression_also_round_trip():
    """The public classes differ in `_encode_Y`-equivalent handling and defaults, so each has its
    own path."""
    X = np.random.default_rng(3).integers(0, 2, size=(60, 16), dtype=np.int32)

    b = BinaryTM(32, 2.0, (4, 4), feat_maxs=1, seed=2)
    b.fit(X, (X[:, 0] == 1).astype(int))
    assert np.array_equal(pickle.loads(pickle.dumps(b)).predict(X)[0], b.predict(X)[0])

    r = RegressionTM(32, 2.0, (4, 4), feat_maxs=1, seed=3)
    r.fit(X, X[:, :8].sum(1).astype(float))
    back = pickle.loads(pickle.dumps(r))
    assert np.allclose(back.predict(X)[0], r.predict(X)[0])


def test_training_is_reproducible_across_thread_counts():
    """End to end version of the per function checks in tests/backends. Every hot loop is an omp
    parallel for, and a race would look like a different random draw rather than a failure."""
    rng = np.random.default_rng(7)
    X = rng.integers(0, 2, size=(150, 16), dtype=np.int32)
    Y = ((X[:, :8].sum(1) > X[:, 8:].sum(1)).astype(int) + (X[:, 0] == 1)) % 3

    def trained(device: str):
        tm = MultiClassTM(128, 2.0, (4, 4), 3, feat_maxs=1, seed=5, device=device)
        for _ in range(4):
            tm.fit(X, Y, clause_drop_p=0.1)
        return tm

    single = trained("cpu:1")
    many = trained("cpu:8")
    if many.device_config._n_threads == 1:
        pytest.skip("no working OpenMP flags on this machine")

    assert np.array_equal(many.get_ta_states(), single.get_ta_states())
    # count_votes sums per clause under `reduction(+:votes[:CLASSES])`; the sum order (and so the
    # float32 rounding) depends on thread count. Discrete's weight update is a clean +-1 decision,
    # insensitive to that noise; guided's `update_weights` adds `lr * grad` directly, so it inherits
    # the tiny (~1e-6) difference. Not a race - ta_states above is the actual determinism check.
    assert np.allclose(many.get_weights(), single.get_weights(), atol=1e-3, rtol=1e-3)
    assert np.array_equal(many.predict(X)[0], single.predict(X)[0])
