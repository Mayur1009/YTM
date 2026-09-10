"""Differential tests for the evaluation kernels.

`calc_clause_outputs_patchwise` and `calc_class_sums` both decide, per clause, whether any patch matches. The
oracles here rebuild that from the clause definition rather than from the packed form, so a bug in
the position gating or the clause to class mapping shows up as a disagreement.
"""

import ctypes
from typing import Any

import numpy as np
import pytest

from ytm._core.backends.cpu import CPUDevice, float_p, int8_p, int32_p
from ytm._core.config import BaseTMConfig
from ytm._core.device_config import DeviceConfig


class Device(CPUDevice):
    def fit_epoch(self, X, Y, clause_drop_p, batch_size, **kwargs): ...
    def fit_sample(self, X, Y, e, **kwargs): ...
    def infer(self, X, batch_size): ...


DEFAULTS: dict[str, Any] = {
    "n_clauses": 16,
    "s": 10.0,
    "dim": (6, 6, 1),
    "n_classes": 3,
    "patch_dim": (3, 3),
    "feat_maxs": 3,
    "seed": 1,
}


def make_device(**kwargs: Any) -> Device:
    return Device(BaseTMConfig(**{**DEFAULTS, **kwargs}), DeviceConfig())


def sprinkle_includes(dev: Device, rng: np.random.Generator, p: float = 0.03) -> None:
    cfg = dev.config
    dev.ta_states[:] = cfg._include_state - 1
    dev.ta_states[rng.random(dev.ta_states.shape) < p] = cfg._include_state


def call_patchwise(dev: Device, X: np.ndarray) -> np.ndarray:
    cfg = dev.config
    X = np.ascontiguousarray(X, dtype=np.int32)
    out = np.zeros((X.shape[0], cfg._total_clauses, cfg._n_patches), dtype=np.int8)
    for e in range(X.shape[0]):
        dev.lib.calc_clause_outputs_patchwise(
            dev.p_clause_position_bounds,
            dev.p_clause_feat_bounds,
            dev.p_bounded_feat_ids,
            dev.p_n_bounded_feats,
            dev.p_clause_density,
            X.ctypes.data_as(int32_p),
            ctypes.c_int(e),
            out.ctypes.data_as(int8_p),
        )
    return out


def call_infer(dev: Device, X: np.ndarray) -> np.ndarray:
    cfg = dev.config
    X = np.ascontiguousarray(X, dtype=np.int32)
    sums = np.zeros((X.shape[0], cfg.n_classes), dtype=np.float32)
    for e in range(X.shape[0]):
        dev.lib.calc_class_sums(
            dev.p_clause_weights,
            dev.p_bias,
            dev.p_clause_position_bounds,
            dev.p_clause_feat_bounds,
            dev.p_bounded_feat_ids,
            dev.p_n_bounded_feats,
            dev.p_clause_density,
            X.ctypes.data_as(int32_p),
            ctypes.c_int(e),
            sums.ctypes.data_as(float_p),
        )
    return sums


def patch_matches_naive(dev: Device, clause: int, x: np.ndarray, py: int, px: int) -> bool:
    """Evaluate the clause on one patch, literal by literal, from the TA states."""
    cfg = dev.config
    half = cfg._n_literals // 2 if cfg.negated_literals else 0
    included = dev.ta_states[clause] >= cfg._include_state
    ny, nx = cfg._n_patches_y, cfg._n_patches_x

    if cfg.position_literals:
        for lit in range(ny - 1):
            if included[lit] and not (py > lit):
                return False
            if half and included[lit + half] and not (py <= lit):
                return False
        for lit in range(nx - 1):
            if included[(ny - 1) + lit] and not (px > lit):
                return False
            if half and included[(ny - 1) + lit + half] and not (px <= lit):
                return False

    ph, pw = cfg._patch_dim
    depth = cfg._dim[2]
    for f in range(cfg._n_raw_patch_feats):
        rel_y, rem = divmod(f, pw * depth)
        rel_x, z = divmod(rem, depth)
        val = x[py * cfg._stride[0] + rel_y, px * cfg._stride[1] + rel_x, z]

        start = cfg._n_position_feats + cfg._literal_offsets[f]
        for b in range(cfg._literal_offsets[f + 1] - cfg._literal_offsets[f]):
            if included[start + b] and not (val > cfg._feat_mins[f] + b):
                return False
            if half and included[start + b + half] and not (val <= cfg._feat_mins[f] + b):
                return False
    return True


def patchwise_naive(dev: Device, X: np.ndarray) -> np.ndarray:
    cfg = dev.config
    out = np.zeros((X.shape[0], cfg._total_clauses, cfg._n_patches_y, cfg._n_patches_x), dtype=np.int8)
    for e, x in enumerate(X):
        for clause in range(cfg._total_clauses):
            for py in range(cfg._n_patches_y):
                for px in range(cfg._n_patches_x):
                    out[e, clause, py, px] = patch_matches_naive(dev, clause, x, py, px)
    return out.reshape(X.shape[0], cfg._total_clauses, cfg._n_patches)


def class_sums_naive(dev: Device, fired: np.ndarray) -> np.ndarray:
    """Sum the weights of the clauses that fired, mapping clauses to classes as the header does."""
    cfg = dev.config
    weights = dev.clause_weights
    per_class = cfg._total_clauses if cfg.coalesced else cfg._total_clauses // cfg.n_classes

    sums = np.zeros((fired.shape[0], cfg.n_classes), dtype=np.float64)
    if cfg.bias:
        sums += dev.bias
    for e in range(fired.shape[0]):
        for clause in np.flatnonzero(fired[e]):
            rel = int(clause) % per_class
            classes = range(cfg.n_classes) if cfg.coalesced else [int(clause) // per_class]
            for c in classes:
                sums[e, c] += weights[c, rel]
    return sums


@pytest.fixture
def setup():
    def _setup(trial: int = 0, p: float = 0.03, **kwargs):
        dev = make_device(**kwargs)
        sprinkle_includes(dev, np.random.default_rng(trial), p)
        dev.pack_clauses(force_repack=True)
        X = np.random.default_rng(100 + trial).integers(0, 4, size=(4, *dev.config._dim), dtype=np.int32)
        return dev, X

    return _setup


class TestPatchwise:
    @pytest.mark.parametrize("trial", range(3))
    def test_matches_literal_evaluation_on_every_patch(self, setup, trial):
        dev, X = setup(trial)
        got = call_patchwise(dev, X)
        expected = patchwise_naive(dev, X)

        assert np.array_equal(got, expected)
        assert 0 < got.mean() < 1, "every patch agreed trivially, the trial proved nothing"

    def test_contradictory_clauses_never_fire(self, setup):
        dev, X = setup(0, p=0.3)
        got = call_patchwise(dev, X)
        invalid = np.flatnonzero(dev.get_packed_clauses().clause_density < 0)

        assert len(invalid) > 0
        assert not got[:, invalid, :].any()

    def test_unconstrained_clauses_fire_everywhere(self, setup):
        dev, X = setup(0)
        dev.ta_states[0] = dev.config._include_state - 1
        dev.pack_clauses(force_repack=True)

        assert call_patchwise(dev, X)[:, 0, :].all()

    def test_position_bounds_gate_the_patches(self):
        """A clause restricted to patch_y > 1 must be silent on the rows below."""
        dev = make_device()
        cfg = dev.config
        dev.ta_states[:] = cfg._include_state - 1
        dev.ta_states[0, 1] = cfg._include_state
        dev.pack_clauses(force_repack=True)

        X = np.zeros((1, *cfg._dim), dtype=np.int32)
        got = call_patchwise(dev, X).reshape(1, cfg._total_clauses, cfg._n_patches_y, cfg._n_patches_x)

        assert not got[0, 0, :2].any()
        assert got[0, 0, 2:].all()


class TestInferSample:
    @pytest.mark.parametrize("trial", range(3))
    @pytest.mark.parametrize("coalesced", [True, False])
    def test_class_sums_match_the_weights_of_the_clauses_that_fired(self, setup, trial, coalesced):
        dev, X = setup(trial, coalesced=coalesced)
        fired = call_patchwise(dev, X).any(axis=2)
        got = call_infer(dev, X)

        assert np.allclose(got, class_sums_naive(dev, fired), atol=1e-4)
        assert fired.any() and not fired.all(), "clauses fired uniformly, the trial proved nothing"

    def test_bias_is_the_starting_point(self, setup):
        dev, X = setup(0, bias=True, bias_init=2.0)
        no_clauses_fire = np.flatnonzero(dev.get_packed_clauses().clause_density < 0)
        if len(no_clauses_fire) == 0:
            pytest.skip("needs at least one silent clause")

        got = call_infer(dev, X)
        assert np.allclose(got, class_sums_naive(dev, call_patchwise(dev, X).any(axis=2)), atol=1e-4)

    def test_silent_model_returns_zero(self, setup):
        dev, X = setup(0)
        dev.ta_states[:] = dev.config._include_state
        dev.pack_clauses(force_repack=True)

        assert np.all(dev.get_packed_clauses().clause_density < 0)
        assert np.array_equal(call_infer(dev, X), np.zeros((X.shape[0], dev.config.n_classes), dtype=np.float32))

    def test_non_coalesced_clauses_only_vote_for_their_own_class(self):
        dev = make_device(coalesced=False, n_clauses=4)
        cfg = dev.config
        dev.ta_states[:] = cfg._include_state
        dev.ta_states[0] = cfg._include_state - 1  # clause 0 alone fires, and it belongs to class 0
        dev.pack_clauses(force_repack=True)

        X = np.zeros((1, *cfg._dim), dtype=np.int32)
        sums = call_infer(dev, X)[0]

        assert sums[0] == pytest.approx(dev.clause_weights[0, 0])
        assert sums[1] == 0.0
        assert sums[2] == 0.0


class TestPythonWrappers:
    """The C is covered above, these check what the wrappers add: batching, reshaping, arguments."""

    def test_class_sums_match_the_per_sample_calls(self, setup):
        dev, X = setup(0)
        assert np.allclose(dev.calc_class_sums(X), call_infer(dev, X))

    def test_class_sums_do_not_accumulate_across_calls(self, setup):
        """The buffer is fresh each call, a stale one would double the votes."""
        dev, X = setup(0)
        first = dev.calc_class_sums(X).copy()  # copy, a reused buffer would compare against itself
        assert np.allclose(dev.calc_class_sums(X), first)

    def test_transform_patchwise_reshape_preserves_order(self, setup):
        """Non square patch grid, so transposing the two patch axes is visible."""
        dev, X = setup(0, dim=(6, 9, 1))
        cfg = dev.config
        assert cfg._n_patches_y != cfg._n_patches_x
        got = dev.transform_patchwise(X, -1)

        assert got.shape == (X.shape[0], cfg._n_clause_banks, cfg._n_clauses, cfg._n_patches_y, cfg._n_patches_x)
        assert np.array_equal(got.reshape(X.shape[0], cfg._total_clauses, cfg._n_patches), call_patchwise(dev, X))

    def test_transform_patchwise_indexes_patches_row_major(self, setup):
        """A clause gated to patch_y > 1 must be silent on the first two rows after reshaping."""
        dev, _ = setup(0)
        cfg = dev.config
        dev.ta_states[:] = cfg._include_state - 1
        dev.ta_states[0, 1] = cfg._include_state
        dev.pack_clauses(force_repack=True)

        got = dev.transform_patchwise(np.zeros((1, *cfg._dim), dtype=np.int32), -1)
        assert not got[0, 0, 0, :2].any()
        assert got[0, 0, 0, 2:].all()

    def test_wic_does_not_mutate_the_patch_weights(self, setup):
        dev, _ = setup(0)
        dev.patch_weights[:] = np.arange(dev.patch_weights.size).reshape(dev.patch_weights.shape)
        before = dev.patch_weights.copy()

        dev.wic(0, 1)
        assert np.array_equal(dev.patch_weights, before)

    def test_wic_responds_to_the_patch_weights(self, setup):
        """All zero patch weights normalise to zero, so nothing passes the threshold."""
        dev, _ = setup(0)
        dev.patch_weights[:] = 0
        assert dev.wic(0, 1).sum() == 0.0

        dev.patch_weights[:] = 1
        assert dev.wic(0, 1).sum() != 0.0

    def test_wac_uses_the_per_sample_target_class(self, setup):
        """Swapping a sample's target must change only that sample's attribution."""
        dev, X = setup(0, coalesced=False)
        a = dev.wac(X, np.array([0, 0, 0, 0]), 1)
        b = dev.wac(X, np.array([0, 1, 0, 0]), 1)

        assert np.array_equal(a[0], b[0])
        assert np.array_equal(a[2:], b[2:])
        assert not np.array_equal(a[1], b[1])


    def test_wic_responds_to_the_patch_weights(self, setup):
        """All zero patch weights normalise to zero, so nothing passes the threshold."""
        dev, _ = setup(0)
        dev.patch_weights[:] = 0
        assert dev.wic(0, 1).sum() == 0.0

        dev.patch_weights[:] = 1
        assert dev.wic(0, 1).sum() != 0.0

    def test_wac_uses_the_per_sample_target_class(self, setup):
        """Swapping a sample's target must change only that sample's attribution."""
        dev, X = setup(0, coalesced=False)
        a = dev.wac(X, np.array([0, 0, 0, 0]), 1)
        b = dev.wac(X, np.array([0, 1, 0, 0]), 1)

        assert np.array_equal(a[0], b[0])
        assert np.array_equal(a[2:], b[2:])
        assert not np.array_equal(a[1], b[1])


class TestTransform:
    """`calc_clause_outputs` answers the same question as the patchwise scan, without the grid."""

    @pytest.mark.parametrize("trial", range(3))
    def test_agrees_with_the_patchwise_scan(self, setup, trial):
        dev, X = setup(trial)
        got = dev.transform(X, -1)

        expected = call_patchwise(dev, X).any(axis=2).astype(np.int8)
        assert np.array_equal(got.reshape(X.shape[0], dev.config._total_clauses), expected)
        assert 0 < got.mean() < 1, "every clause agreed trivially, the trial proved nothing"

    def test_shape_and_dtype(self, setup):
        dev, X = setup(0, coalesced=False)
        cfg = dev.config
        got = dev.transform(X, -1)

        assert got.shape == (X.shape[0], cfg._n_clause_banks, cfg._n_clauses)
        assert cfg._n_clause_banks > 1
        assert got.dtype == np.int8

    def test_contradictory_clauses_never_fire(self, setup):
        dev, X = setup(0, p=0.3)
        invalid = np.flatnonzero(dev.get_packed_clauses().clause_density < 0)
        assert len(invalid) > 0

        got = dev.transform(X, -1).reshape(X.shape[0], dev.config._total_clauses)
        assert not got[:, invalid].any()

    def test_position_bounds_gate_the_result(self):
        """A clause allowed only on patch_y > 1 must still fire when such a patch matches."""
        dev = make_device()
        cfg = dev.config
        dev.ta_states[:] = cfg._include_state - 1
        dev.ta_states[0, 1] = cfg._include_state
        dev.pack_clauses(force_repack=True)

        got = dev.transform(np.zeros((1, *cfg._dim), dtype=np.int32), -1)
        assert got[0, 0, 0] == 1

        # narrow it to a row that does not exist, and it must go silent
        dev.ta_states[0, cfg._n_patches_y - 2] = cfg._include_state
        dev.ta_states[0, 0 + cfg._n_literals // 2] = cfg._include_state
        dev.pack_clauses(force_repack=True)
        assert dev.transform(np.zeros((1, *cfg._dim), dtype=np.int32), -1)[0, 0, 0] == 0

    def test_does_not_touch_the_patch_weights(self, setup):
        """Going through `evaluate` would increment them as a side effect."""
        dev, X = setup(0)
        before = dev.patch_weights.copy()

        dev.transform(X, -1)
        assert np.array_equal(dev.patch_weights, before)
