"""Differential tests for the inference kernels.

`eval_sample_patchwise` and `infer_sample` both decide, per clause, whether any patch matches. The
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
    def transform(self, X, batch_size, force_repack=False): ...
    def transform_patchwise(self, X, batch_size, force_repack=False): ...
    def wic(self, class_id, polarity, pw_th=0.0, force_repack=False): ...
    def wac(self, X, target_classes, polarity, force_repack=False): ...


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
        dev.lib.eval_sample_patchwise(
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
        dev.lib.infer_sample(
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
