"""The paths a training run on real data never executes.

Everything else in `update.c` shows up as degraded accuracy and is better caught end to end. These
three are config combinations a smoke run does not reach: `clause_drop_p` defaults to 0,
`coalesced` defaults to True, and `label_sampling` defaults to off.
"""

import ctypes
from typing import ClassVar

import numpy as np
import pytest

from ytm._core.device_config import DeviceConfig
from ytm._core.utils import Feedback
from ytm._discrete.backends.cpu import CPUDevice
from ytm._discrete.config import TMConfig

_int_p = ctypes.POINTER(ctypes.c_int32)
_float_p = ctypes.POINTER(ctypes.c_float)
_int8_p = ctypes.POINTER(ctypes.c_int8)
_uint8_p = ctypes.POINTER(ctypes.c_uint8)

UNWRITTEN = 99  # not a valid Feedback value, so an untouched slot is visible


def make_device(device: str = "cpu:1", **kwargs) -> CPUDevice:
    cfg = {"n_clauses": 6, "s": 10.0, "dim": (4, 4, 1), "n_classes": 3, "T": 50.0, "feat_maxs": 1, "seed": 1}
    cfg.update(kwargs)
    dev = CPUDevice(TMConfig(**cfg), DeviceConfig(device=device))
    dev.pack_clauses(force_repack=True)
    return dev


def decide(dev: CPUDevice, prob: np.ndarray, label_probs: np.ndarray, drop: np.ndarray | None = None) -> np.ndarray:
    """Run `decide_feedback` with every clause firing, and return the raw feedback buffer."""
    cfg = dev.config
    fb = np.full((cfg._n_clauses, cfg.n_classes), UNWRITTEN, dtype=np.uint8)
    drop = np.zeros(cfg._total_clauses, dtype=np.int8) if drop is None else drop

    dev.lib.decide_feedback(
        ctypes.c_uint64(7),
        np.zeros(cfg._total_clauses, dtype=np.int32).ctypes.data_as(_int_p),  # every clause matched patch 0
        dev.packed_clauses.clause_density.ctypes.data_as(_int_p),
        drop.ctypes.data_as(_int8_p),
        np.ascontiguousarray(prob, dtype=np.float32).ctypes.data_as(_float_p),
        np.ascontiguousarray(label_probs, dtype=np.float32).ctypes.data_as(_float_p),
        ctypes.c_int(0),
        np.abs(dev.clause_weights).ctypes.data_as(_float_p),
        fb.ctypes.data_as(_uint8_p),
    )
    return fb


def certain(dev: CPUDevice) -> tuple[np.ndarray, np.ndarray]:
    """`prob` and `label_probs` that make both draws pass, so the decision is deterministic."""
    n = dev.config.n_classes
    return np.ones(n, dtype=np.float32), np.ones((1, n), dtype=np.float32)


def test_dropped_clauses_get_no_feedback():
    """`clause_drop_p` is 0 by default, so a training run never sets the mask. If it were ignored,
    dropped clauses would keep learning and the model would still look fine."""
    dev = make_device()
    prob, label_probs = certain(dev)

    live = decide(dev, prob, label_probs)
    assert (live == Feedback.T1A).any(), "the fixture must produce feedback when nothing is dropped"

    dropped = decide(dev, prob, label_probs, drop=np.ones(dev.config._total_clauses, dtype=np.int8))
    assert np.all(dropped == Feedback.NONE)


def test_a_zero_label_probability_blocks_that_class():
    """Only reachable through `label_sampling=True`, which is off by default."""
    dev = make_device()
    prob, label_probs = certain(dev)
    label_probs[0, 1] = 0.0  # class 1 is silenced

    fb = decide(dev, prob, label_probs)
    assert np.all(fb[:, 1] == Feedback.NONE)
    assert (fb[:, [0, 2]] != Feedback.NONE).any(), "the other classes must still act"


@pytest.mark.parametrize("coalesced", [True, False])
def test_every_feedback_slot_is_written_exactly_once(coalesced):
    """The buffer is `[rel_clause, class]`. Coalesced makes `rel_clause == clause` so the layout is
    trivially right; non coalesced is where two clauses could collide or a slot be left unwritten."""
    dev = make_device(coalesced=coalesced, n_clauses=6, n_classes=3)
    cfg = dev.config
    prob, label_probs = certain(dev)

    fb = decide(dev, prob, label_probs)

    assert fb.shape == (cfg._n_clauses, cfg.n_classes)
    assert not (fb == UNWRITTEN).any(), "a slot was allocated but never written"

    # every clause owns a distinct set of slots, so the total written equals the work done
    per_clause = cfg.n_classes if coalesced else 1
    assert fb.size == cfg._total_clauses * per_clause


class TestThreadDeterminism:
    """All four entry points parallelise over clauses. The default device is one thread, so without
    this the parallel versions of these loops are never executed at all."""

    BIG: ClassVar[dict] = {"n_clauses": 512, "n_classes": 6, "dim": (8, 8), "patch_dim": (3, 3), "feat_maxs": 3}

    def _pair(self):
        single = make_device(device="cpu:1", **self.BIG)
        many = make_device(device="cpu:8", **self.BIG)
        assert many.device_config._n_threads > 1, "this machine resolved no working OpenMP flags"

        rng = np.random.default_rng(41)
        states = np.where(
            rng.random(single.ta_states.shape) < 0.04, single.config._include_state, single.config._include_state - 1
        ).astype(np.uint32)
        for dev in (single, many):
            dev.ta_states[:] = states
            dev.pack_clauses(force_repack=True)
        return single, many

    def test_decide_feedback_agrees(self):
        single, many = self._pair()
        n = single.config.n_classes
        prob = np.linspace(-1.0, 1.0, n, dtype=np.float32)
        label_probs = np.full((1, n), 0.7, dtype=np.float32)

        expected = decide(single, prob, label_probs)
        for _ in range(5):
            assert np.array_equal(decide(many, prob, label_probs), expected)

    def test_update_weights_agrees(self):
        single, many = self._pair()
        cfg = single.config
        fb = np.random.default_rng(42).integers(0, 4, size=(cfg._n_clauses, cfg.n_classes)).astype(np.uint8)

        for dev in (single, many):
            dev.clause_weights[:] = np.arange(dev.clause_weights.size, dtype=np.float32).reshape(
                dev.clause_weights.shape
            ) % 7 - 3
            dev.lib.update_weights(fb.ctypes.data_as(_uint8_p), dev.clause_weights.ctypes.data_as(_float_p))

        assert np.array_equal(many.clause_weights, single.clause_weights)

    def test_update_clauses_agrees(self):
        """Several classes write the same clause's TA states in sequence, so the loop body is not
        a pure per index write the way the others are."""
        single, many = self._pair()
        cfg = single.config
        fb = np.random.default_rng(43).integers(0, 4, size=(cfg._n_clauses, cfg.n_classes)).astype(np.uint8)
        X = np.zeros((1, *cfg._dim), dtype=np.int32)
        pids = np.zeros(cfg._total_clauses, dtype=np.int32)

        for dev in (single, many):
            dev.packed_clauses.is_clause_synced[:] = 1
            dev.lib.update_clauses(
                ctypes.c_uint64(5),
                pids.ctypes.data_as(_int_p),
                X.ctypes.data_as(_int_p),
                ctypes.c_int(0),
                dev.config._feat_mins.ctypes.data_as(_int_p),
                dev.config._literal_offsets.ctypes.data_as(_int_p),
                fb.ctypes.data_as(_uint8_p),
                dev.p_ta_states,
                dev.p_is_clause_synced,
            )

        assert np.array_equal(many.ta_states, single.ta_states)
        assert np.array_equal(many.packed_clauses.is_clause_synced, single.packed_clauses.is_clause_synced)
