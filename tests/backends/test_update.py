"""The paths a training run on real data never executes.

Everything else in `update.c` shows up as degraded accuracy and is better caught end to end. These
three are config combinations a smoke run does not reach: `clause_drop_p` defaults to 0,
`coalesced` defaults to True, and `label_sampling` defaults to off.
"""

import ctypes

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


def make_device(**kwargs) -> CPUDevice:
    cfg = {"n_clauses": 6, "s": 10.0, "dim": (4, 4, 1), "n_classes": 3, "T": 50.0, "feat_maxs": 1, "seed": 1}
    cfg.update(kwargs)
    dev = CPUDevice(TMConfig(**cfg), DeviceConfig())
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
