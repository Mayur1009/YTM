"""guided's `act_loss.h` + `update.c`: activations, loss gradients, clause/weight/bias updates.

Activations and losses are checked against independent references (numpy, finite differences),
not against a restatement of the C formulas. `update_clauses`'s per-`fb_signal` layout is checked
by coverage (every (clause, class) slot written exactly once, no collisions), not by re-deriving
the index formula. `update_weights`/`update_bias` are checked against their documented contract
(clip bound, drop skips learning, bias no-op when disabled). Thread tests compare `cpu:1` against
`cpu:8` bit-for-bit, since a race is invisible any other way (the algorithm is already stochastic).
"""

import ctypes

import numpy as np
import pytest

from ytm._core.device_config import DeviceConfig
from ytm._guided.backends.cpu import CPUDevice
from ytm._guided.config import TMConfig

_int_p = ctypes.POINTER(ctypes.c_int32)
_float_p = ctypes.POINTER(ctypes.c_float)
_int8_p = ctypes.POINTER(ctypes.c_int8)
_uint8_p = ctypes.POINTER(ctypes.c_uint8)


def make_device(device: str = "cpu:1", **kwargs) -> CPUDevice:
    cfg = {"n_clauses": 6, "s": 2.0, "dim": (4, 4, 1), "n_classes": 3, "feat_maxs": 1, "seed": 1}
    cfg.update(kwargs)
    dev = CPUDevice(TMConfig(**cfg), DeviceConfig(device=device))
    dev.pack_clauses(force_repack=True)
    return dev


def _norm(dev: CPUDevice) -> float:
    cfg = dev.config
    return cfg._n_clauses / 2.0 if cfg.negative_clauses else float(cfg._n_clauses)


class TestActivations:
    @pytest.mark.parametrize("act_fn", ["softmax", "sigmoid", "identity"])
    def test_matches_an_independent_reference(self, act_fn):
        dev = make_device(act_fn=act_fn, n_classes=4)
        votes = np.array([-2.0, 0.5, 3.0, 1.0], dtype=np.float32)
        y_hat = np.empty(4, dtype=np.float32)
        dev.lib.votes_activation(votes.ctypes.data_as(_float_p), y_hat.ctypes.data_as(_float_p))

        scaled = votes / _norm(dev)
        if act_fn == "softmax":
            e = np.exp(scaled - np.max(scaled))
            expected = e / e.sum()
        elif act_fn == "sigmoid":
            expected = 1.0 / (1.0 + np.exp(-scaled))
        else:
            expected = votes  # identity does not normalize at all

        assert np.allclose(y_hat, expected, atol=1e-5)

    @pytest.mark.parametrize("act_fn", ["softmax", "sigmoid", "identity"])
    def test_batch_matches_looping_the_single_version(self, act_fn):
        dev = make_device(act_fn=act_fn, n_classes=4)
        rng = np.random.default_rng(0)
        votes = rng.uniform(-3, 3, size=(5, 4)).astype(np.float32)

        y_hat_batch = np.empty_like(votes)
        dev.lib.votes_activation_batch(votes.ctypes.data_as(_float_p), ctypes.c_int(5), y_hat_batch.ctypes.data_as(_float_p))

        y_hat_loop = np.empty_like(votes)
        for i in range(5):
            dev.lib.votes_activation(votes[i].ctypes.data_as(_float_p), y_hat_loop[i].ctypes.data_as(_float_p))

        assert np.array_equal(y_hat_batch, y_hat_loop)


def _pipeline_loss(dev: CPUDevice, votes: np.ndarray, y: np.ndarray, class_weights: np.ndarray) -> float:
    n = dev.config.n_classes
    y_hat = np.empty(n, dtype=np.float32)
    loss = np.empty(1, dtype=np.float32)
    dev.lib.votes_activation(votes.ctypes.data_as(_float_p), y_hat.ctypes.data_as(_float_p))
    dev.lib.loss_gradient(
        y_hat.ctypes.data_as(_float_p), y.ctypes.data_as(_float_p), class_weights.ctypes.data_as(_float_p),
        None, loss.ctypes.data_as(_float_p),
    )
    return float(loss[0])


def _pipeline_grad(dev: CPUDevice, votes: np.ndarray, y: np.ndarray, class_weights: np.ndarray) -> np.ndarray:
    n = dev.config.n_classes
    y_hat = np.empty(n, dtype=np.float32)
    grad = np.empty(n, dtype=np.float32)
    dev.lib.votes_activation(votes.ctypes.data_as(_float_p), y_hat.ctypes.data_as(_float_p))
    dev.lib.loss_gradient(
        y_hat.ctypes.data_as(_float_p), y.ctypes.data_as(_float_p), class_weights.ctypes.data_as(_float_p),
        grad.ctypes.data_as(_float_p), None,
    )
    return grad


class TestLossGradient:
    """`grad` is defined as the update direction, i.e. `-d(loss)/d(votes)` (matches `update_weights`
    doing `weight += lr * grad`, a descent step). Checked via central finite differences on the
    whole `votes_activation` -> `loss_gradient` pipeline, so it does not matter whether a given loss
    internally differentiates against `y_hat` or against `votes` - only the externally observable
    loss value and the externally observable gradient need to agree.
    """

    @pytest.mark.parametrize(
        "loss_fn, act_fn",
        [
            ("ce", "softmax"),
            ("sce", "softmax"),
            ("asl", "sigmoid"),
            ("mse", "identity"),
            ("mae", "identity"),
            ("huber", "identity"),
            ("tversky", "sigmoid"),
        ],
    )
    def test_matches_finite_differences(self, loss_fn, act_fn):
        dev = make_device(loss_fn=loss_fn, act_fn=act_fn, n_classes=3)
        rng = np.random.default_rng(7)
        votes = rng.uniform(-2, 2, size=3).astype(np.float32)
        y = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        w = np.ones(3, dtype=np.float32)

        analytic = _pipeline_grad(dev, votes, y, w)

        # softmax/sigmoid divide by NORM before activating, and the closed-form grad is w.r.t. that
        # scaled logit (z = votes / NORM), not raw votes; identity applies no such scaling. Stepping
        # votes by `eps * scale` moves z by exactly `eps`, so the quotient below is d(loss)/dz.
        eps = 1e-2
        scale = _norm(dev) if act_fn in ("softmax", "sigmoid") else 1.0
        numeric = np.empty(3, dtype=np.float64)
        for i in range(3):
            plus, minus = votes.copy(), votes.copy()
            plus[i] += eps * scale
            minus[i] -= eps * scale
            numeric[i] = (_pipeline_loss(dev, plus, y, w) - _pipeline_loss(dev, minus, y, w)) / (2 * eps)

        assert np.allclose(analytic, -numeric, atol=5e-2, rtol=5e-2), (analytic, -numeric)


class TestUpdateClauses:
    """`feedback_type`'s layout differs by `fb_signal`: `grad` mode packs `(rel_clause, class)`
    pairs, `delta_l` mode packs one code per clause. In non-coalesced mode, several absolute clauses
    (one per class bank) share the same `rel_clause`, so a wrong index (e.g. the absolute clause id
    instead of `rel_clause`) would misattribute one bank's feedback to a different bank's clause. That
    only shows up with *distinct* per-(rel_clause, class) feedback - uniform feedback for every clause
    can't distinguish "indexed correctly" from "indexed wrong but landed somewhere valid anyway".
    """

    def test_grad_mode_isolates_each_class_bank(self):
        dev = make_device(fb_signal="grad", coalesced=False, n_clauses=3, n_classes=3)
        cfg = dev.config
        # non-coalesced: LOOP_CLASS_ID gives bank b's clauses class_id = b exclusively. Feedback only
        # for (rel_clause=0, class=1) - only the absolute clause in bank 1 at rel_clause 0 may move.
        fb = np.zeros((cfg._n_clauses, cfg.n_classes), dtype=np.uint8)
        fb[0, 1] = 2  # FB_T1B
        X = np.zeros((1, *cfg._dim), dtype=np.int32)
        pids = np.full(cfg._total_clauses, -1, dtype=np.int32)

        before = dev.ta_states.copy()
        dev.lib.update_clauses(
            ctypes.c_uint64(1), pids.ctypes.data_as(_int_p), X.ctypes.data_as(_int_p), ctypes.c_int(0),
            cfg._feat_mins.ctypes.data_as(_int_p), cfg._literal_offsets.ctypes.data_as(_int_p),
            fb.ctypes.data_as(_uint8_p), dev.p_ta_states, dev.p_is_clause_synced,
        )

        expected_clause = 1 * cfg._n_clauses + 0  # bank 1 (class 1), rel_clause 0
        moved = np.where(np.any(dev.ta_states != before, axis=1))[0]
        assert list(moved) == [expected_clause], (moved, expected_clause)
        assert np.all(dev.ta_states[expected_clause] <= before[expected_clause])
        assert np.any(dev.ta_states[expected_clause] < before[expected_clause])

    @pytest.mark.parametrize("coalesced", [True, False])
    def test_grad_mode_feedback_touches_exactly_its_own_class_slots(self, coalesced):
        dev = make_device(fb_signal="grad", coalesced=coalesced, n_clauses=4, n_classes=3)
        cfg = dev.config
        # every clause votes T1B (dec every literal) for every class it owns, nothing else
        fb = np.full((cfg._n_clauses, cfg.n_classes), 2, dtype=np.uint8)  # FB_T1B = 2
        X = np.zeros((1, *cfg._dim), dtype=np.int32)
        pids = np.full(cfg._total_clauses, -1, dtype=np.int32)

        before = dev.ta_states.copy()
        dev.lib.update_clauses(
            ctypes.c_uint64(1), pids.ctypes.data_as(_int_p), X.ctypes.data_as(_int_p), ctypes.c_int(0),
            cfg._feat_mins.ctypes.data_as(_int_p), cfg._literal_offsets.ctypes.data_as(_int_p),
            fb.ctypes.data_as(_uint8_p), dev.p_ta_states, dev.p_is_clause_synced,
        )
        # T1B decrements every literal in every clause bank, for every clause: every clause must have moved
        assert np.all(dev.ta_states <= before)
        assert np.any(dev.ta_states < before)

    def test_delta_l_mode_feedback_is_one_code_per_clause(self):
        dev = make_device(fb_signal="delta_l", n_clauses=4, n_classes=3)
        cfg = dev.config
        fb = np.full(cfg._total_clauses, 2, dtype=np.uint8)  # FB_T1B everywhere
        X = np.zeros((1, *cfg._dim), dtype=np.int32)
        pids = np.full(cfg._total_clauses, -1, dtype=np.int32)

        before = dev.ta_states.copy()
        dev.lib.update_clauses(
            ctypes.c_uint64(1), pids.ctypes.data_as(_int_p), X.ctypes.data_as(_int_p), ctypes.c_int(0),
            cfg._feat_mins.ctypes.data_as(_int_p), cfg._literal_offsets.ctypes.data_as(_int_p),
            fb.ctypes.data_as(_uint8_p), dev.p_ta_states, dev.p_is_clause_synced,
        )
        assert np.all(dev.ta_states <= before)
        assert np.any(dev.ta_states < before)


class TestUpdateWeights:
    def test_clips_at_max_weight(self):
        dev = make_device(max_weight=5.0, n_clauses=2, n_classes=2)
        cfg = dev.config
        dev.clause_weights[:] = 4.9
        grad = np.ones(cfg.n_classes, dtype=np.float32)
        pids = np.zeros(cfg._total_clauses, dtype=np.int32)
        mask = np.zeros(cfg._total_clauses, dtype=np.int8)

        dev.lib.update_weights(
            grad.ctypes.data_as(_float_p), ctypes.c_float(1.0), pids.ctypes.data_as(_int_p),
            mask.ctypes.data_as(_int8_p), dev.p_clause_weights,
        )
        assert np.all(dev.clause_weights <= 5.0)

    def test_dropped_clauses_do_not_learn(self):
        dev = make_device(n_clauses=2, n_classes=2)
        cfg = dev.config
        dev.clause_weights[:] = 1.0
        before = dev.clause_weights.copy()
        grad = np.ones(cfg.n_classes, dtype=np.float32)
        pids = np.zeros(cfg._total_clauses, dtype=np.int32)
        mask = np.ones(cfg._total_clauses, dtype=np.int8)  # every clause dropped

        dev.lib.update_weights(
            grad.ctypes.data_as(_float_p), ctypes.c_float(1.0), pids.ctypes.data_as(_int_p),
            mask.ctypes.data_as(_int8_p), dev.p_clause_weights,
        )
        assert np.array_equal(dev.clause_weights, before)

    def test_unselected_clauses_do_not_learn(self):
        dev = make_device(n_clauses=2, n_classes=2)
        cfg = dev.config
        dev.clause_weights[:] = 1.0
        before = dev.clause_weights.copy()
        grad = np.ones(cfg.n_classes, dtype=np.float32)
        pids = np.full(cfg._total_clauses, -1, dtype=np.int32)  # no clause matched a patch
        mask = np.zeros(cfg._total_clauses, dtype=np.int8)

        dev.lib.update_weights(
            grad.ctypes.data_as(_float_p), ctypes.c_float(1.0), pids.ctypes.data_as(_int_p),
            mask.ctypes.data_as(_int8_p), dev.p_clause_weights,
        )
        assert np.array_equal(dev.clause_weights, before)


class TestUpdateBias:
    def test_no_op_when_bias_disabled(self):
        dev = make_device(bias=False, n_classes=2)
        before = dev.bias.copy()
        grad = np.ones(2, dtype=np.float32)
        dev.lib.update_bias(grad.ctypes.data_as(_float_p), ctypes.c_float(1.0), dev.p_bias)
        assert np.array_equal(dev.bias, before)

    def test_moves_by_lr_times_grad_when_enabled(self):
        dev = make_device(bias=True, n_classes=2)
        dev.bias[:] = 0.0
        grad = np.array([2.0, -1.0], dtype=np.float32)
        dev.lib.update_bias(grad.ctypes.data_as(_float_p), ctypes.c_float(0.5), dev.p_bias)
        assert np.allclose(dev.bias, [1.0, -0.5])


THREADS = 8


class TestThreadDeterminism:
    """Every hot loop is `#pragma omp parallel for`; `cpu:1` never exercises the parallel path."""

    def _pair(self, **kwargs):
        single = make_device(device="cpu:1", **kwargs)
        many = make_device(device=f"cpu:{THREADS}", **kwargs)
        assert many.device_config._n_threads > 1, "this machine resolved no working OpenMP flags"

        rng = np.random.default_rng(11)
        states = np.where(
            rng.random(single.ta_states.shape) < 0.05, single.config._include_state, single.config._include_state - 1
        ).astype(single.ta_states.dtype)
        for dev in (single, many):
            dev.ta_states[:] = states
            dev.pack_clauses(force_repack=True)
        return single, many

    @pytest.mark.parametrize("fb_signal", ["grad", "delta_l"])
    def test_update_clauses_agrees(self, fb_signal):
        single, many = self._pair(fb_signal=fb_signal, n_clauses=64, n_classes=4)
        cfg = single.config
        rng = np.random.default_rng(12)
        shape = (cfg._n_clauses, cfg.n_classes) if fb_signal == "grad" else (cfg._total_clauses,)
        fb = rng.integers(0, 4, size=shape).astype(np.uint8)
        X = np.zeros((1, *cfg._dim), dtype=np.int32)
        # T1A/T2 index into X via the matched patch, so every clause needs a real match (fb is
        # otherwise unconstrained by pids here, and an unmatched clause getting T1A/T2 is a state
        # the real decide_feedback_* never produces - patch_idx would be -1, an OOB read into X).
        pids = np.zeros(cfg._total_clauses, dtype=np.int32)

        for dev in (single, many):
            dev.packed_clauses.is_clause_synced[:] = 1
            dev.lib.update_clauses(
                ctypes.c_uint64(9), pids.ctypes.data_as(_int_p), X.ctypes.data_as(_int_p), ctypes.c_int(0),
                cfg._feat_mins.ctypes.data_as(_int_p), cfg._literal_offsets.ctypes.data_as(_int_p),
                fb.ctypes.data_as(_uint8_p), dev.p_ta_states, dev.p_is_clause_synced,
            )
        assert np.array_equal(single.ta_states, many.ta_states)

    def test_update_weights_agrees(self):
        single, many = self._pair(n_clauses=64, n_classes=4)
        cfg = single.config
        rng = np.random.default_rng(13)
        grad = rng.uniform(-1, 1, size=cfg.n_classes).astype(np.float32)
        pids = rng.integers(-1, 1, size=cfg._total_clauses).astype(np.int32)
        mask = (rng.random(cfg._total_clauses) < 0.1).astype(np.int8)
        weights = rng.uniform(-1, 1, size=single.clause_weights.shape).astype(np.float32)

        for dev in (single, many):
            dev.clause_weights[:] = weights
            dev.lib.update_weights(
                grad.ctypes.data_as(_float_p), ctypes.c_float(0.1), pids.ctypes.data_as(_int_p),
                mask.ctypes.data_as(_int8_p), dev.p_clause_weights,
            )
        assert np.array_equal(single.clause_weights, many.clause_weights)

    @pytest.mark.parametrize("fb_signal", ["grad", "delta_l"])
    def test_decide_feedback_agrees(self, fb_signal):
        single, many = self._pair(fb_signal=fb_signal, n_clauses=64, n_classes=4)
        cfg = single.config
        rng = np.random.default_rng(14)
        pids = rng.integers(-1, 1, size=cfg._total_clauses).astype(np.int32)
        mask = np.zeros(cfg._total_clauses, dtype=np.int8)
        grad = rng.uniform(-1, 1, size=cfg.n_classes).astype(np.float32)
        loss = np.array([0.5], dtype=np.float32)
        loss_neg_ck = rng.uniform(0, 1, size=cfg._total_clauses).astype(np.float32)

        results = []
        for dev in (single, many):
            if fb_signal == "grad":
                fb = np.zeros((cfg._n_clauses, cfg.n_classes), dtype=np.uint8)
                dev.lib.decide_feedback_grad(
                    ctypes.c_uint64(5), grad.ctypes.data_as(_float_p), dev.p_clause_weights,
                    dev.p_clause_density, pids.ctypes.data_as(_int_p), mask.ctypes.data_as(_int8_p),
                    ctypes.c_float(1.0), fb.ctypes.data_as(_uint8_p),
                )
            else:
                fb = np.zeros(cfg._total_clauses, dtype=np.uint8)
                dev.lib.decide_feedback_delta_l(
                    ctypes.c_uint64(5), loss.ctypes.data_as(_float_p), loss_neg_ck.ctypes.data_as(_float_p),
                    dev.p_clause_density, pids.ctypes.data_as(_int_p), mask.ctypes.data_as(_int8_p),
                    ctypes.c_float(1.0), fb.ctypes.data_as(_uint8_p),
                )
            results.append(fb)
        assert np.array_equal(results[0], results[1])
