import abc
import types
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from tqdm import tqdm

from ..args import TMArgs


@dataclass
class PackedClauses:
    clause_position_bounds: Any
    clause_feat_bounds: Any
    bounded_feat_ids: Any
    n_bounded_feats: Any
    clause_density: Any
    is_clause_synced: Any


def tqdm_bar(iter, **kwargs):
    args = {
        "leave": False,
        "dynamic_ncols": True,
        "bar_format": "{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
    }
    for k, v in kwargs.items():
        args[k] = v
    return tqdm(iter, **args)


class BaseDevice(abc.ABC):
    xp: types.ModuleType
    _softmax: Callable
    _expit: Callable
    _log_softmax: Callable

    def __init__(self, args: TMArgs):
        self.args = args

        self.n_clause_banks = 1 if self.args.coalesced else self.args.n_classes
        self.total_clauses = self.n_clause_banks * self.args.n_clauses

        # The number of raw features in a patch
        self.n_raw_patch_feats = self.args.patch_dim[0] * self.args.patch_dim[1] * self.args.dim[2]

        self.n_patches_y = ((self.args.dim[0] - self.args.patch_dim[0]) // self.args.stride[0]) + 1
        self.n_patches_x = ((self.args.dim[1] - self.args.patch_dim[1]) // self.args.stride[1]) + 1
        self.n_patches = self.n_patches_y * self.n_patches_x

        # Number of position features. Uses thermometer encoding so need 1 less than the possible positions.
        self.n_position_feats = (self.n_patches_y - 1) + (self.n_patches_x - 1)

        # Thermometer bits per feature: max - min
        self.therm_bits = self.args.feat_maxs - self.args.feat_mins

        # The number of literals needed to represent all features using thermometer encoding
        self.n_patch_feats = int(np.sum(self.therm_bits))

        # Literal offsets: prefix sum for indexing into literals by feature
        self.literal_offsets = np.zeros(self.n_raw_patch_feats + 1, dtype=np.int32)
        self.literal_offsets[1:] = np.cumsum(self.therm_bits)

        # Total number of literals for all features + position
        self.n_literals = self.n_patch_feats + self.n_position_feats

        if self.args.negated_literals:
            self.n_literals *= 2

        if self.args.max_includes <= 0 or self.args.max_includes > self.n_literals:
            self.args.max_includes = self.n_literals

        # Precompute lit_to_fid lookup table: O(1) lookup instead of O(N_RAW_PATCH_FEATS) scan
        self.lit_to_fid = np.zeros(self.n_patch_feats, dtype=np.int32)
        for fid in range(self.n_raw_patch_feats):
            for lit in range(self.literal_offsets[fid], self.literal_offsets[fid + 1]):
                self.lit_to_fid[lit] = fid

        self.np_rng = np.random.default_rng(self.args.seed)

        self.dev_init()

    @abc.abstractmethod
    def dev_init(self):
        pass

    @abc.abstractmethod
    def _to_host(self, arr):
        """Return a copy of arr on host (numpy). `.copy()` for CPU, `.get()` (device->host transfer) for CUDA."""

    def _init_packed_clauses(self):
        self.packed_clauses = PackedClauses(
            clause_position_bounds=self.xp.empty((self.total_clauses, 4), dtype=np.int32),
            clause_feat_bounds=self.xp.empty((self.total_clauses, self.n_raw_patch_feats, 2), dtype=np.int32),
            bounded_feat_ids=self.xp.empty((self.total_clauses, self.n_raw_patch_feats), dtype=np.int32),
            n_bounded_feats=self.xp.empty(self.total_clauses, dtype=np.int32),
            clause_density=self.xp.empty(self.total_clauses, dtype=np.int32),
            is_clause_synced=self.xp.zeros(self.total_clauses, dtype=np.int8),
        )

    @abc.abstractmethod
    def set_threads(self, n: int):
        pass

    @abc.abstractmethod
    def pack_clauses(self, force_repack: bool = False):
        pass

    @abc.abstractmethod
    def fit_epoch(self, X: np.ndarray, Y: np.ndarray, clause_drop_p: float, batch_size: int, lr: float | None = None):
        pass

    @abc.abstractmethod
    def infer(self, X: np.ndarray, batch_size: int):
        pass

    @abc.abstractmethod
    def transform(self, X: np.ndarray, batch_size: int):
        pass

    @abc.abstractmethod
    def transform_patchwise(self, X: np.ndarray, batch_size: int):
        pass

    @abc.abstractmethod
    def load_state_dict(self, state_dict: dict):
        pass

    def _init_clauses(self):
        if self.args.ta_init == "middle":
            self.ta_states = self.xp.full(
                (self.total_clauses, self.n_literals),
                self.args.include_state - 1,
                dtype=np.uint32,
            )
        elif self.args.ta_init == "random":
            self.ta_states = self.xp.asarray(
                self.np_rng.integers(0, self.args.n_states, size=(self.total_clauses, self.n_literals)),
                dtype=np.uint32,
            )
        elif self.args.ta_init == "random_include":
            choice = self.np_rng.integers(0, 2, size=(self.total_clauses, self.n_literals))
            states = np.where(choice == 1, self.args.include_state, self.args.include_state - 1)
            self.ta_states = self.xp.asarray(states, dtype=np.uint32)
        elif isinstance(self.args.ta_init, str) and self.args.ta_init.startswith("random_"):
            n = int(self.args.ta_init[len("random_") :])
            mid = self.args.include_state - 1
            low = max(0, mid - n)
            high = min(self.args.n_states - 1, mid + n)
            self.ta_states = self.xp.asarray(
                self.np_rng.integers(low, high + 1, size=(self.total_clauses, self.n_literals)),
                dtype=np.uint32,
            )
        else:
            self.ta_states = self.xp.full(
                (self.total_clauses, self.n_literals),
                int(self.args.ta_init),
                dtype=np.uint32,
            )

    def _init_weights(self):
        shape = (self.args.n_classes, self.args.n_clauses)

        if self.args.weight_init == "random":
            mag = self.np_rng.uniform(0.0, 1.0, size=shape).astype(np.float32)
        else:
            mag = np.full(shape, float(self.args.weight_init), dtype=np.float32)

        if self.args.negative_clauses:
            n_neg_polarity = self.args.n_clauses // 2
            sign = np.ones(shape, dtype=np.float32)
            if self.args.coalesced:
                for i in range(self.args.n_classes):
                    pol = np.ones(self.args.n_clauses, dtype=np.float32)
                    pol[n_neg_polarity:] = -1.0
                    sign[i, :] = self.np_rng.permutation(pol)
            else:
                sign[:, n_neg_polarity:] = -1.0
            mag = mag * sign

        self.clause_weights = self.xp.asarray(mag, dtype=np.float32)

        if self.args.track_patch_weights:
            self.patch_weights = self.xp.zeros((self.total_clauses, self.n_patches), dtype=np.int32)
        else:
            self.patch_weights = self.xp.zeros((1, 1), dtype=np.int32)

    def _init_frozen_clauses(self):
        self.frozen_clauses = self.xp.zeros((self.n_clause_banks, self.args.n_clauses), dtype=np.int8)

    def _init_act_fn(self):
        if callable(self.args.act_fn):
            self.act_fn = self.args.act_fn
            self.dact_fn = lambda act: self.xp.ones_like(act)
        elif self.args.act_fn == "softmax":
            self.act_fn = lambda v: self._softmax(v, axis=-1)
            self.dact_fn = lambda act: self.xp.ones_like(act)
        elif self.args.act_fn == "sigmoid":
            self.act_fn = self._expit
            self.dact_fn = lambda act: act * (1.0 - act)
        elif self.args.act_fn == "identity":
            self.act_fn = lambda v: v
            self.dact_fn = lambda act: self.xp.ones_like(act)
        else:
            raise NotImplementedError(f"act_fn '{self.args.act_fn}' not implemented")

    def _init_loss_fn(self):
        if callable(self.args.loss_fn):
            _fn = self.args.loss_fn

            def _loss_fn(v, y, grad, **kwargs):
                return float(_fn(v, y, grad, **kwargs))
        else:
            lw = self.xp.asarray(
                self.args.loss_fn_kwargs.get("class_weights", np.ones(self.args.n_classes)),
                dtype=np.float32,
            )
            if self.args.loss_fn == "ce":
                gamma = self.args.loss_fn_kwargs.get("gamma", 0.0)
                if self.args.act_fn == "softmax":

                    def _loss_fn(v, y, grad, **kwargs):
                        act = self.act_fn(v)
                        fw = (1.0 - float(self.xp.dot(y, act))) ** gamma
                        grad[:] = (lw * fw * (y - act)).astype(np.float32)
                        return float(-self.xp.sum(lw * y * self._log_softmax(v))) * fw
                else:

                    def _loss_fn(v, y, grad, **kwargs):
                        act = self.act_fn(v)
                        p_t = y * act + (1.0 - y) * (1.0 - act)
                        fw = (1.0 - p_t) ** gamma
                        grad[:] = (lw * fw * (y - act)).astype(np.float32)
                        return float(-self.xp.sum(lw * fw * (y * self.xp.log(act + 1e-7) + (1 - y) * self.xp.log(1 - act + 1e-7))))
            elif self.args.loss_fn == "sce":
                alpha = self.args.loss_fn_kwargs.get("alpha", 1.0)
                beta = self.args.loss_fn_kwargs.get("beta", 1.0)
                eps = self.args.loss_fn_kwargs.get("eps", 1e-4)
                if self.args.act_fn == "softmax":

                    def _loss_fn(v, y, grad, **kwargs):
                        act = self.act_fn(v)
                        logy = self.xp.log(y + eps)
                        A = float(self.xp.dot(act, logy))
                        grad[:] = (lw * (alpha * (y - act) + beta * act * (logy - A))).astype(np.float32)
                        ce = float(-self.xp.sum(lw * y * self._log_softmax(v)))
                        rce = float(-self.xp.sum(lw * act * logy))
                        return alpha * ce + beta * rce
                else:

                    def _loss_fn(v, y, grad, **kwargs):
                        act = self.act_fn(v)
                        bce_grad = y - act
                        rce_grad = self.xp.log((y + eps) / (1.0 - y + eps)) * act * (1.0 - act)
                        grad[:] = (lw * (alpha * bce_grad + beta * rce_grad)).astype(np.float32)
                        bce = float(-self.xp.sum(lw * (y * self.xp.log(act + eps) + (1 - y) * self.xp.log(1 - act + eps))))
                        rce = float(-self.xp.sum(lw * (act * self.xp.log(y + eps) + (1 - act) * self.xp.log(1 - y + eps))))
                        return alpha * bce + beta * rce
            elif self.args.loss_fn == "mse":

                def _loss_fn(v, y, grad, **kwargs):
                    act = self.act_fn(v)
                    grad[:] = (2 * lw * (y - act) * self.dact_fn(act)).astype(np.float32)
                    return float(self.xp.sum(lw * (y - act) ** 2))
            elif self.args.loss_fn == "mae":

                def _loss_fn(v, y, grad, **kwargs):
                    act = self.act_fn(v)
                    grad[:] = (lw * self.xp.sign(y - act) * self.dact_fn(act)).astype(np.float32)
                    return float(self.xp.sum(lw * self.xp.abs(y - act)))
            elif self.args.loss_fn == "huber":
                delta = self.args.loss_fn_kwargs.get("delta", 1.0)

                def _loss_fn(v, y, grad, **kwargs):
                    act = self.act_fn(v)
                    r = y - act
                    grad[:] = (lw * self.xp.clip(r, -delta, delta) * self.dact_fn(act)).astype(np.float32)
                    huber = self.xp.where(self.xp.abs(r) <= delta, 0.5 * r**2, delta * (self.xp.abs(r) - 0.5 * delta))
                    return float(self.xp.sum(lw * huber))
            elif self.args.loss_fn == "tversky":
                alpha = self.args.loss_fn_kwargs.get("alpha", 0.5)
                beta = self.args.loss_fn_kwargs.get("beta", 0.5)
                gamma = self.args.loss_fn_kwargs.get("gamma", 1.0)
                eps = self.args.loss_fn_kwargs.get("eps", 1e-6)
                if self.args.act_fn == "sigmoid":

                    def _loss_fn(v, y, grad, **kwargs):
                        act = self.act_fn(v)
                        tp = float(self.xp.dot(lw * y, act))
                        fp = float(self.xp.dot(lw * (1.0 - y), act))
                        fn = float(self.xp.dot(lw * y, 1.0 - act))
                        N = tp + eps
                        D = tp + alpha * fp + beta * fn + eps
                        T = N / D
                        fw = gamma * (1.0 - T) ** (gamma - 1.0)
                        coeff = alpha + y * (1.0 - alpha - beta)
                        grad[:] = (fw * lw * (y * D - N * coeff) / D**2 * act * (1.0 - act)).astype(np.float32)
                        return float((1.0 - T) ** gamma)
                else:

                    def _loss_fn(v, y, grad, **kwargs):
                        act = self.act_fn(v)
                        tp = float(self.xp.dot(lw * y, act))
                        fp = float(self.xp.dot(lw * (1.0 - y), act))
                        fn = float(self.xp.dot(lw * y, 1.0 - act))
                        N = tp + eps
                        D = tp + alpha * fp + beta * fn + eps
                        T = N / D
                        fw = gamma * (1.0 - T) ** (gamma - 1.0)
                        coeff = alpha + y * (1.0 - alpha - beta)
                        grad[:] = (fw * lw * (y * D - N * coeff) / D**2 * self.dact_fn(act)).astype(np.float32)
                        return float((1.0 - T) ** gamma)
            elif self.args.loss_fn == "asl":
                gamma_pos = self.args.loss_fn_kwargs.get("gamma_pos", 0.0)
                gamma_neg = self.args.loss_fn_kwargs.get("gamma_neg", 4.0)
                clip = self.args.loss_fn_kwargs.get("clip", 0.05)
                eps = self.args.loss_fn_kwargs.get("eps", 1e-8)

                def _loss_fn(v, y, grad, **kwargs):
                    p = self.xp.clip(self.act_fn(v), eps, 1.0 - eps)
                    pm = self.xp.clip(p - clip, 0.0, 1.0) if clip > 0 else p
                    active_neg = (p - clip) > 0 if clip > 0 else self.xp.ones_like(p, dtype=bool)
                    pm = self.xp.clip(pm, eps, 1.0 - eps)

                    loss_pos = (1.0 - p) ** gamma_pos * self.xp.log(p)
                    loss_neg = (pm**gamma_neg) * self.xp.log(1.0 - pm)
                    loss = -float(self.xp.sum(y * loss_pos + (1.0 - y) * loss_neg))

                    grad_pos = (1.0 - p) ** (gamma_pos + 1.0) - gamma_pos * (1.0 - p) ** gamma_pos * p * self.xp.log(p)
                    grad_neg = (
                        gamma_neg * pm ** (gamma_neg - 1.0) * self.xp.log(1.0 - pm) - pm**gamma_neg / (1.0 - pm)
                    ) * p * (1.0 - p)
                    grad_neg = self.xp.where(active_neg, grad_neg, 0.0)

                    grad[:] = (y * grad_pos + (1.0 - y) * grad_neg).astype(np.float32)
                    return loss
            else:
                raise NotImplementedError(f"loss_fn '{self.args.loss_fn}' not implemented")
        self.loss_fn = _loss_fn

    def freeze_clauses(self, class_id: int, clause_ids: list[int] | np.ndarray):
        clause_ids = np.asarray(clause_ids, dtype=np.int32)
        self.frozen_clauses[class_id, clause_ids] = 1

    def unfreeze_clauses(self):
        self.frozen_clauses.fill(0)

    def get_weights(self):
        return self._to_host(self.clause_weights)

    def get_ta_states(self):
        return self._to_host(self.ta_states).reshape((self.n_clause_banks, self.args.n_clauses, self.n_literals))

    def get_patch_weights(self):
        if not self.args.track_patch_weights:
            raise RuntimeError("track_patch_weights is False, so no patch_weights were saved.")
        return self._to_host(self.patch_weights).reshape(self.n_clause_banks, self.args.n_clauses, self.n_patches_y, self.n_patches_x)

    def get_state_dict(self):
        return {
            "ta_states": self._to_host(self.ta_states),
            "clause_weights": self._to_host(self.clause_weights),
            "patch_weights": self._to_host(self.patch_weights),
        }

    def get_packed_clauses(self) -> PackedClauses:
        pc = self.packed_clauses
        return PackedClauses(
            clause_position_bounds=self._to_host(pc.clause_position_bounds),
            clause_feat_bounds=self._to_host(pc.clause_feat_bounds),
            bounded_feat_ids=self._to_host(pc.bounded_feat_ids),
            n_bounded_feats=self._to_host(pc.n_bounded_feats),
            clause_density=self._to_host(pc.clause_density),
            is_clause_synced=self._to_host(pc.is_clause_synced),
        )

    def _build_header(self):
        header = f"""
#define TOTAL_CLAUSES {self.total_clauses}
#define S {float(self.args.s)}f
#define CLASSES {self.args.n_classes}
#define HEIGHT {self.args.dim[0]}
#define WIDTH {self.args.dim[1]}
#define DEPTH {self.args.dim[2]}
#define PATCH_HEIGHT {self.args.patch_dim[0]}
#define PATCH_WIDTH {self.args.patch_dim[1]}
#define STRIDE_Y {self.args.stride[0]}
#define STRIDE_X {self.args.stride[1]}
#define MAX_WEIGHT {float(self.args.max_weight)}f
#define MAX_INCLUDED_LITERALS {self.args.max_includes}
#define INCLUDE_STATE {self.args.include_state}
#define MAX_TA_STATE {self.args.n_states - 1}
#define N_RAW_PATCH_FEATS {self.n_raw_patch_feats}
#define N_PATCH_FEATS {self.n_patch_feats}
#define N_POSITION_FEATS {self.n_position_feats}
#define N_PATCHES_Y {self.n_patches_y}
#define N_PATCHES_X {self.n_patches_x}
#define N_PATCHES {self.n_patches}
#define N_LITERALS {self.n_literals}
#define NEGATED_LITERALS {1 if self.args.negated_literals else 0}
#define POSITION_LITERALS {1 if self.args.position_literals else 0}
#define COALESCED {1 if self.args.coalesced else 0}
#define WEIGHTED {1 if self.args.weighted else 0}
#define NEGATIVE_CLAUSES {1 if self.args.negative_clauses else 0}
#define ALLOW_POLARITY_CHANGE {1 if self.args.allow_polarity_change else 0}
#define TYPE1A_FB {0 if self.args.skip_t1a_fb else 1}
#define TYPE1B_FB {0 if self.args.skip_t1b_fb else 1}
#define TYPE2_FB {0 if self.args.skip_t2_fb else 1}
#define TRACK_PATCH_WEIGHTS {1 if self.args.track_patch_weights else 0}
#define BOOST_TP_FB {1 if self.args.boost_tp_fb else 0}
#define WARPS_PER_CLAUSE {self.args.warps_per_clause}
"""
        return header
