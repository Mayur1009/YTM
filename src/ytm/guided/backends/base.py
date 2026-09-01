import abc
import types
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from tqdm import tqdm

from ..args import ACT_FN_CODES, FB_SIGNAL_CODES, LOSS_FN_CODES, TMArgs


def _dict_to_header(prefix: str, codes: dict[str, int]) -> str:
    return "\n".join(f"#define {prefix}_{name.upper()} {code}" for name, code in codes.items())


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

        self._rng = np.random.default_rng(self.args.seed + 1)

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
    def fit_epoch(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        clause_drop_p: float,
        batch_size: int,
        lr: float | None = None,
        lambda_: float | None = None,
    ):
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

    def _init_loss_fn(self):
        self._loss_class_weights = self.xp.asarray(
            self.args.loss_fn_kwargs.get("class_weights", np.ones(self.args.n_classes)), dtype=np.float32
        )

    def _init_clauses(self):
        if self.args.ta_init == "middle":
            self.ta_states = self.xp.full(
                (self.total_clauses, self.n_literals),
                self.args.include_state - 1,
                dtype=np.uint32,
            )
        elif self.args.ta_init == "random":
            self.ta_states = self.xp.asarray(
                self._rng.integers(0, self.args.n_states, size=(self.total_clauses, self.n_literals)),
                dtype=np.uint32,
            )
        elif self.args.ta_init == "random_include":
            choice = self._rng.integers(0, 2, size=(self.total_clauses, self.n_literals))
            states = np.where(choice == 1, self.args.include_state, self.args.include_state - 1)
            self.ta_states = self.xp.asarray(states, dtype=np.uint32)
        elif isinstance(self.args.ta_init, str) and self.args.ta_init.startswith("random:"):
            n = int(self.args.ta_init[len("random:") :])
            mid = self.args.include_state - 1
            low = max(0, mid - n)
            high = min(self.args.n_states - 1, mid + n)
            self.ta_states = self.xp.asarray(
                self._rng.integers(low, high + 1, size=(self.total_clauses, self.n_literals)),
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
            mag = self._rng.uniform(0.0, 1.0, size=shape).astype(np.float32)
        elif isinstance(self.args.weight_init, str) and self.args.weight_init.startswith("random:"):
            n = float(self.args.weight_init[len("random:") :])
            mag = self._rng.uniform(0.0, n, size=shape).astype(np.float32)
        else:
            mag = np.full(shape, float(self.args.weight_init), dtype=np.float32)

        if self.args.negative_clauses:
            n_neg_polarity = self.args.n_clauses // 2
            sign = np.ones(shape, dtype=np.float32)
            if self.args.coalesced:
                for i in range(self.args.n_classes):
                    pol = np.ones(self.args.n_clauses, dtype=np.float32)
                    pol[n_neg_polarity:] = -1.0
                    sign[i, :] = self._rng.permutation(pol)
            else:
                sign[:, n_neg_polarity:] = -1.0
            mag = mag * sign

        self.clause_weights = self.xp.asarray(mag, dtype=np.float32)

        if self.args.track_patch_weights:
            self.patch_weights = self.xp.zeros((self.total_clauses, self.n_patches), dtype=np.int32)
        else:
            self.patch_weights = self.xp.zeros((1, 1), dtype=np.int32)

    def _init_bias(self):
        if not self.args.bias:
            self.bias = self.xp.zeros((1,), dtype=np.float32)
        else:
            if isinstance(self.args.bias_init, float):
                self.bias = self.xp.full((self.args.n_classes,), self.args.bias_init, dtype=np.float32)
            elif self.args.bias_init == "random":
                bias_rand = self._rng.uniform(0.0, 1.0, size=(self.args.n_classes,))
                self.bias = self.xp.asarray(bias_rand, dtype=np.float32)
            else:
                raise ValueError

    def _init_frozen_clauses(self):
        self.frozen_clauses = self.xp.zeros((self.n_clause_banks, self.args.n_clauses), dtype=np.int8)

    def freeze_clauses(self, class_id: int, clause_ids: list[int] | np.ndarray):
        clause_ids = np.asarray(clause_ids, dtype=np.int32)
        self.frozen_clauses[class_id, clause_ids] = 1

    def unfreeze_clauses(self):
        self.frozen_clauses.fill(0)

    def get_weights(self):
        return self._to_host(self.clause_weights)

    def get_bias(self):
        if not self.args.bias:
            raise RuntimeError("`bias` should be set to `True` to get bias.")
        return self._to_host(self.bias)

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
            "bias": self._to_host(self.bias),
            "rng": self._rng,
        }

    def load_state_dict(self, state_dict: dict):
        self.ta_states = self.xp.asarray(state_dict["ta_states"])
        self.clause_weights = self.xp.asarray(state_dict["clause_weights"])
        self.patch_weights = self.xp.asarray(state_dict["patch_weights"])
        self.bias = self.xp.asarray(state_dict["bias"])
        self._rng = state_dict["rng"]
        self.packed_clauses.is_clause_synced.fill(0)

    def __getstate__(self):
        return {"args": asdict(self.args), "params": self.get_state_dict()}

    def __setstate__(self, state):
        self.__init__(TMArgs(**state["args"]))
        self.load_state_dict(state["params"])

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
        fb_signal_code = FB_SIGNAL_CODES[self.args.fb_signal]
        act_fn_code = ACT_FN_CODES[self.args.act_fn]
        loss_fn_code = LOSS_FN_CODES[self.args.loss_fn]

        kw = self.args.loss_fn_kwargs
        loss_fn = self.args.loss_fn
        loss_gamma = kw.get("gamma", 1.0 if loss_fn == "tversky" else 0.0)
        loss_eps = kw.get("eps", 1e-4 if loss_fn == "sce" else (1e-7 if loss_fn == "ce" else 1e-6))
        loss_alpha = kw.get("alpha", 1.0 if loss_fn == "sce" else 0.5)
        loss_beta = kw.get("beta", 1.0 if loss_fn == "sce" else 0.5)
        loss_delta = kw.get("delta", 1.0)
        loss_clip = kw.get("clip", 0.05)
        loss_gamma_pos = kw.get("gamma_pos", 0.0)
        loss_gamma_neg = kw.get("gamma_neg", 4.0)

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
#define BIAS {1 if self.args.bias else 0}

{_dict_to_header("FB_SIGNAL", FB_SIGNAL_CODES)}
{_dict_to_header("ACT", ACT_FN_CODES)}
{_dict_to_header("LOSS", LOSS_FN_CODES)}

#define FB_SIGNAL {fb_signal_code}
#define ACT_FN {act_fn_code}
#define LOSS_FN {loss_fn_code}
#define LOSS_GAMMA {float(loss_gamma)}f
#define LOSS_EPS {float(loss_eps)}f
#define LOSS_ALPHA {float(loss_alpha)}f
#define LOSS_BETA {float(loss_beta)}f
#define LOSS_DELTA {float(loss_delta)}f
#define LOSS_CLIP {float(loss_clip)}f
#define LOSS_GAMMA_POS {float(loss_gamma_pos)}f
#define LOSS_GAMMA_NEG {float(loss_gamma_neg)}f
"""
        return header
