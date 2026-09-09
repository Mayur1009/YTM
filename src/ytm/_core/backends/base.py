import abc
import types

import numpy as np
from tqdm import tqdm

from ..config import BaseTMConfig
from ..device_config import DeviceConfig
from .types import PackedClauses


def tqdm_bar(iterable, **kwargs):
    args = {
        "leave": False,
        "dynamic_ncols": True,
        "bar_format": "{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
    }
    args.update(kwargs)
    return tqdm(iterable, **args)


class BaseDevice(abc.ABC):
    xp: types.ModuleType

    def __init__(self, config: BaseTMConfig, device_config: DeviceConfig):
        self.config = config
        self.device_config = device_config
        self._rng = np.random.default_rng(self.config.seed + 1)
        self.dev_init()

    @abc.abstractmethod
    def dev_init(self): ...

    @abc.abstractmethod
    def _to_host(self, arr) -> np.ndarray: ...

    @abc.abstractmethod
    def pack_clauses(self, force_repack: bool = False): ...

    @abc.abstractmethod
    def fit_epoch(self, X: np.ndarray, Y: np.ndarray, clause_drop_p: float, batch_size: int, **kwargs): ...

    @abc.abstractmethod
    def fit_sample(self, X, Y, e: int, **kwargs): ...

    @abc.abstractmethod
    def infer(self, X: np.ndarray, batch_size: int): ...

    @abc.abstractmethod
    def transform(self, X: np.ndarray, batch_size: int, force_repack: bool = False): ...

    @abc.abstractmethod
    def transform_patchwise(self, X: np.ndarray, batch_size: int, force_repack: bool = False): ...

    @abc.abstractmethod
    def wic(self, class_id: int, polarity: int, pw_th: float = 0.0, force_repack: bool = False): ...

    @abc.abstractmethod
    def wac(self, X: np.ndarray, target_classes: np.ndarray, polarity: int, force_repack: bool = False): ...

    def set_threads(self, n: int):
        raise NotImplementedError(f"set_threads is not supported on {self.device_config.device!r}.")

    # == initializations ==
    def _init_clauses(self):
        cfg = self.config
        shape = (cfg._total_clauses, cfg._n_literals)

        if cfg.ta_init == "middle":
            states = np.full(shape, cfg._include_state - 1)
        elif cfg.ta_init == "random":
            states = self._rng.integers(0, cfg.n_states, size=shape)
        elif cfg.ta_init == "random_include":
            states = np.where(self._rng.integers(0, 2, size=shape) == 1, cfg._include_state, cfg._include_state - 1)
        elif isinstance(cfg.ta_init, str):
            band = int(cfg.ta_init[len("random:") :])
            mid = cfg._include_state - 1
            states = self._rng.integers(max(0, mid - band), min(cfg.n_states - 1, mid + band) + 1, size=shape)
        else:
            states = np.full(shape, int(cfg.ta_init))

        self.ta_states = self.xp.asarray(states, dtype=np.uint32)

    def _init_weights(self):
        cfg = self.config
        shape = (cfg.n_classes, cfg._n_clauses)

        if cfg.weight_init == "random":
            mag = self._rng.uniform(0.0, 1.0, size=shape)
        elif isinstance(cfg.weight_init, str):
            mag = self._rng.uniform(0.0, float(cfg.weight_init[len("random:") :]), size=shape)
        else:
            mag = np.full(shape, float(cfg.weight_init))

        if cfg.negative_clauses:
            n_neg = cfg._n_clauses // 2
            sign = np.ones(shape, dtype=np.float32)
            if cfg.coalesced:
                pol = np.ones(cfg._n_clauses, dtype=np.float32)
                pol[n_neg:] = -1.0
                for i in range(cfg.n_classes):
                    sign[i, :] = self._rng.permutation(pol)
            else:
                sign[:, n_neg:] = -1.0
            mag = mag * sign

        self.clause_weights = self.xp.asarray(mag, dtype=np.float32)

    def _init_bias(self):
        cfg = self.config
        if not cfg.bias:
            self.bias = self.xp.zeros(1, dtype=np.float32)
        elif cfg.bias_init == "random":
            self.bias = self.xp.asarray(self._rng.uniform(0.0, 1.0, size=cfg.n_classes), dtype=np.float32)
        else:
            self.bias = self.xp.full(cfg.n_classes, float(cfg.bias_init), dtype=np.float32)

    def _init_patch_weights(self):
        cfg = self.config
        shape = (cfg._total_clauses, cfg._n_patches) if cfg.track_patch_weights else (1, 1)
        self.patch_weights = self.xp.zeros(shape, dtype=np.int32)

    def _init_packed_clauses(self):
        cfg = self.config
        self.packed_clauses = PackedClauses(
            clause_feat_bounds=self.xp.empty((cfg._total_clauses, cfg._n_raw_patch_feats, 2), dtype=np.int32),
            clause_position_bounds=self.xp.empty((cfg._total_clauses, 4), dtype=np.int32),
            bounded_feat_ids=self.xp.empty((cfg._total_clauses, cfg._n_raw_patch_feats), dtype=np.int32),
            n_bounded_feats=self.xp.empty(cfg._total_clauses, dtype=np.int32),
            clause_density=self.xp.empty(cfg._total_clauses, dtype=np.int32),
            is_clause_synced=self.xp.zeros(cfg._total_clauses, dtype=np.int8),
        )

    # == Getters ==
    def get_ta_states(self) -> np.ndarray:
        cfg = self.config
        return self._to_host(self.ta_states).reshape(cfg._n_clause_banks, cfg._n_clauses, cfg._n_literals)

    def get_weights(self) -> np.ndarray:
        return self._to_host(self.clause_weights)

    def get_bias(self) -> np.ndarray:
        if not self.config.bias:
            raise RuntimeError("`bias` must be True to get the bias.")
        return self._to_host(self.bias)

    def get_patch_weights(self) -> np.ndarray:
        cfg = self.config
        if not cfg.track_patch_weights:
            raise RuntimeError("`track_patch_weights` must be True to get the patch weights.")
        return self._to_host(self.patch_weights).reshape(cfg._n_clause_banks, cfg._n_clauses, cfg._n_patches_y, cfg._n_patches_x)

    def get_packed_clauses(self) -> PackedClauses:
        """Host copy of the packed form. Call `pack_clauses` first if it may be stale."""
        pc = self.packed_clauses
        return PackedClauses(
            clause_feat_bounds=self._to_host(pc.clause_feat_bounds),
            clause_position_bounds=self._to_host(pc.clause_position_bounds),
            bounded_feat_ids=self._to_host(pc.bounded_feat_ids),
            n_bounded_feats=self._to_host(pc.n_bounded_feats),
            clause_density=self._to_host(pc.clause_density),
            is_clause_synced=self._to_host(pc.is_clause_synced),
        )
