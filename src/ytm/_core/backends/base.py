import abc
import types
from collections.abc import Iterator

import numpy as np

from ..config import BaseTMConfig
from ..device_config import DeviceConfig
from ..utils import PackedClauses


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
    def _code_sections(self) -> dict[str, str]: ...

    @abc.abstractmethod
    def pack_clauses(self, force_repack: bool = False, full: bool = False): ...

    @abc.abstractmethod
    def fit_epoch(self, X: np.ndarray, Y: np.ndarray, clause_drop_p: float, batch_size: int, *args, **kwargs): ...

    @abc.abstractmethod
    def fit_sample(self, rng_key: int, buf, e: int): ...

    @abc.abstractmethod
    def _fit_eval(self, buf, e: int, rng_key): ...

    @abc.abstractmethod
    def _fit_voting(self, buf): ...

    @abc.abstractmethod
    def _fit_decide_fb(self, buf, e: int, rng_key: int): ...

    @abc.abstractmethod
    def _fit_apply_fb(self, buf, e: int, rng_key): ...

    @abc.abstractmethod
    def _fit_update_weights(self, buf): ...

    @abc.abstractmethod
    def calc_class_sums(self, X: np.ndarray, force_repack: bool = False) -> np.ndarray: ...

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

    @staticmethod
    def make_device(config: BaseTMConfig, device_config: DeviceConfig) -> "BaseDevice":
        """Build the device for `device_config`. Each backends package sets this on its classes."""
        raise NotImplementedError("the backends package must set `make_device` on its device classes.")

    def _build_code(self) -> str:
        return self.config._header + "\n".join(self._code_sections().values())

    def _fit_samples(self, pbar) -> Iterator[tuple[int, int]]:
        for e in pbar:
            yield e, int(self._rng.integers(1, 1 << 62, dtype=np.uint64))

    def _fit_drop_mask(self, clause_drop_p: float):
        cfg = self.config
        if clause_drop_p <= 0.0:
            return self.xp.zeros(cfg._total_clauses, dtype=np.int8)
        return self.xp.asarray((self._rng.random(cfg._total_clauses) <= clause_drop_p).astype(np.int8))

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

        self.ta_states = self.xp.asarray(states, dtype=cfg._ta_dtype)

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

    def get_patch_weights(self) -> np.ndarray:
        cfg = self.config
        if not cfg.track_patch_weights:
            raise RuntimeError("`track_patch_weights` must be True to get the patch weights.")
        return self._to_host(self.patch_weights).reshape(cfg._n_clause_banks, cfg._n_clauses, cfg._n_patches_y, cfg._n_patches_x)

    def get_packed_clauses(self) -> PackedClauses:
        pc = self.packed_clauses
        return PackedClauses(
            clause_feat_bounds=self._to_host(pc.clause_feat_bounds),
            clause_position_bounds=self._to_host(pc.clause_position_bounds),
            bounded_feat_ids=self._to_host(pc.bounded_feat_ids),
            n_bounded_feats=self._to_host(pc.n_bounded_feats),
            clause_density=self._to_host(pc.clause_density),
            is_clause_synced=self._to_host(pc.is_clause_synced),
        )

    # == serialization ==
    def get_state_dict(self) -> dict:
        return {
            "ta_states": self._to_host(self.ta_states),
            "clause_weights": self._to_host(self.clause_weights),
            "patch_weights": self._to_host(self.patch_weights),
            "rng": self._rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict) -> None:
        """Written in place, so anything already pointing at these arrays stays valid."""
        self.ta_states[:] = self.xp.asarray(state["ta_states"])
        self.clause_weights[:] = self.xp.asarray(state["clause_weights"])
        self.patch_weights[:] = self.xp.asarray(state["patch_weights"])
        self._rng.bit_generator.state = state["rng"]
        self.packed_clauses.is_clause_synced.fill(0)

    def __getstate__(self) -> dict:
        return {
            "config": self.config,
            "params": self.get_state_dict(),
        }

    def __setstate__(self, state: dict) -> None:
        self.__init__(state["config"], DeviceConfig())
        self.load_state_dict(state["params"])

    def to(self, device_config: DeviceConfig) -> "BaseDevice":
        """The same model on a different backend. Returns a new device, `self` is left alone."""
        new = type(self).make_device(self.config, device_config)
        new.load_state_dict(self.get_state_dict())
        return new
