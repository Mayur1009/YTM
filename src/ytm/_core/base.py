import abc
from dataclasses import dataclass, fields

import numpy as np

from .backends.base import BaseDevice
from .config import BaseTMConfig
from .device_config import DeviceConfig
from .utils import prepare_X


@dataclass
class ClauseInfo:
    """Packed clause representation returned by :meth:`BaseTM.get_clauses`.

    Attributes
    ----------
    feature_bounds : ndarray of shape (n_clause_banks, n_clauses, n_raw_patch_feats * 2)
        Closed ``[lower, upper]`` feature inclusion bounds per clause.
    position_bounds : ndarray of shape (n_clause_banks, n_clauses, 4) or None
        Closed ``[min_y, max_y, min_x, max_x]`` position bounds per clause.
        ``None`` when the input has a single patch.
    has_contra : ndarray of shape (n_clause_banks, n_clauses), dtype int8
        ``1`` when the clause is unsatisfiable (contains a contradiction).
    clause_len : ndarray of shape (n_clause_banks, n_clauses), dtype int
        Number of included literals per clause. ``0`` means the clause is
        vacuous and fires on every input.
    """

    feature_bounds: np.ndarray
    position_bounds: np.ndarray | None
    has_contra: np.ndarray
    clause_len: np.ndarray


class BaseTM(abc.ABC):
    config_cls: type[BaseTMConfig]
    cpu_device_cls: type[BaseDevice]

    def __init__(self, dev: BaseDevice):
        self.dev = dev
        self._rng = np.random.default_rng(self.config.seed)

    @property
    def config(self) -> BaseTMConfig:
        return self.dev.config

    @property
    def device_config(self) -> DeviceConfig:
        return self.dev.device_config

    @staticmethod
    @abc.abstractmethod
    def _cuda_device_cls() -> type[BaseDevice]: ...

    def _build_device(self, config: BaseTMConfig, device_config: DeviceConfig, state: dict | None = None) -> BaseDevice:
        cls = self.cpu_device_cls if device_config.kind == "cpu" else self._cuda_device_cls()
        return cls(config, device_config, state)

    def to(self, device: str, **device_kwargs) -> None:
        """Move the model to another device, in place."""
        self.dev = self._build_device(self.config, DeviceConfig(device=device, **device_kwargs), self.dev.get_state_dict())

    def __getstate__(self) -> dict:
        """Config is saved as its constructor arguments, minus `ta_init`: its only use is at construction."""
        cfg = self.config
        return {
            "config": {f.name: getattr(cfg, f.name) for f in fields(cfg) if f.init and f.name != "ta_init"},
            "params": self.dev.get_state_dict(),
            "rng": self._rng,
        }

    def __setstate__(self, state: dict) -> None:
        """Unpickling always lands on `cpu:1`; use `.to()` afterwards to move it."""
        cfg = self.config_cls(**state["config"])
        self.dev = self._build_device(cfg, DeviceConfig(), state["params"])
        self._rng = state["rng"]

    @abc.abstractmethod
    def _fit(self, X: np.ndarray, Y: np.ndarray, *args, **kwargs): ...

    def _prepare_X(self, X: np.ndarray) -> np.ndarray:
        return prepare_X(self.config, X)

    def score(self, X: np.ndarray, batch_size: int = -1, force_repack: bool = False) -> np.ndarray:
        return self.dev.calc_class_sums(self._prepare_X(X), batch_size, force_repack)

    @abc.abstractmethod
    def to_prob(self, class_sums: np.ndarray) -> np.ndarray:
        """Map class sums from `score`/`predict` to [0, 1]."""

    def transform(self, X: np.ndarray, batch_size: int = -1, force_repack: bool = False) -> np.ndarray:
        return self.dev.transform(self._prepare_X(X), batch_size, force_repack)

    def transform_patchwise(self, X: np.ndarray, batch_size: int = -1, force_repack: bool = False) -> np.ndarray:
        return self.dev.transform_patchwise(self._prepare_X(X), batch_size, force_repack)

    def wic(self, class_id: int, polarity: int, pw_th: float = 0.0, force_repack: bool = False) -> np.ndarray:
        if self.config._n_patches > 1 and not self.config.track_patch_weights:
            raise ValueError("track_patch_weights=True is required for wic() on a convolutional model.")
        return self.dev.wic(class_id, polarity, pw_th, force_repack)

    def wac(self, X: np.ndarray, target_classes: np.ndarray, polarity: int, batch_size: int = -1, force_repack: bool = False) -> np.ndarray:
        return self.dev.wac(self._prepare_X(X), target_classes, polarity, batch_size, force_repack)

    def set_threads(self, n: int) -> None:
        self.dev.set_threads(max(1, n))

    # == getters ==
    def get_weights(self) -> np.ndarray:
        return self.dev.get_weights()

    def get_ta_states(self) -> np.ndarray:
        return self.dev.get_ta_states()

    def get_patch_weights(self) -> np.ndarray:
        return self.dev.get_patch_weights()

    def get_literals(self) -> np.ndarray:
        return np.asarray(self.get_ta_states() >= self.config._include_state, dtype=np.uint8)

    def get_clauses(self, force_repack: bool = True, full: bool = True) -> ClauseInfo:
        cfg = self.config
        self.dev.pack_clauses(force_repack, full)
        packed = self.dev.get_packed_clauses()
        shape = (cfg._n_clause_banks, cfg._n_clauses)

        position_bounds = None
        if cfg._n_patches > 1:
            position_bounds = packed.clause_position_bounds.reshape(*shape, 4)

        # Scatter the compacted bounds back to one entry per feature, unconstrained ones spanning
        # the full range, then undo the internal zero basing so the intervals are in user units.
        n_feat = cfg._n_raw_patch_feats
        feature_bounds = np.empty((cfg._total_clauses, n_feat, 2), dtype=np.int32)
        feature_bounds[:, :, 0] = 0
        feature_bounds[:, :, 1] = cfg._therm_bits

        fids = packed.clause_feat_ids.reshape(cfg._total_clauses, n_feat)
        compact = packed.clause_feat_bounds.reshape(cfg._total_clauses, n_feat, 2)
        used = np.arange(n_feat)[None, :] < packed.clause_n_feats[:, None]
        ci, si = np.nonzero(used)
        feature_bounds[ci, fids[ci, si]] = compact[ci, si]

        feature_bounds = feature_bounds.reshape(*shape, n_feat, 2) + cfg._feat_mins.reshape(-1, 1)

        return ClauseInfo(
            feature_bounds=feature_bounds.reshape(*shape, cfg._n_raw_patch_feats * 2),
            position_bounds=position_bounds,
            has_contra=packed.has_contra.reshape(*shape),
            clause_len=packed.clause_len.reshape(*shape),
        )
