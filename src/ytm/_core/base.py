import abc

import numpy as np

from .backends.base import BaseDevice
from .config import BaseTMConfig
from .device_config import DeviceConfig
from .utils import ClauseInfo


class BaseTM(abc.ABC):
    def __init__(self, dev: BaseDevice):
        self.dev = dev
        self._rng = np.random.default_rng(self.config.seed)

    @property
    def config(self) -> BaseTMConfig:
        return self.dev.config

    @property
    def device_config(self) -> DeviceConfig:
        return self.dev.device_config

    def to(self, device: str, **device_kwargs) -> None:
        """Move the model to another device, in place."""
        self.dev = self.dev.to(DeviceConfig(device=device, **device_kwargs))

    @abc.abstractmethod
    def _fit(self, X: np.ndarray, Y: np.ndarray, *args, **kwargs): ...

    def score(self, X: np.ndarray, batch_size: int = -1, force_repack: bool = False) -> np.ndarray:
        return self.dev.calc_class_sums(np.ascontiguousarray(X), batch_size, force_repack)

    def transform(self, X: np.ndarray, batch_size: int = -1, force_repack: bool = False) -> np.ndarray:
        return self.dev.transform(np.ascontiguousarray(X), batch_size, force_repack)

    def transform_patchwise(self, X: np.ndarray, batch_size: int = -1, force_repack: bool = False) -> np.ndarray:
        return self.dev.transform_patchwise(np.ascontiguousarray(X), batch_size, force_repack)

    def wic(self, class_id: int, polarity: int, pw_th: float = 0.0, force_repack: bool = False) -> np.ndarray:
        if self.config._n_patches > 1 and not self.config.track_patch_weights:
            raise ValueError("track_patch_weights=True is required for wic() on a convolutional model.")
        return self.dev.wic(class_id, polarity, pw_th, force_repack)

    def wac(self, X: np.ndarray, target_classes: np.ndarray, polarity: int, force_repack: bool = False) -> np.ndarray:
        return self.dev.wac(np.ascontiguousarray(X), target_classes, polarity, force_repack)

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
        if cfg.position_literals or cfg._n_patches > 1:
            position_bounds = packed.clause_position_bounds.reshape(*shape, 4)

        return ClauseInfo(
            feature_bounds=packed.clause_feat_bounds.reshape(*shape, cfg._n_raw_patch_feats * 2),
            position_bounds=position_bounds,
            clause_density=packed.clause_density.reshape(*shape),
        )
