from typing import Unpack

import numpy as np

from .._core.backends.base import BaseDevice
from .._core.base import BaseTM as CoreBaseTM
from .._core.config import BaseTMConfig
from .._core.device_config import DeviceConfig
from .._core.utils import split_device_kwargs
from .backends.cpu import CPUDevice
from .config import T_Config, TMConfig


def _build_device(config: BaseTMConfig, device_config: DeviceConfig) -> BaseDevice:
    if device_config._device_kind == "cpu":
        return CPUDevice(config, device_config)
    from .backends.cuda import CUDADevice  # lazy: don't require cupy on cpu-only installs

    return CUDADevice(config, device_config)


class BaseTM(CoreBaseTM):
    config: TMConfig
    config_cls: type[TMConfig] = TMConfig
    cpu_device_cls = CPUDevice

    def __init__(
        self,
        n_clauses: int,
        T: float | tuple[float, float],
        s: float,
        dim: int | tuple[int, ...],
        n_classes: int,
        **opt: Unpack[T_Config],
    ):
        cfg_kw, dev_kw = split_device_kwargs(dict(opt))
        config = self.config_cls(n_clauses=n_clauses, T=T, s=s, dim=dim, n_classes=n_classes, **cfg_kw)
        super().__init__(_build_device(config, DeviceConfig(**dev_kw)))

    def to(self, device: str, **device_kwargs) -> None:
        """Move the model to another device, in place."""
        new = _build_device(self.config, DeviceConfig(device=device, **device_kwargs))
        new.load_state_dict(self.dev.get_state_dict())
        self.dev = new

    def _fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        shuffle: bool = True,
        clause_drop_p: float = 0.0,
        batch_size: int = -1,
        label_sampling: bool = False,
    ) -> None:
        X = self._prepare_X(X)

        iota = self._rng.permutation(X.shape[0]) if shuffle else np.arange(X.shape[0])
        X = X[iota]

        encoded_Y = np.asarray(self._encode_Y(Y[iota]), dtype=np.float32, order="C")
        label_probs = np.asarray(self._label_sampler(encoded_Y, label_sampling), dtype=np.float32, order="C")

        self.dev.fit_epoch(X, encoded_Y, clause_drop_p, batch_size, label_probs)

    def score(self, X: np.ndarray, batch_size: int = -1, force_repack: bool = False, clip_class_sums: bool = False) -> np.ndarray:
        class_sums = super().score(X, batch_size, force_repack)
        if clip_class_sums:
            class_sums = np.clip(class_sums, self.config._T_min, self.config._T_max)
        return class_sums

    def _encode_Y(self, Y: np.ndarray) -> np.ndarray:
        assert Y.ndim == 2, f"Y must be 2D array (samples, outputs), got {Y.ndim}D"
        assert np.unique(Y).tolist() == [0, 1], "Y must be binary (0 or 1)"
        return ((np.copy(Y).astype(np.float32) * 2) - 1) * self.config._T_max

    def _label_sampler(self, encoded_Y: np.ndarray, label_sampling: bool) -> np.ndarray:
        cfg = self.config
        if cfg.n_classes == 1:
            return np.ones_like(encoded_Y, dtype=np.float32)

        label_probs = np.full_like(encoded_Y, cfg.q / (cfg.n_classes - 1), dtype=np.float32)
        label_probs[encoded_Y == cfg._T_max] = 1.0
        return label_probs
