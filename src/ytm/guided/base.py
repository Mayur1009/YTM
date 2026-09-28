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
    if device_config.kind == "cpu":
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
        s: float,
        dim: int | tuple[int, ...],
        n_classes: int,
        **opt: Unpack[T_Config],
    ):
        cfg_kw, dev_kw = split_device_kwargs(dict(opt))
        config = self.config_cls(n_clauses=n_clauses, s=s, dim=dim, n_classes=n_classes, **cfg_kw)
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
        lr: float | None = None,
        lambda_: float | None = None,
    ) -> float:
        assert Y.ndim == 2, f"Y must be 2D array (samples, outputs), got {Y.ndim}D"
        X = self._prepare_X(X)

        iota = self._rng.permutation(X.shape[0]) if shuffle else np.arange(X.shape[0])
        X = X[iota]
        Y = np.asarray(Y, dtype=np.float32, order="C")[iota]

        return self.dev.fit_epoch(X, Y, clause_drop_p, batch_size, lr, lambda_)

    def raw_votes(self, X: np.ndarray, batch_size: int = -1, force_repack: bool = False) -> np.ndarray:
        return self.dev.raw_votes(self._prepare_X(X), batch_size, force_repack)
