from typing import Unpack, cast

import numpy as np

from .._core.base import BaseTM as CoreBaseTM
from .._core.device_config import DeviceConfig, split_device_kwargs
from .config import T_Config, TMConfig


class BaseTM(CoreBaseTM):
    @staticmethod
    def _config_cls() -> type[TMConfig]:
        return TMConfig

    @staticmethod
    def _device_cls(kind: str):
        if kind == "cpu":
            from .backends.cpu import CPUDevice

            return CPUDevice
        from .backends.cuda import CUDADevice

        return CUDADevice

    def __init__(
        self,
        n_clauses: int,
        s: float,
        dim: int | tuple[int, ...],
        n_classes: int,
        **opt: Unpack[T_Config],
    ):
        cfg_kw, dev_kw = split_device_kwargs(dict(opt))
        config = self._config_cls()(n_clauses=n_clauses, s=s, dim=dim, n_classes=n_classes, **cfg_kw)
        super().__init__(config, DeviceConfig(**dev_kw))

    @property
    def config(self) -> TMConfig:
        return cast(TMConfig, self.dev.config)

    def _fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        shuffle: bool = True,
        clause_drop_p: float = 0.0,
        lr: float | None = None,
        lambda_: float | None = None,
    ):
        assert Y.ndim == 2, f"Y must be 2D array (samples, outputs), got {Y.ndim}D"
        X = self.config.prepare_X(X)

        iota = self._rng.permutation(X.shape[0]) if shuffle else np.arange(X.shape[0])
        X = X[iota]
        Y = np.asarray(Y, dtype=np.float32, order="C")[iota]

        cfg = self.config
        return self.dev.fit_epoch(X, Y, clause_drop_p, lr=cfg.lr if lr is None else lr, lambda_=cfg.lambda_ if lambda_ is None else lambda_)

    def to_prob(self, class_sums: np.ndarray) -> np.ndarray:
        """No-op, simply returns the activated class sums (``class_sums``) as is."""
        return class_sums
