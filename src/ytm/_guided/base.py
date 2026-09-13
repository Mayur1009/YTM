from typing import Unpack

import numpy as np

from .._core.base import BaseTM as CoreBaseTM
from .._core.device_config import DeviceConfig
from .._core.utils import split_device_kwargs
from .backends import make_device
from .config import T_Config, TMConfig


class BaseTM(CoreBaseTM):
    config: TMConfig
    config_cls: type[TMConfig] = TMConfig

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
        super().__init__(make_device(config, DeviceConfig(**dev_kw)))

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
        cfg = self.config
        assert np.prod(X.shape[1:]) == np.prod(cfg._dim), (
            f"Expected input features to match dim {cfg._dim}, but got {X.shape[1:]}"
        )
        assert Y.ndim == 2, f"Y must be 2D array (samples, outputs), got {Y.ndim}D"

        iota = self._rng.permutation(X.shape[0]) if shuffle else np.arange(X.shape[0])
        X = np.asarray(X, dtype=np.int32, order="C")[iota]
        Y = np.asarray(Y, dtype=np.float32, order="C")[iota]

        return self.dev.fit_epoch(X, Y, clause_drop_p, batch_size, lr, lambda_)
