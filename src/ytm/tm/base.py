from typing import cast

import numpy as np

from .._core.base import BaseTM as CoreBaseTM
from .._core.device_config import DeviceConfig, split_device_kwargs
from .config import TMConfig


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
        T: float | tuple[float, float],
        s: float,
        dim: int | tuple[int, ...],
        n_classes: int,
        **opt,
    ):
        cfg_kw, dev_kw = split_device_kwargs(dict(opt))
        config = self._config_cls()(n_clauses=n_clauses, T=T, s=s, dim=dim, n_classes=n_classes, **cfg_kw)
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
        label_sampling: bool = False,
    ) -> None:
        X = self.config.prepare_X(X)

        iota = self._rng.permutation(X.shape[0]) if shuffle else np.arange(X.shape[0])
        X = X[iota]

        encoded_Y = np.asarray(self._encode_Y(Y[iota]), dtype=np.float32, order="C")
        label_probs = np.asarray(self._label_sampler(encoded_Y, label_sampling), dtype=np.float32, order="C")

        self.dev.fit_epoch(X, encoded_Y, clause_drop_p, label_probs=label_probs)

    def _encode_Y(self, Y: np.ndarray) -> np.ndarray:
        return np.where(Y == 1, self.config._T_max, self.config._T_min).astype(np.float32)

    def _label_sampler(self, encoded_Y: np.ndarray, label_sampling: bool) -> np.ndarray:
        cfg = self.config
        if cfg.n_classes == 1:
            return np.ones_like(encoded_Y, dtype=np.float32)

        label_probs = np.full_like(encoded_Y, cfg.q / (cfg.n_classes - 1), dtype=np.float32)
        label_probs[encoded_Y == cfg._T_max] = 1.0
        return label_probs

    def score(self, X: np.ndarray, force_repack: bool = False, clip_class_sums: bool = False) -> np.ndarray:
        """Calculate class sums for the given X.

        Parameters
        ----------
        X : array-like of int, shape (N, *dim) or (N, prod(dim))
            Inputs, each feature within ``[feat_mins, feat_maxs]``.

        force_repack : bool, default=False
            Repack every clause first, not only the changed ones.

        clip_class_sums : bool, default=False
            Clip the class sums to ``[T_min, T_max]``.

        Returns
        -------
        ndarray of shape (N, n_classes), float32
            Class sums.
        """
        class_sums = super().score(X, force_repack)
        if clip_class_sums:
            class_sums = np.clip(class_sums, self.config._T_min, self.config._T_max)
        return class_sums

    def to_prob(self, class_sums: np.ndarray) -> np.ndarray:
        """Convert the class sums to probabilities.

        ``(clip(class_sums, T_min, T_max) - T_min) / (T_max - T_min)``
        """
        t_min, t_max = self.config._T_min, self.config._T_max
        return (np.clip(class_sums, t_min, t_max) - t_min) / (t_max - t_min)
