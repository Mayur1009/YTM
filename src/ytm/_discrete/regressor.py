from typing import Unpack

import numpy as np

from .base import BaseTM
from .config import RegressionConfig, T_args


class RegressionTM(BaseTM):
    config: RegressionConfig
    config_cls = RegressionConfig

    def __init__(
        self,
        n_clauses: int,
        T: float,
        s: float,
        dim: int | tuple[int, ...],
        y_range: tuple[float, float],
        **opt: Unpack[T_args],
    ):
        opt["negative_clauses"] = False
        super().__init__(n_clauses, (0.0, float(T)), s, dim, 1, y_range=y_range, **opt)

    def _encode_Y(self, Y: np.ndarray) -> np.ndarray:
        lo, hi = self.config.y_range
        return ((Y - lo) / (hi - lo)).astype(np.float32) * self.config._T_max

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        shuffle: bool = True,
        clause_drop_p: float = 0.0,
        batch_size: int = -1,
    ):
        assert Y.ndim == 1, "Y must be 1D array (samples,)"
        assert X.shape[0] == Y.shape[0], "X and Y must have the same number of samples"
        return self._fit(X, Y.reshape(-1, 1), shuffle, clause_drop_p, batch_size)

    def predict(self, X: np.ndarray, clip_class_sums: bool = True, force_repack: bool = False):
        lo, hi = self.config.y_range
        class_sums = self.score(X, force_repack, clip_class_sums)
        preds = (class_sums[:, 0] / self.config._T_max) * (hi - lo) + lo
        return preds, class_sums
