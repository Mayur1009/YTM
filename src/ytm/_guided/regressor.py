from typing import Unpack

import numpy as np

from .backends.act_loss import MSE
from .base import BaseTM
from .config import T_Config


class RegressionTM(BaseTM):
    def __init__(self, n_clauses: int, s: float, dim: int | tuple[int, ...], **opt: Unpack[T_Config]):
        opt.setdefault("act_loss", MSE())
        super().__init__(n_clauses, s, dim, 1, **opt)

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        shuffle: bool = True,
        clause_drop_p: float = 0.0,
        batch_size: int = -1,
        lr: float | None = None,
        lambda_: float | None = None,
    ) -> float:
        assert Y.ndim == 1, "Y must be 1D array (samples,)"
        assert X.shape[0] == Y.shape[0], "X and Y must have the same number of samples"
        return self._fit(X, Y.reshape(-1, 1), shuffle, clause_drop_p, batch_size, lr, lambda_)

    def predict(self, X: np.ndarray, batch_size: int = -1, force_repack: bool = False):
        class_sums = self.score(X, batch_size, force_repack=force_repack)
        return class_sums[:, 0], class_sums
