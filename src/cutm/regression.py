# WARN: UNTESTED
import numpy as np
from typing import Unpack
from .base import BaseTM, BaseTMOptArgs, FitOptArgs


class RegressionTM(BaseTM):
    def __init__(
        self,
        number_of_clauses,
        T,
        s,
        dim: tuple[int, int, int],
        **opt_args: Unpack[BaseTMOptArgs],
    ):
        opt_args["init_neg_weights"] = False  # Regression does not have negative polarity
        opt_args["negative_polarity"] = False  # Regression does not have negative polarity
        super().__init__(number_of_clauses, T, s, dim=dim, n_classes=1, **opt_args)

    def fit(self, X: np.ndarray, Y, is_X_encoded=False, **opt_args: Unpack[FitOptArgs]):
        X = X.reshape(X.shape[0], X.shape[1], 1)

        self.max_y = np.max(Y)
        self.min_y = np.min(Y)

        encoded_Y = ((Y - self.min_y) / (self.max_y - self.min_y) * self.T).astype(np.int32)
        encoded_X = self.encode(X) if not is_X_encoded else X
        self._fit(encoded_X, encoded_Y, **opt_args)
        return

    def predict(self, X: np.ndarray, is_X_encoded=False, block_size: int | None = None, grid_size: int | None = None):
        encoded_X = self.encode(X) if not is_X_encoded else X
        class_sums = self._score_batch(encoded_X, block_size=block_size, grid_size=grid_size)
        preds = 1.0 * (class_sums[0, :]) * (self.max_y - self.min_y) / (self.T) + self.min_y
        return preds
