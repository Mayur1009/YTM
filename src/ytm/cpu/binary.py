from typing import Unpack
import numpy as np
from .base import BaseTM, BaseTMOptArgs, FitOptArgs


class BinaryTM(BaseTM):
    def __init__(
        self,
        number_of_clauses_per_class: int,
        T: int,
        s: float,
        dim: tuple[int, int, int],
        **opt_args: Unpack[BaseTMOptArgs],
    ):
        super().__init__(
            number_of_clauses_per_class=number_of_clauses_per_class, T=T, s=s, dim=dim, n_classes=1, **opt_args
        )

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray[tuple[int], np.dtype[np.uint32]],
        is_X_encoded: bool = False,
        **opt_args: Unpack[FitOptArgs],
    ):
        assert Y.ndim == 1, "Y must be 1D array (samples,)"
        assert np.min(Y) >= 0 and np.max(Y) <= 1, "Y must contain binary values (0 and 1)."
        assert X.shape[0] == Y.shape[0], "X and Y must have the same number of samples."

        encoded_Y = np.where(Y == 1, self.T, -self.T).reshape(-1, 1)

        encoded_X = self.encode(X) if not is_X_encoded else X
        return self._fit(encoded_X, encoded_Y, **opt_args)

    def score(
        self,
        X: np.ndarray,
        is_X_encoded: bool,
    ):
        encoded_X = self.encode(X) if not is_X_encoded else X
        return self._score_batch(encoded_X).squeeze()

    def predict(
        self,
        X: np.ndarray,
        is_X_encoded: bool = False,
    ):
        class_sums = self.score(X, is_X_encoded)
        preds = (class_sums >= 0).squeeze().astype(np.uint32)
        return preds, class_sums
