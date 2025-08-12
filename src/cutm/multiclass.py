from typing import Unpack
import numpy as np
from .base import BaseTM, BaseTMOptArgs, FitOptArgs


class MultiClassTM(BaseTM):
    def __init__(
        self,
        number_of_clauses_per_class: int,
        T: int,
        s: float,
        dim: tuple[int, int, int],
        n_classes: int,
        **opt_args: Unpack[BaseTMOptArgs]
    ):
        super().__init__(
            number_of_clauses_per_class=number_of_clauses_per_class,
            T=T,
            s=s,
            dim=dim,
            n_classes=n_classes,
            **opt_args
        )

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray[tuple[int], np.dtype[np.uint32]],
        is_X_encoded: bool = False,
        **opt_args: Unpack[FitOptArgs]
    ):
        assert Y.ndim == 1, "Y must be 1D array (samples,)"
        assert X.shape[0] == Y.shape[0], "X and Y must have the same number of samples."

        encoded_Y = np.empty((Y.shape[0], self.number_of_outputs), dtype=np.int32)
        for i in range(self.number_of_outputs):
            encoded_Y[:, i] = np.where(Y == i, self.T, -self.T)

        encoded_X = self.encode(X) if not is_X_encoded else X
        self._fit(encoded_X, encoded_Y, **opt_args)

    def score(
        self,
        X: np.ndarray,
        is_X_encoded: bool,
        block_size: int | None = None,
    ):
        encoded_X = self.encode(X) if not is_X_encoded else X
        return self._score_batch(encoded_X, block_size=block_size)

    def predict(
        self,
        X: np.ndarray,
        is_X_encoded: bool = False,
        block_size: int | None = None,
    ):
        class_sums = self.score(X, is_X_encoded, block_size=block_size)
        preds = np.argmax(class_sums, axis=1)
        return preds, class_sums
