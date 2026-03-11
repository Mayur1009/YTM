import numpy as np
from .base import BaseTM


class MultiOutputTM(BaseTM):
    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        shuffle: bool = True,
        clause_drop_p: float = 0.0,
        batch_size: int = -1,
    ):
        assert Y.ndim == 2, f"Y must be 2D array (samples, outputs), got {Y.ndim}D"
        assert X.shape[0] == Y.shape[0], "X and Y must have the same number of samples"

        return self._fit(X, Y, shuffle, clause_drop_p, batch_size)

    def predict(self, X: np.ndarray, batch_size: int = -1):
        class_sums = self.score(X, batch_size)
        preds = (class_sums >= 0).astype(np.uint32)
        return preds, class_sums
