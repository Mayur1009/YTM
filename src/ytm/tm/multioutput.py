import numpy as np
from .base import BaseTM


class MultiOutputTM(BaseTM):
    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray[tuple[int, int], np.dtype[np.int8]],
        is_X_encoded: bool = False,
        shuffle: bool = True,
        clause_drop_p: float = 0.0,
    ):
        assert Y.ndim == 2, f"Y must be 2D array (samples, outputs), got {Y.ndim}D"
        assert X.shape[0] == Y.shape[0], "X and Y must have the same number of samples"

        encoded_X = self.encode(X) if not is_X_encoded else X
        return self._fit(encoded_X, Y, shuffle, clause_drop_p)

    def predict(self, X: np.ndarray, is_X_encoded: bool = False, clip_class_sums: bool = False):
        encoded_X = self.encode(X) if not is_X_encoded else X
        class_sums = self.score(encoded_X, clip_class_sums)
        preds = (class_sums >= 0).astype(np.uint32)
        return preds, class_sums
