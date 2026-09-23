import numpy as np
from .base import BaseTM


class MultiClassTM(BaseTM):
    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray[tuple[int], np.dtype[np.uint32]],
        is_X_encoded: bool = False,
        shuffle: bool = True,
        clause_drop_p: float = 0.0,
    ):
        assert Y.ndim == 1, "Y must be 1D array (samples,)"
        assert X.shape[0] == Y.shape[0], "X and Y must have the same number of samples."

        encoded_Y = np.empty((Y.shape[0], self.args.n_classes), dtype=np.int8)
        for i in range(self.args.n_classes):
            encoded_Y[:, i] = np.where(Y == i, 1, 0)

        encoded_X = self.encode(X) if not is_X_encoded else X
        return self._fit(encoded_X, encoded_Y, shuffle, clause_drop_p)

    def predict(self, X: np.ndarray, is_X_encoded: bool = False, clip_class_sums: bool = False):
        encoded_X = self.encode(X) if not is_X_encoded else X
        class_sums = self.score(encoded_X, clip_class_sums)
        preds = np.argmax(class_sums, axis=1)
        return preds, class_sums
