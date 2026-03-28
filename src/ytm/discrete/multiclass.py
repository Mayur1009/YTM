import numpy as np
from .base import BaseTM


class MultiClassTM(BaseTM):
    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        shuffle: bool = True,
        clause_drop_p: float = 0.0,
        batch_size: int = -1,
    ):
        assert Y.ndim == 1, "Y must be 1D array (samples,)"
        assert X.shape[0] == Y.shape[0], "X and Y must have the same number of samples."

        encoded_Y = np.empty((Y.shape[0], self.args.n_classes), dtype=np.int8)
        for i in range(self.args.n_classes):
            encoded_Y[:, i] = np.where(Y == i, 1, 0)

        return self._fit(X, encoded_Y, shuffle, clause_drop_p, batch_size)

    def predict(self, X: np.ndarray, batch_size: int = -1):
        class_sums = self.score(X, batch_size)
        preds = np.argmax(class_sums, axis=1)
        return preds, class_sums
