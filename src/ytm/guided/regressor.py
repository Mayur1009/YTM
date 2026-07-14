import numpy as np
from .base import BaseTM


class RegressionTM(BaseTM):
    def __init__(self, n_clauses: int, s: float, dim: tuple, y_range: tuple, **opt_args):
        assert y_range[0] < y_range[1], "y_range[0] must be < y_range[1]"
        self.y_range = y_range
        opt_args["negative_clauses"] = False
        q = opt_args.get("q", 1.0)
        if q > 1.0:
            print(f"Warning: Got q = {q}, q > 1.0 not supported for regression, setting q=1.0")
        opt_args["q"] = 1.0
        super().__init__(n_clauses=n_clauses, s=s, dim=dim, n_classes=1, **opt_args)

    def _encode_Y(self, Y: np.ndarray) -> np.ndarray:
        return ((Y - self.y_range[0]) / (self.y_range[1] - self.y_range[0])).astype(np.float32)

    def _calc_label_probs(self, encoded_Y: np.ndarray) -> np.ndarray:
        return np.ones_like(encoded_Y, dtype=np.float32)

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

    def predict(self, X: np.ndarray, batch_size: int = -1):
        class_sums = self.score(X, batch_size)
        preds = class_sums[:, 0] * (self.y_range[1] - self.y_range[0]) + self.y_range[0]
        return preds, class_sums
