import numpy as np
from .base import BaseTM


class RegressionTM(BaseTM):
    def __init__(self, n_clauses: int, T: float, s: float, dim: tuple, y_range: tuple, **opt_args):
        assert y_range[0] < y_range[1], "y_range[0] must be < y_range[1]"
        self.y_range = y_range
        opt_args["negative_clauses"] = False
        q = opt_args.get("q", 1.0)
        if q > 1.0:
            print(f"Warning: Got q = {q}, q > 1.0 not supported for regression, setting q=1.0")
        opt_args["q"] = 1.0
        super().__init__(n_clauses=n_clauses, T=(0.0, float(T)), s=s, dim=dim, n_classes=1, **opt_args)

    def _target_sampling(self, Y: np.ndarray) -> np.ndarray:
        return np.copy(Y).astype(np.float32) * self.args.T_max

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
        encoded_Y = ((Y - self.y_range[0]) / (self.y_range[1] - self.y_range[0])).reshape(-1, 1).astype(np.float32)
        return self._fit(X, encoded_Y, shuffle, clause_drop_p, batch_size)

    def predict(self, X: np.ndarray, batch_size: int = -1, clip_class_sums: bool = True):
        class_sums = self.score(X, batch_size, clip_class_sums)
        preds = (class_sums[:, 0] / self.args.T_max) * (self.y_range[1] - self.y_range[0]) + self.y_range[0]
        return preds, class_sums
