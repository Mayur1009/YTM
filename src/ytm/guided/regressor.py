import numpy as np
from .base import BaseTM


class RegressionTM(BaseTM):
    def __init__(self, n_clauses: int, s: float, dim: tuple, act_fn: str = "identity", loss_fn: str = "mse", **opt_args):
        super().__init__(n_clauses=n_clauses, s=s, dim=dim, n_classes=1, act_fn=act_fn, loss_fn=loss_fn, **opt_args)

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        shuffle: bool = True,
        clause_drop_p: float = 0.0,
        batch_size: int = -1,
        lr: float | None = None,
        lambda_: float | None = None,
    ):
        assert Y.ndim == 1, "Y must be 1D array (samples,)"
        assert X.shape[0] == Y.shape[0], "X and Y must have the same number of samples"
        return self._fit(X, Y.reshape(-1, 1), shuffle, clause_drop_p, batch_size, lr=lr, lambda_=lambda_)

    def predict(self, X: np.ndarray, batch_size: int = -1):
        class_sums = self.score(X, batch_size)
        return class_sums[:, 0], class_sums
