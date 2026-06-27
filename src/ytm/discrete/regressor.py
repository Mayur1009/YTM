import numpy as np
from .base import BaseTM


class RegressionTM(BaseTM):
    """Tsetlin Machine for regression.

    ``negative_clauses`` is forced to ``False`` and ``q`` is clamped to 1.0.

    Parameters
    ----------
    y_range : tuple of (float, float)
        ``(min, max)`` of the target variable. Used for normalization.
        Must satisfy ``y_range[0] < y_range[1]``.

    See :class:`~ytm.discrete.base.BaseTM` for all other constructor parameters.
    """

    def __init__(self, n_clauses: int, T: float, s: float, dim: tuple, y_range: tuple, **opt_args):
        assert y_range[0] < y_range[1], "y_range[0] must be < y_range[1]"
        self.y_range = y_range
        opt_args["negative_clauses"] = False
        q = opt_args.get("q", 1.0)
        if q > 1.0:
            print(f"Warning: Got q = {q}, q > 1.0 not supported for regression, setting q=1.0")
        opt_args["q"] = 1.0
        super().__init__(n_clauses=n_clauses, T=(0.0, float(T)), s=s, dim=dim, n_classes=1, **opt_args)

    def _encode_Y(self, Y: np.ndarray) -> np.ndarray:
        encoded_Y = ((Y - self.y_range[0]) / (self.y_range[1] - self.y_range[0])).astype(np.float32)
        return encoded_Y * self.args.T_max

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        shuffle: bool = True,
        clause_drop_p: float = 0.0,
        batch_size: int = -1,
    ):
        """Train for one epoch.

        Parameters
        ----------
        X : ndarray of shape (N, ...)
            Input samples. Feature count must match ``dim``.
        Y : ndarray of shape (N,), dtype float
            Continuous targets. Should fall within ``y_range``.
        shuffle : bool, default=True
            Shuffle samples before each epoch.
        clause_drop_p : float, default=0.0
            Probability of randomly dropping a clause.
        batch_size : int, default=-1
            Process samples in batches. ``-1`` processes all at once.
            Use when X cannot fit in GPU memory.
        """
        assert Y.ndim == 1, "Y must be 1D array (samples,)"
        assert X.shape[0] == Y.shape[0], "X and Y must have the same number of samples"
        return self._fit(X, Y.reshape(-1, 1), shuffle, clause_drop_p, batch_size)

    def predict(self, X: np.ndarray, batch_size: int = -1, clip_class_sums: bool = True):
        """Predict.

        Parameters
        ----------
        X : ndarray of shape (N, ...)
            Input samples.
        batch_size : int, default=-1
            Inference batch size. ``-1`` processes all at once.
        clip_class_sums : bool, default=True
            Clip votes to ``[0, T]`` before denormalization.

        Returns
        -------
        preds : ndarray of shape (N,), dtype float
            Predicted values in ``y_range``.
        class_sums : ndarray of shape (N, 1)
            Vote sums before denormalization.
        """
        class_sums = self.score(X, batch_size, clip_class_sums)
        preds = (class_sums[:, 0] / self.args.T_max) * (self.y_range[1] - self.y_range[0]) + self.y_range[0]
        return preds, class_sums
