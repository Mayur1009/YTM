import numpy as np

from .base import BaseTM


class MultiClassTM(BaseTM):
    """TM for multi-class classification: each sample has exactly one of ``n_classes`` classes.

    The target class is pushed towards ``T_max``, the sampled other classes towards ``T_min``. See
    :class:`~ytm.simple.base.BaseTM` for the parameters.
    """

    def _encode_Y(self, Y: np.ndarray) -> np.ndarray:
        """Class labels of shape (N,) to one-hot."""
        Y = np.asarray(Y)
        assert Y.ndim == 1, "Y must be class labels of shape (N,)"
        return self._to_targets(np.eye(self.n_classes, dtype=np.uint32)[Y])

    def predict(self, X: np.ndarray):
        """Predict the class of each sample: the class with the highest class sum.

        Returns
        -------
        preds : ndarray of shape (N,)
            Predicted class.

        class_sums : ndarray of shape (N, n_classes)
            Weighted vote sum of each class.
        """
        class_sums = self.score(X)
        return np.argmax(class_sums, axis=1), class_sums


class MultiOutputTM(BaseTM):
    """TM for multi-label classification: each sample can have any number of ``n_classes`` labels.

    Each label set to 1 is pushed towards ``T_max``, each sampled label set to 0 towards ``T_min``.
    See :class:`~ytm.simple.base.BaseTM` for the parameters.
    """

    decision_threshold: float = 0.0

    def _encode_Y(self, Y: np.ndarray) -> np.ndarray:
        """Multi-hot labels of shape (N, n_classes)."""
        Y = np.asarray(Y)
        assert Y.ndim == 2 and Y.shape[1] == self.n_classes, "Y must be multi-hot of shape (N, n_classes)"
        return self._to_targets(Y)

    def predict(self, X: np.ndarray):
        """Predict the labels of each sample: 1 where the class sum is above ``decision_threshold``.

        Returns
        -------
        preds : ndarray of shape (N, n_classes), uint32
            Predicted labels, 0 or 1.

        class_sums : ndarray of shape (N, n_classes)
            Weighted vote sum of each label.
        """
        class_sums = self.score(X)
        return (class_sums > self.decision_threshold).astype(np.uint32), class_sums


class BinaryTM(BaseTM):
    """TM for binary classification: one output, each sample is 0 or 1.

    Clauses keep their polarity (``allow_polarity_change=False``), as in the original TM where half
    the clauses vote for and half against the output. See :class:`~ytm.simple.base.BaseTM` for the
    parameters.
    """

    decision_threshold: float = 0.0

    def __init__(self, n_clauses: int, T: float | tuple[float, float], s: float, dim: int | tuple[int, ...], **opt):
        opt.setdefault("allow_polarity_change", False)
        super().__init__(n_clauses, T, s, dim, 1, **opt)

    def _encode_Y(self, Y: np.ndarray) -> np.ndarray:
        """Targets 0 or 1 of shape (N,)."""
        Y = np.asarray(Y)
        assert Y.ndim == 1, "Y must be 0 or 1 of shape (N,)"
        return self._to_targets(Y.reshape(-1, 1))

    def predict(self, X: np.ndarray):
        """Predict each sample: 1 where the class sum is above ``decision_threshold``, else 0.

        Returns
        -------
        preds : ndarray of shape (N,), uint32
            Predicted output, 0 or 1.

        class_sums : ndarray of shape (N, 1)
            Weighted vote sum.
        """
        class_sums = self.score(X)
        return (class_sums[:, 0] > self.decision_threshold).astype(np.uint32), class_sums


class RegressionTM(BaseTM):
    """TM for regression: one real-valued target per sample (Regression TM).

    Targets in ``y_range`` are scaled to class sum targets in ``[0, T]``, and predictions are scaled
    back. Defaults differ from :class:`~ytm.simple.base.BaseTM`: ``negative_clauses=False``.

    Parameters
    ----------
    n_clauses : int
        Number of clauses.

    T : float
        Vote threshold. Class sums are clipped to ``[0, T]``.

    s : float
        Specificity, at least 1.0.

    dim : int or tuple of int
        Input shape: ``D``, ``(H, W)`` or ``(H, W, C)``. Inputs are binary.

    y_range : tuple of (float, float)
        Range ``(lo, hi)`` of the targets.

    **opt
        Other options of :class:`~ytm.simple.base.BaseTM`.
    """

    def __init__(self, n_clauses: int, T: float, s: float, dim: int | tuple[int, ...], y_range: tuple[float, float], **opt):
        assert y_range[0] < y_range[1], f"`y_range[0]` must be < `y_range[1]`, got {y_range}"
        self.y_range = (float(y_range[0]), float(y_range[1]))
        opt.setdefault("negative_clauses", False)
        super().__init__(n_clauses, (0.0, float(T)), s, dim, 1, **opt)

    def _encode_Y(self, Y: np.ndarray) -> np.ndarray:
        """Targets of shape (N,), scaled linearly from ``y_range`` to ``[0, T]``."""
        Y = np.asarray(Y, dtype=np.float32)
        assert Y.ndim == 1, "Y must be 1D array (samples,)"
        lo, hi = self.y_range
        assert np.all((Y >= lo) & (Y <= hi)), f"Y must be in y_range {self.y_range}, got [{Y.min()}, {Y.max()}]"
        return ((Y - lo) / (hi - lo) * self.T_max).astype(np.float32).reshape(-1, 1)

    def predict(self, X: np.ndarray, clip_class_sums: bool = True):
        """Predict the target of each sample: the class sum scaled from ``[0, T]`` back to ``y_range``.

        Parameters
        ----------
        X : array-like of bool, shape (N, *dim) or (N, prod(dim))
            Binary inputs.

        clip_class_sums : bool, default=True
            Clip class sums to ``[0, T]`` before scaling, so predictions stay within ``y_range``.

        Returns
        -------
        preds : ndarray of shape (N,)
            Predicted targets.

        class_sums : ndarray of shape (N, 1)
            Weighted vote sum.
        """
        lo, hi = self.y_range
        class_sums = self.score(X)
        if clip_class_sums:
            class_sums = np.clip(class_sums, self.T_min, self.T_max)
        return class_sums[:, 0] / self.T_max * (hi - lo) + lo, class_sums


__all__ = ["BaseTM", "BinaryTM", "MultiClassTM", "MultiOutputTM", "RegressionTM"]
