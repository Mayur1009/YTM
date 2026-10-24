from typing import Literal

import numpy as np


class Binarizer:
    """Binarize continuous features into thermometer or one-hot encoded bins.

    Transforms features into binary representations suitable for TM input.

    Parameters
    ----------
    bins : int
        Number of inner bins (thresholds). Thermometer output has ``bins``
        bits per feature; one-hot has ``bins + 1``.
    edge_bins : bool, default=False
        If True, add two extra bins covering values below ``min_val`` and
        above ``max_val``.
    method : {"uniform", "quantile"}, default="uniform"
        Strategy for computing thresholds. ``"uniform"`` spaces thresholds
        evenly between min and max. ``"quantile"`` places thresholds at
        equal-frequency percentiles.
    encoding : {"thermometer", "onehot"}, default="thermometer"
        ``"thermometer"`` sets all bits up to the bin index.
        ``"onehot"`` sets exactly one bit.

    Examples
    --------
    >>> binarizer = Binarizer(bins=8)
    >>> binarizer.fit(X_train)
    >>> X_bin = binarizer.transform(X_train)
    >>> X_tr = binarizer.collapse_bins(X_bin)
    """

    def __init__(
        self,
        bins: int,
        edge_bins: bool = False,
        method: Literal["uniform", "quantile"] = "uniform",
        encoding: Literal["thermometer", "onehot"] = "thermometer",
    ):
        self.bins = bins
        self.edge_bins = edge_bins
        self.method = method
        self.encoding = encoding

    def fit(self, X: np.ndarray, thresholds=None, min_val=None, max_val=None) -> None:
        """Compute bin thresholds from data.

        Parameters
        ----------
        X : ndarray of shape (N, ...)
            Input data used to compute thresholds.
        thresholds : array-like of shape (bins,), optional
            Explicit inner thresholds. Ignores ``method`` if provided.
        min_val : float, optional
            Override minimum value. Defaults to ``X.min()``.
        max_val : float, optional
            Override maximum value. Defaults to ``X.max()``.
        """
        assert X.ndim >= 2
        self.min_val = np.min(X) if min_val is None else min_val
        self.max_val = np.max(X) if max_val is None else max_val
        if thresholds is not None:
            assert len(thresholds) == self.bins
            inner = thresholds
        elif self.method == "quantile":
            pct = np.linspace(0, 100, self.bins + 2)[1:-1]
            inner = np.percentile(X, pct)
        elif self.method == "uniform":
            inner = np.linspace(self.min_val, self.max_val, self.bins + 2)[1:-1]
        else:
            raise ValueError(f"Unknown method: {self.method}")

        self.thresholds = np.concatenate(([self.min_val], inner, [self.max_val]))
        assert len(self.thresholds) == self.bins + 2

    def _transform_thermometer(self, X):
        if not self.edge_bins:
            thresh = self.thresholds[1:-1].reshape(*(1,) * len(X.shape), self.bins)
            out_shape = (*X.shape, self.bins)
        else:
            thresh = self.thresholds.reshape(*(1,) * len(X.shape), self.bins + 2)
            out_shape = (*X.shape, self.bins + 2)

        out = np.asarray(X[..., None] >= thresh, dtype=np.uint32).reshape(out_shape)
        return out

    def _transform_onehot(self, X):
        # Out is 1 bin more than thermometer
        out_shape = (*X.shape, self.bins + 1)
        out = np.empty(out_shape, dtype=np.uint32)
        for i in range(len(self.thresholds) - 1):
            out[..., i] = ((X >= self.thresholds[i]) & (X < self.thresholds[i + 1])).astype(np.uint32)

        if self.edge_bins:
            lt_min = (X < self.min_val).astype(np.uint32)
            ge_max = (X >= self.max_val).astype(np.uint32)
            out = np.concatenate((lt_min[..., None], out, ge_max[..., None]), axis=-1)

        return out

    def _collapse_onehot(self, X):
        out = np.argmax(X, axis=-1)
        if self.edge_bins:
            out = out - 1
        return out

    def _collapse_thermometer(self, X):
        out = X.any(axis=-1) * (X.shape[-1] - np.argmax(X[..., ::-1], axis=-1))

        if self.edge_bins:
            out = out - 1

        return out

    def collapse_bins(self, X: np.ndarray) -> np.ndarray:
        """Reverse a binarized array back to integer bin indices.

        Parameters
        ----------
        X : ndarray of shape (N, ..., binned_dim)
            Binarized array produced by :meth:`transform`.

        Returns
        -------
        ndarray of shape (N, ...)
            Integer bin index per feature.
        """

        assert X.ndim >= 3, "Input should be at least a 3D array (N, ... , binned_dim)"

        if self.encoding == "thermometer":
            if self.edge_bins:
                assert X.shape[-1] == self.bins + 2, (
                    f"Last dim should contain {self.bins + 2} elements, got {X.shape[-1]}"
                )
            else:
                assert X.shape[-1] == self.bins, f"Last dim should contain {self.bins} elements, got {X.shape[-1]}"
            return self._collapse_thermometer(X)

        elif self.encoding == "onehot":
            if self.edge_bins:
                assert X.shape[-1] == self.bins + 3, (
                    f"Last dim should contain {self.bins + 3} elements, got {X.shape[-1]}"
                )
            else:
                assert X.shape[-1] == self.bins + 1, (
                    f"Last dim should contain {self.bins + 1} elements, got {X.shape[-1]}"
                )
            return self._collapse_onehot(X)
        else:
            raise ValueError(f"Unknown encoding: {self.encoding}")

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Binarize ``X`` using the fitted thresholds.

        Parameters
        ----------
        X : ndarray of shape (N, ...)
            Input data.

        Returns
        -------
        ndarray of shape (N, ..., bins) or (N, ..., bins+1)
            Binarized output. Last dimension size depends on ``encoding`` and
            ``edge_bins``.
        """
        if self.encoding == "thermometer":
            return self._transform_thermometer(X)
        elif self.encoding == "onehot":
            return self._transform_onehot(X)
        else:
            raise ValueError(f"Unknown encoding: {self.encoding}")

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit thresholds and binarize ``X`` in one step.

        Parameters
        ----------
        X : ndarray of shape (N, ...)
            Input data.

        Returns
        -------
        ndarray
            Binarized output. See :meth:`transform` for shape details.
        """
        self.fit(X)
        return self.transform(X)


