from typing import Literal
import numpy as np


class ThermometerBinarizer:
    def __init__(self, ch: int = 8):
        self.ch = ch

    def binarize_gray(self, X):
        assert X.ndim == 3, "Input should be a 3D array (N, H, W)"
        thresholds = (np.arange(1, self.ch + 1) * 255 / (self.ch + 1)).reshape(1, 1, 1, self.ch)
        out = np.asarray(X[:, :, :, None] >= thresholds, dtype=np.uint32).reshape(
            (X.shape[0], X.shape[1], X.shape[2], self.ch)
        )
        return out

    def binarize_rgb(self, X):
        assert X.ndim == 4, "Input should be a 4D array (N, H, W, C)"
        thresholds = (np.arange(1, self.ch + 1) * 255 / (self.ch + 1)).reshape(1, 1, 1, 1, self.ch)
        out = (X[:, :, :, :, None] >= thresholds).reshape((X.shape[0], X.shape[1], X.shape[2], X.shape[3] * self.ch))
        return np.asarray(out, dtype=np.uint32)


class Binarizer:
    def __init__(
        self,
        bins: int,
        edge_bins: bool = False,
        method: Literal["uniform", "quantile"] = "uniform",
        type: Literal["thermometer", "onehot"] = "thermometer",
    ):
        self.bins = bins
        self.edge_bins = edge_bins
        self.method = method
        self.type = type

    def fit(self, X, thresholds=None, min_val=None, max_val=None):
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
        def f(b):
            nz = np.argwhere(b).ravel()
            z = nz[-1] + 1 if len(nz) > 0 else 0
            return z
        out = np.apply_along_axis(f, -1, X)

        if self.edge_bins:
            out = out - 1

        return out

    def collapse_bins(self, X):

        assert X.ndim >= 3, "Input should be at least a 3D array (N, ... , binned_dim)"

        if self.type == "thermometer":
            if self.edge_bins:
                assert X.shape[-1] == self.bins + 2, (
                    f"Last dim should contain {self.bins + 2} elements, got {X.shape[-1]}"
                )
            else:
                assert X.shape[-1] == self.bins, f"Last dim should contain {self.bins} elements, got {X.shape[-1]}"
            return self._collapse_thermometer(X)

        elif self.type == "onehot":
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
            raise ValueError(f"Unknown type: {self.type}")

    def transform(self, X):
        if self.type == "thermometer":
            return self._transform_thermometer(X)
        elif self.type == "onehot":
            return self._transform_onehot(X)
        else:
            raise ValueError(f"Unknown type: {self.type}")

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


