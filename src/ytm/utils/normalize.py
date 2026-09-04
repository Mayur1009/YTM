import numpy as np


def norm_asymmetric(arr: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Zero-anchored per-sample normalization.

    Expects ``(N, *feat_shape)``. For each sample independently, negative
    values are scaled by that sample's own ``|min|`` so the most negative
    value maps to ``-1``; positive values are scaled by that sample's own
    ``max`` so the most positive value maps to ``1``. Zero stays zero.
    """
    assert arr.ndim >= 2, "expected (N, *feat_shape)"
    axis = tuple(range(1, arr.ndim))
    out = arr.copy()
    neg_min = out.min(axis=axis, keepdims=True)
    pos_max = out.max(axis=axis, keepdims=True)
    out = np.where(out < 0, out / (-neg_min + eps), out)
    out = np.where(out > 0, out / (pos_max + eps), out)
    return out
