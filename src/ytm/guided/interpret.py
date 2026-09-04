import numpy as np

from ..utils import norm_asymmetric
from .base import BaseTM


def wac(tm: BaseTM, X, target_classes=None, batch_size: int = -1, force_repack: bool = False, normalize: bool = True):
    """
    Local interpretation (WAC): weighted activated clauses for a set of input
    samples. Delegates to the native (CPU/CUDA) :meth:`~BaseTM.wac`, in
    batches of ``batch_size`` so ``X`` need not fit on the device all at once.
    """
    X = X.reshape(X.shape[0], *tm.args.dim)
    N = X.shape[0]
    if batch_size == -1:
        batch_size = N

    if target_classes is None:
        target_classes = np.argmax(tm.score(X), axis=1)
    target_classes = np.asarray(target_classes, dtype=np.int32)

    H, W, D = tm.args.dim
    wac_output = np.zeros((N, H, W, D), dtype=np.float32)

    for i in range(0, N, batch_size):
        batch_end = min(i + batch_size, N)
        wac_output[i:batch_end] = tm.wac(X[i:batch_end], target_classes[i:batch_end], 1, force_repack=force_repack)

    if normalize:
        wac_output = norm_asymmetric(wac_output)

    return wac_output


def wic(tm: BaseTM, force_repack: bool = False, normalize: bool = True):
    """
    Global interpretation (WIC) for a set of input samples. Delegates to the
    native (CPU/CUDA) :meth:`~BaseTM.wic`.
    """
    n_classes = tm.args.n_classes
    H, W, D = tm.args.dim
    wic_output = np.zeros((n_classes, H, W, D), dtype=np.float32)

    for class_id in range(n_classes):
        wic_output[class_id] = tm.wic(class_id, 1, force_repack=force_repack)

    if normalize:
        wic_output = norm_asymmetric(wic_output)

    return wic_output
