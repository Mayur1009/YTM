import numpy as np

from ..utils import norm_asymmetric
from .base import BaseTM


def wac(tm: BaseTM, X, target_classes=None, batch_size: int = -1, force_repack: bool = False, normalize: bool = True):
    """Compute local interpretation (WAC).

    Each pixel in the output accumulates the weighted clause patterns of all
    positively-weighted clauses that activate on that patch position. Designed
    for image data; adapt ``clause_patterns`` for other domains.

    Delegates to the native (CPU/CUDA) :meth:`~BaseTM.wac` implementation.

    Parameters
    ----------
    tm : BaseTM
        Trained TM model. Must have been trained with a patch-based ``dim``
        and ``patch_dim``.
    X : ndarray of shape (N, ...)
        Input samples. Reshaped internally to ``(N, *tm.args.dim)``.
    target_classes : array-like of int of shape (N,), optional
        Target class index per sample. If ``None``, uses ``argmax`` of
        ``score(X)`` (i.e., the predicted class).
    batch_size : int, default=-1
        Number of samples processed per batch. ``-1`` processes all samples
        at once. Use a smaller value when ``X`` is too large to copy to the
        device in one shot.
    force_repack : bool, default=False
        Force clause repacking even if a cached result exists.
    normalize : bool, default=True
        If ``True``, normalize each sample's map independently via
        :func:`ytm.utils.norm_asymmetric`.

    Returns
    -------
    ndarray of shape (N, H, W, D)
        Positive values indicate features voting high, negative values low,
        zero means no contribution.
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
    """Compute global interpretation(WIC).

    Accumulates clause patterns weighted by clause weight and normalized
    patch-vote counts across all valid, positively-weighted clauses.
    Designed for image data; adapt ``clause_patterns`` for other domains.

    Delegates to the native (CPU/CUDA) :meth:`~BaseTM.wic` implementation.

    Parameters
    ----------
    tm : BaseTM
        Trained TM model. Must have been trained with
        ``track_patch_weights=True`` (default).
    force_repack : bool, default=False
        Force clause repacking even if a cached result exists.
    normalize : bool, default=True
        If ``True``, normalize each class map independently via
        :func:`ytm.utils.norm_asymmetric`.

    Returns
    -------
    ndarray of shape (n_classes, H, W, D)
        Per-class global interpretation map. Positive values indicate
        features strongly associated with the class.

    Raises
    ------
    ValueError
        If ``tm`` was trained with ``track_patch_weights=False`` and this is
        a convolutional (patch-based) model.
    """
    n_classes = tm.args.n_classes
    H, W, D = tm.args.dim
    wic_output = np.zeros((n_classes, H, W, D), dtype=np.float32)

    for class_id in range(n_classes):
        wic_output[class_id] = tm.wic(class_id, 1, force_repack=force_repack)

    if normalize:
        wic_output = norm_asymmetric(wic_output)

    return wic_output
