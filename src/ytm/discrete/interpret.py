import numpy as np
from .base import BaseTM


def wac(tm: BaseTM, X, target_classes=None, batch_size: int = -1, force_repack: bool = False):
    """Compute local interpretation (WAC).

    Each pixel in the output accumulates the weighted clause patterns of all
    positively-weighted clauses that activate on that patch position. Designed
    for image data; adapt ``clause_patterns`` for other domains.

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
        Batch size for :meth:`~BaseTM.transform_patchwise`. ``-1`` processes
        all at once.
    force_repack : bool, default=False
        Force clause repacking even if a cached result exists.

    Returns
    -------
    ndarray of shape (N, H, W, D)
        Positive values indicate features voting high, negative values low,
        zero means no contribution.
    """

    # X shoudld be (N, *tm.args.dim)
    X = X.reshape(X.shape[0], *tm.args.dim)
    N = X.shape[0]

    weights = tm.get_weights()  # (n_classes, n_clauses) or (n_clause_banks, n_clauses)
    feature_bounds, position_bounds, is_valid = tm.get_clauses(force_repack)  # fb: (n_clause_banks, n_clauses, n_raw_patch_feats * 2)
    feature_bounds = feature_bounds.reshape(feature_bounds.shape[0], feature_bounds.shape[1], tm.dev.n_raw_patch_feats, 2)

    # Convert bounds to single value: lower + upper - feat_min - feat_max
    # Positive = high, negative = low, zero = don't care
    H, W, D = tm.args.dim
    ph, pw = tm.args.patch_dim
    feat_mins = tm.args.feat_mins.reshape(ph, pw, D)
    feat_maxs = tm.args.feat_maxs.reshape(ph, pw, D)
    clause_patterns = (
        (feature_bounds[..., 0] + feature_bounds[..., 1]).reshape(feature_bounds.shape[0], feature_bounds.shape[1], ph, pw, D)
        - feat_mins
        - feat_maxs
    )

    # Get activations per patch
    patch_outputs = tm.transform_patchwise(X, batch_size)  # (N, n_clause_banks, n_clauses, n_patches_y, n_patches_x)

    # Determine target class per sample if not specified
    if target_classes is None:
        target_classes = np.argmax(tm.score(X), axis=1)

    sy, sx = tm.args.stride
    n_clause_banks = tm.dev.n_clause_banks
    n_clauses = tm.args.n_clauses
    wac_output = np.zeros((N, H, W, D), dtype=np.float32)

    for e in range(N):
        tc = target_classes[e]
        for bank in range(n_clause_banks):
            for ci in range(n_clauses):
                w = weights[tc, ci] if tm.args.coalesced else weights[bank, ci]
                if w > 0:
                    active = np.argwhere(patch_outputs[e, bank, ci] > 0)
                    if len(active) == 0:
                        continue

                    cp = clause_patterns[bank, ci]  # (ph, pw, D)

                    for py, px in active:
                        y0 = py * sy
                        x0 = px * sx
                        wac_output[e, y0 : y0 + ph, x0 : x0 + pw, :] += cp * w

    return wac_output


def wic(tm: BaseTM, force_repack: bool = False):
    """Compute global interpretation(WIC).

    Accumulates clause patterns weighted by clause weight and normalized
    patch-vote counts across all valid, positively-weighted clauses.
    Designed for image data; adapt ``clause_patterns`` for other domains.

    Parameters
    ----------
    tm : BaseTM
        Trained TM model. Must have been trained with
        ``track_patch_weights=True`` (default).
    force_repack : bool, default=False
        Force clause repacking even if a cached result exists.

    Returns
    -------
    ndarray of shape (n_classes, H, W, D)
        Per-class global interpretation map. Positive values indicate
        features strongly associated with the class.

    Raises
    ------
    ValueError
        If ``tm`` was trained with ``track_patch_weights=False``.
    """
    if not tm.args.track_patch_weights:
        raise ValueError("The model should be trained with `track_patch_weights=True` to get the global interpretation.")

    # Get weights, clauses, and patch_weights
    weights = tm.get_weights()  # (n_classes, n_clauses) or (n_clause_banks, n_clauses)
    patch_weights = tm.get_patch_weights().astype(np.float32)  # (n_clause_banks, n_clauses, n_patches_y, n_patches_x)
    patch_weights = patch_weights / (patch_weights.max(axis=(-2, -1), keepdims=True) + 1e-7)
    feature_bounds, position_bounds, is_valid = tm.get_clauses(force_repack)  # fb: (n_clause_banks, n_clauses, n_raw_patch_feats * 2)
    feature_bounds = feature_bounds.reshape(feature_bounds.shape[0], feature_bounds.shape[1], tm.dev.n_raw_patch_feats, 2)

    # Convert bounds to single value: lower + upper - feat_min - feat_max
    # Positive = high, negative = low, zero = don't care
    H, W, D = tm.args.dim
    ph, pw = tm.args.patch_dim
    feat_mins = tm.args.feat_mins.reshape(ph, pw, D)
    feat_maxs = tm.args.feat_maxs.reshape(ph, pw, D)
    clause_patterns = (
        (feature_bounds[..., 0] + feature_bounds[..., 1]).reshape(feature_bounds.shape[0], feature_bounds.shape[1], ph, pw, D)
        - feat_mins
        - feat_maxs
    )

    sy, sx = tm.args.stride
    n_classes = tm.args.n_classes
    n_clauses = tm.args.n_clauses
    wic_output = np.zeros((n_classes, H, W, D), dtype=np.float32)

    for class_id in range(n_classes):
        bank = 0 if tm.args.coalesced else class_id
        for ci in range(n_clauses):
            if not is_valid[bank, ci]:
                continue

            w = weights[class_id, ci] if tm.args.coalesced else weights[bank, ci]
            if w <= 0:
                continue

            cp = clause_patterns[bank, ci]  # (ph, pw, D)

            if position_bounds is not None:
                min_y, max_y, min_x, max_x = position_bounds[bank, ci]
            else:
                min_y, min_x = 0, 0
                max_y, max_x = tm.dev.n_patches_y - 1, tm.dev.n_patches_x - 1

            clause_pw = np.zeros((H, W, D))
            for py in range(int(min_y), int(max_y) + 1):
                for px in range(int(min_x), int(max_x) + 1):
                    pw_val = patch_weights[bank, ci, py, px]
                    if pw_val > 0:
                        y0, x0 = py * sy, px * sx
                        clause_pw[y0 : y0 + ph, x0 : x0 + pw, :] += cp * pw_val

            wic_output[class_id] += w * clause_pw

    return wic_output
