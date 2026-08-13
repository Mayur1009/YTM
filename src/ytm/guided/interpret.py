import numpy as np
from .base import BaseTM


def wac(tm: BaseTM, X, target_classes=None, batch_size: int = -1, force_repack: bool = False, normalize: bool = True):
    """
    A template to compute the local interpretation, also called the WAC (Weighted Activated Clauses) for a set of input samples. This should work for image data, but can be adpated to other domains as well.
    """

    # X shoudld be (N, *tm.args.dim)
    X = X.reshape(X.shape[0], *tm.args.dim)
    N = X.shape[0]

    weights = tm.get_weights()  # (n_classes, n_clauses) or (n_clause_banks, n_clauses)
    feature_bounds, _, _ = tm.get_clauses(force_repack)  # fb: (n_clause_banks, n_clauses, n_raw_patch_feats * 2)
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

    if normalize:
        for e in range(N):
            img = wac_output[e]
            if img.min() < 0:
                img[img < 0] = img[img < 0] / (-1 * img[img < 0].min() + 1e-7)
            if img.max() > 0:
                img[img > 0] = img[img > 0] / (img[img > 0].max() + 1e-7)

    return wac_output


def wic(tm: BaseTM, force_repack: bool = False, normalize: bool = True):
    """
    A template to compute the global interpretation, also called the WIC for a set of input samples. This should work for image data, but can be adpated to other domains as well.
    """
    if not tm.args.track_patch_weights:
        raise ValueError("The model should be trained with `track_patch_weights=True` to get the global interpretation.")

    # Get weights, clauses, and patch_weights
    weights = tm.get_weights()  # (n_classes, n_clauses) or (n_clause_banks, n_clauses)
    if tm.dev.n_patches_y == 1 and tm.dev.n_patches_x == 1:
        patch_weights = np.ones((tm.dev.n_clause_banks, tm.args.n_clauses, 1, 1), dtype=np.float32)
    else:
        patch_weights = tm.get_patch_weights().astype(np.float32)  # (n_clause_banks, n_clauses, n_patches_y, n_patches_x)
        patch_weights = patch_weights / (patch_weights.max(axis=(-2, -1), keepdims=True) + 1e-7)
    feature_bounds, position_bounds, clause_density = tm.get_clauses(force_repack)  # fb: (n_clause_banks, n_clauses, n_raw_patch_feats * 2)
    feature_bounds = feature_bounds.reshape(feature_bounds.shape[0], feature_bounds.shape[1], tm.dev.n_raw_patch_feats, 2)

    is_valid = clause_density != -1

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

    if normalize:
        for c in range(n_classes):
            img = wic_output[c]
            if img.min() < 0:
                img[img < 0] = img[img < 0] / (-1 * img[img < 0].min() + 1e-7)
            if img.max() > 0:
                img[img > 0] = img[img > 0] / (img[img > 0].max() + 1e-7)

    return wic_output
