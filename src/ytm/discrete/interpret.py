import numpy as np
from .base import BaseTM


def wac(tm: BaseTM, X, target_classes=None):
    """
    A template to compute the local interpretation, also called the WAC (Weighted Activated Clauses) for a set of input samples. This should work for image data, but can be adpated to other domains as well.
    """

    # X shoudld be (N, *tm.args.dim)
    X = X.reshape(X.shape[0], *tm.args.dim)
    N = X.shape[0]

    weights = tm.get_weights()  # (n_classes, n_clauses) or (n_clause_banks, n_clauses)
    feature_bounds, position_bounds, is_valid = (
        tm.get_clauses()
    )  # fb: (n_clause_banks, n_clauses, n_raw_patch_feats * 2)
    feature_bounds = feature_bounds.reshape(
        feature_bounds.shape[0], feature_bounds.shape[1], tm.dev.n_raw_patch_feats, 2
    )

    # Convert bounds to single value: lower + upper - feat_min - feat_max
    # Positive = high, negative = low, zero = don't care
    H, W, D = tm.args.dim
    ph, pw = tm.args.patch_dim
    feat_mins = tm.args.feat_mins.reshape(ph, pw, D)
    feat_maxs = tm.args.feat_maxs.reshape(ph, pw, D)
    clause_patterns = (
        (feature_bounds[..., 0] + feature_bounds[..., 1]).reshape(
            feature_bounds.shape[0], feature_bounds.shape[1], ph, pw, D
        )
        - feat_mins
        - feat_maxs
    )

    # Get activations per patch
    patch_outputs = tm.transform_patchwise(X)  # (N, n_clause_banks, n_clauses, n_patches_y, n_patches_x)

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
