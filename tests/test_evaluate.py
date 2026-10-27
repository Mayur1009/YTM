import numpy as np
import pytest

from .support import discrete, set_clauses, set_ta_states, set_weights


def _weights(shape, cells):
    w = np.zeros(shape, dtype=np.float32)
    for (cls, clause), v in cells.items():
        w[cls, clause] = v
    return w


def test_clause_fires_only_when_every_included_literal_holds(device):
    """A wrong literal index or a negation offset on the wrong half would fire on the wrong inputs."""
    tm = discrete("multi", device)
    set_clauses(tm, {0: [0, 5]})  # x0 AND NOT x1; clauses 1-3 stay empty with weight 0
    set_weights(tm, _weights((2, 4), {(0, 0): 3, (1, 0): -2}))
    X = np.array([[1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [1, 0, 1, 1]])
    assert np.array_equal(tm.score(X, force_repack=True), [[3, -2], [0, 0], [0, 0], [3, -2]])


def test_position_contradiction_never_fires(device):
    """A position literal and its negation pack as has_contra with clause_len 0; without has_contra it would fire as an empty clause."""
    tm = discrete("binary", device, n_clauses=2, dim=(1, 4, 1), patch_dim=(1, 2))
    set_clauses(tm, {0: [1, 5]})  # px > 1 AND px <= 1
    set_weights(tm, [[3, 0]])
    X = np.array([[0, 1, 0, 0], [1, 1, 1, 1]]).reshape(2, 1, 4, 1)
    assert np.array_equal(tm.score(X, force_repack=True)[:, 0], [0, 0])


def test_empty_clause_always_fires_and_votes_add_up(device):
    """`clause_len == 0` short-circuits to 1; two firing clauses must sum per class with signs."""
    tm = discrete("multi", device)
    set_clauses(tm, {})
    set_weights(tm, _weights((2, 4), {(0, 0): 2, (1, 0): -1, (0, 1): -5, (1, 1): 4}))
    assert np.array_equal(tm.score(np.zeros((3, 4), dtype=int), force_repack=True), np.tile([-3, 3], (3, 1)))


def test_include_threshold_is_exactly_include_state(device):
    """Off-by-one in `is_included`: state include_state - 1 must be excluded, include_state included."""
    tm = discrete("multi", device)
    w = _weights((2, 4), {(0, 0): 3})
    set_weights(tm, w)
    X = np.array([[0, 0, 0, 0], [1, 0, 0, 0]])
    set_clauses(tm, {})  # everything at include_state - 1: clause 0 is empty and fires for both rows
    assert np.array_equal(tm.score(X, force_repack=True)[:, 0], [3, 3])
    set_clauses(tm, {0: [0]})  # literal 0 at include_state: clause 0 now needs x0
    assert np.array_equal(tm.score(X, force_repack=True)[:, 0], [0, 3])


def test_uncoalesced_classes_only_sum_their_own_bank(device):
    """`weight_offset` / `LOOP_CLASS_ID`: with separate banks each class must see only its own clauses."""
    tm = discrete("multi", device, n_clauses=2, coalesced=False)
    set_clauses(tm, {})
    set_weights(tm, [[1, 1], [2, 3]])
    assert np.array_equal(tm.score(np.zeros((1, 4), dtype=int), force_repack=True), [[2, 5]])


def test_thermometer_literals_bound_the_value_to_an_interval(device):
    """Literal 1 is `x0 > 1`, negated literal 2 (index 8) is `x0 <= 2`: together x0 == 2. Wrong bounds fire on 1 or 3."""
    tm = discrete("binary", device, n_clauses=2, dim=(2, 1, 1), feat_maxs=3)
    set_clauses(tm, {0: [1, 8]})
    set_weights(tm, [[3, 0]])
    X = np.array([[v, 0] for v in range(4)])
    assert np.array_equal(tm.score(X, force_repack=True)[:, 0], [0, 0, 3, 0])


def test_conv_clause_fires_if_any_window_matches(device):
    """Windows start at x0, x1, x2; the clause wants the window's first pixel set. Only-second-pixel must not count."""
    tm = discrete("binary", device, n_clauses=2, dim=(1, 4, 1), patch_dim=(1, 2))
    set_clauses(tm, {0: [2]})
    set_weights(tm, [[3, 0]])
    X = np.array([[0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]]).reshape(3, 1, 4, 1)
    assert np.array_equal(tm.score(X, force_repack=True)[:, 0], [3, 0, 3])


def test_position_literal_restricts_which_windows_count(device):
    """Literal 1 on the x axis is `px > 1`, so only the window at px=2 may match; earlier windows must be ignored."""
    tm = discrete("binary", device, n_clauses=2, dim=(1, 4, 1), patch_dim=(1, 2))
    set_clauses(tm, {0: [1, 2]})
    set_weights(tm, [[3, 0]])
    X = np.array([[1, 1, 0, 1], [0, 0, 1, 0], [1, 1, 1, 0]]).reshape(3, 1, 4, 1)
    assert np.array_equal(tm.score(X, force_repack=True)[:, 0], [0, 3, 3])


def _reference_score(tm, X):
    cfg = tm.config
    H, W, C = cfg._dim
    h, w = cfg._patch_dim
    sy, sx = cfg._stride
    ny, nx = cfg._n_patches_y, cfg._n_patches_x
    included = tm.get_ta_states() >= cfg._include_state
    weights = tm.get_weights()
    X = X.reshape(-1, H, W, C)

    lits = []
    for py in range(ny):
        for px in range(nx):
            feats = X[:, py * sy : py * sy + h, px * sx : px * sx + w, :].reshape(len(X), -1)
            if cfg._n_position_feats:
                pos = np.concatenate([np.arange(ny - 1) < py, np.arange(nx - 1) < px])
                feats = np.concatenate([np.broadcast_to(pos, (len(X), len(pos))), feats], axis=1)
            lits.append(np.concatenate([feats, 1 - feats], axis=1) if cfg.negated_literals else feats)
    lits = np.stack(lits, axis=1).astype(bool)

    sums = np.zeros((len(X), cfg.n_classes), dtype=np.float32)
    for c in range(cfg.n_classes):
        bank = included[0 if cfg.coalesced else c]
        fires = ~(bank[None, None] & ~lits[:, :, None]).any(axis=3)
        sums[:, c] = fires.any(axis=1).astype(np.float32) @ weights[c]
    return sums


@pytest.mark.parametrize(
    "dim, patch_dim, stride",
    [((200, 1, 1), None, (1, 1)), ((12, 12, 2), (6, 6), (2, 1)), ((9, 10, 1), (3, 4), (1, 2))],
    ids=["flat_4_segments", "conv_depth2_stride", "conv_small"],
)
@pytest.mark.parametrize("coalesced", [True, False], ids=["coalesced", "banks"])
@pytest.mark.parametrize("negated_literals", [True, False], ids=["negated", "plain"])
def test_binary_features_match_a_reference_on_random_clauses(device, dim, patch_dim, stride, coalesced, negated_literals):
    """Random sparse clauses on binary inputs: every segment, patch, stride and position bound must agree with a numpy reference."""
    tm = discrete(
        "multi",
        device,
        n_clauses=64,
        n_classes=3,
        dim=dim,
        patch_dim=patch_dim,
        stride=stride,
        coalesced=coalesced,
        negated_literals=negated_literals,
    )
    cfg = tm.config
    rng = np.random.default_rng(0)
    states = np.where(rng.random((cfg._total_clauses, cfg._n_literals)) < 0.01, cfg._include_state, cfg._include_state - 1)
    set_ta_states(tm, states)
    set_weights(tm, rng.integers(-5, 6, size=(cfg.n_classes, cfg._n_clauses)))

    X = (rng.random((50, int(np.prod(dim)))) < 0.3).astype(np.uint8)
    ref = _reference_score(tm, X)
    assert np.unique(ref).size > 3
    np.testing.assert_array_equal(tm.score(X.reshape(50, *dim), force_repack=True), ref)
