import numpy as np
import pytest

from ytm._core.config import BaseTMConfig

from . import common_harness as ch

# depth 3 and stride 2 with a non square patch, so a transposed or dropped term cannot hide
CONV = BaseTMConfig(n_clauses=4, s=10.0, dim=(9, 7, 3), patch_dim=(3, 2), stride=(2, 1), n_classes=2, feat_maxs=5, seed=1)
FLAT = BaseTMConfig(n_clauses=4, s=10.0, dim=(4, 4, 1), n_classes=2, feat_maxs=3, seed=1)


def sample(cfg: BaseTMConfig, seed: int = 0) -> np.ndarray:
    """Distinct values everywhere, so reading the wrong cell gives the wrong answer."""
    h, w, d = cfg._dim
    return np.arange(h * w * d, dtype=np.int32).reshape(h, w, d)


class TestClip:
    @pytest.mark.parametrize(
        "v, expected",
        [(-5.0, -1.0), (-1.0, -1.0), (0.0, 0.0), (3.0, 3.0), (9.0, 3.0)],
        ids=["below", "at lo", "inside", "at hi", "above"],
    )
    def test_bounds_are_inclusive(self, v, expected):
        """`calc_update_prob` clips votes at exactly T_MIN and T_MAX, where the probability is meant
        to reach zero. An exclusive bound there would leave a residual update forever."""
        assert ch.clip(FLAT, v, -1.0, 3.0) == expected


class TestIsIncluded:
    @pytest.mark.parametrize("offset, included", [(-1, False), (0, True), (1, True)])
    def test_the_include_threshold_is_inclusive(self, offset, included):
        """`>` instead of `>=` would shift what every literal in every model means."""
        assert ch.is_included(FLAT, FLAT._include_state + offset) is included

    @pytest.mark.parametrize("state, included", [(0, False), (255, True)])
    def test_the_ends_of_the_state_range(self, state, included):
        assert ch.is_included(FLAT, state) is included


class TestGetFeatureValue:
    """The flat feature id decodes to (rel_y, rel_x, z), then strides to an absolute cell.
    A wrong term reads a real but wrong pixel, which every layer above would happily use."""

    @pytest.mark.parametrize("cfg", [CONV, FLAT], ids=["conv", "flat"])
    def test_matches_a_numpy_index_for_every_feature_and_patch(self, cfg):
        X = sample(cfg)
        ph, pw = cfg._patch_dim
        sy, sx = cfg._stride
        depth = cfg._dim[2]

        for py in range(cfg._n_patches_y):
            for px in range(cfg._n_patches_x):
                for fid in range(cfg._n_raw_patch_feats):
                    rel_y, rem = divmod(fid, pw * depth)
                    rel_x, z = divmod(rem, depth)
                    expected = X[py * sy + rel_y, px * sx + rel_x, z]
                    assert ch.get_feature_value(cfg, X, py, px, fid) == expected, (py, px, fid)

    def test_every_cell_of_a_patch_is_reached_exactly_once(self):
        """Catches a decode that aliases two feature ids onto one cell, which a spot check would miss."""
        X = sample(CONV)
        seen = [ch.get_feature_value(CONV, X, 1, 1, fid) for fid in range(CONV._n_raw_patch_feats)]
        assert len(set(seen)) == CONV._n_raw_patch_feats

    def test_stride_moves_the_patch(self):
        X = sample(CONV)
        sy, sx = CONV._stride
        first = ch.get_feature_value(CONV, X, 0, 0, 0)
        assert ch.get_feature_value(CONV, X, 1, 0, 0) == X[sy, 0, 0] != first
        assert ch.get_feature_value(CONV, X, 0, 1, 0) == X[0, sx, 0] != first


class TestMatchPatch:
    def _bounds(self, cfg, constraints: dict[int, tuple[int, int]]) -> np.ndarray:
        """Full width bounds, so an unconstrained feature is only unconstrained by its interval."""
        b = np.zeros((cfg._n_raw_patch_feats, 2), dtype=np.int32)
        b[:, 0], b[:, 1] = cfg._feat_mins[: cfg._n_raw_patch_feats], cfg._feat_maxs[: cfg._n_raw_patch_feats]
        for fid, (lo, hi) in constraints.items():
            b[fid] = (lo, hi)
        return b

    def test_all_constraints_must_hold(self):
        X = sample(FLAT)
        v0, v1 = (ch.get_feature_value(FLAT, X, 0, 0, f) for f in (0, 1))
        ids = np.array([0, 1], dtype=np.int32)

        assert ch.match_patch(FLAT, X, 0, 0, self._bounds(FLAT, {0: (v0, v0), 1: (v1, v1)}), ids)
        assert not ch.match_patch(FLAT, X, 0, 0, self._bounds(FLAT, {0: (v0, v0), 1: (v1 + 1, v1 + 1)}), ids)

    def test_only_the_listed_features_are_read(self):
        """It walks `bounded_feat_ids`, not every feature. If it looped all of them it would read
        bounds that pack_clauses never wrote for a contradictory clause."""
        X = sample(FLAT)
        bounds = self._bounds(FLAT, {})
        bounds[2] = (99, -99)  # impossible, but feature 2 is not in the list

        assert ch.match_patch(FLAT, X, 0, 0, bounds, np.array([0], dtype=np.int32))
        assert not ch.match_patch(FLAT, X, 0, 0, bounds, np.array([0, 2], dtype=np.int32))

    def test_an_empty_list_matches_anything(self):
        X = sample(FLAT)
        assert ch.match_patch(FLAT, X, 0, 0, self._bounds(FLAT, {}), np.array([], dtype=np.int32))


class TestClauseOutput:
    def _args(self, cfg, constraints: dict[int, tuple[int, int]], window=None):
        n_feats = cfg._n_raw_patch_feats
        feat_bounds = np.zeros((1, n_feats, 2), dtype=np.int32)
        feat_bounds[0, :, 0] = cfg._feat_mins[:n_feats]
        feat_bounds[0, :, 1] = cfg._feat_maxs[:n_feats]
        for fid, (lo, hi) in constraints.items():
            feat_bounds[0, fid] = (lo, hi)

        ids = np.zeros((1, n_feats), dtype=np.int32)
        ids[0, : len(constraints)] = sorted(constraints)
        full = (0, cfg._n_patches_y - 1, 0, cfg._n_patches_x - 1)
        return (
            np.array([window or full], dtype=np.int32),
            feat_bounds,
            ids,
            np.array([len(constraints)], dtype=np.int32),
        )

    def test_a_contradiction_never_fires_and_never_reads_its_bounds(self):
        """Bounds are stale for a contradictory clause unless packed with full=True, so reading them
        would act on whatever the previous pack left behind."""
        X = sample(CONV)
        pos, feat_bounds, ids, n = self._args(CONV, {})
        feat_bounds[:] = -12345  # poison: touching these would not give 0
        assert ch.clause_output(CONV, X, 0, pos, feat_bounds, ids, n, -1) == 0

    def test_an_empty_clause_always_fires(self):
        X = sample(CONV)
        pos, feat_bounds, ids, n = self._args(CONV, {})
        feat_bounds[:] = -12345  # not read either, density 0 short circuits first
        assert ch.clause_output(CONV, X, 0, pos, feat_bounds, ids, n, 0) == 1

    def test_fires_when_any_patch_in_the_window_matches(self):
        X = sample(CONV)
        target = (1, 1)
        v = ch.get_feature_value(CONV, X, *target, 0)
        args = self._args(CONV, {0: (v, v)})
        assert ch.clause_output(CONV, X, 0, *args, 1) == 1

    def test_does_not_look_outside_the_position_window(self):
        """The window is the clause's position constraint. Scanning past it would let a clause fire
        on a patch its position literals exclude."""
        X = sample(CONV)
        target = (1, 1)
        v = ch.get_feature_value(CONV, X, *target, 0)

        inside = self._args(CONV, {0: (v, v)}, window=(1, 1, 1, 1))
        outside = self._args(CONV, {0: (v, v)}, window=(0, 0, 0, 0))
        assert ch.clause_output(CONV, X, 0, *inside, 1) == 1
        assert ch.clause_output(CONV, X, 0, *outside, 1) == 0
