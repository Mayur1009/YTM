import numpy as np
import pytest

from ytm._core.config import BaseTMConfig

from . import feedback_harness as fh

# At S = 1 every geometric draw is 1, so the feedbacks touch every literal in range and the result
# is exact. That is what lets the range boundaries be asserted rather than sampled.
BASE = {"n_clauses": 4, "n_classes": 2, "feat_maxs": 3, "seed": 1}
DET = BaseTMConfig(**BASE, s=1.0, dim=(4, 3, 1), patch_dim=(2, 2), stride=(1, 1))
DET_NO_POS = BaseTMConfig(**BASE, s=1.0, dim=(2, 2, 1), position_literals=False)
DET_NO_NEG = BaseTMConfig(**BASE, s=1.0, dim=(2, 2, 1), negated_literals=False, position_literals=False)
STOCHASTIC = BaseTMConfig(**BASE, s=4.0, dim=(2, 2, 1), position_literals=False)

MID = 100  # a state far from both clamps, so a step is always visible


def satisfied_mask(cfg: BaseTMConfig, X: np.ndarray, py: int, px: int) -> np.ndarray:
    """Which literals the patch satisfies, from the thermometer definition rather than from the C.

    Positive literal `b` of feature `f` asserts `val > feat_min[f] + b`; the negated half asserts
    `val <= feat_min[f] + b`. Position literal `b` asserts `py > b` (and `px > b` for the x block).
    """
    n = cfg._n_literals
    half = n // 2 if cfg.negated_literals else 0
    sat = np.zeros(n, dtype=bool)

    if cfg.position_literals:
        ny, nx = cfg._n_patches_y - 1, cfg._n_patches_x - 1
        for b in range(ny):
            sat[b] = py > b
            if half:
                sat[b + half] = py <= b
        for b in range(nx):
            sat[ny + b] = px > b
            if half:
                sat[ny + b + half] = px <= b

    for fid in range(cfg._n_raw_patch_feats):
        lo = cfg._n_position_feats + cfg._literal_offsets[fid]
        n_bits = cfg._literal_offsets[fid + 1] - cfg._literal_offsets[fid]
        pw = cfg._patch_dim[1]
        depth = cfg._dim[2]
        rel_y, rem = divmod(fid, pw * depth)
        rel_x, z = divmod(rem, depth)
        val = X[py * cfg._stride[0] + rel_y, px * cfg._stride[1] + rel_x, z]

        for b in range(n_bits):
            sat[lo + b] = val > cfg._feat_mins[fid] + b
            if half:
                sat[lo + b + half] = val <= cfg._feat_mins[fid] + b
    return sat


def sample(cfg: BaseTMConfig) -> np.ndarray:
    h, w, d = cfg._dim
    rng = np.random.default_rng(3)
    return rng.integers(0, cfg.feat_maxs + 1, size=(h, w, d)).astype(np.int32)


class TestRangePrimitives:
    @pytest.mark.parametrize("fn, delta", [(fh.inc_lits, 1), (fh.dec_lits, -1)])
    def test_only_the_given_range_moves_and_by_one(self, fn, delta):
        ta = fh.states(DET, MID)
        fn(DET, ta, 3, 7)
        expected = np.full(DET._n_literals, MID, dtype=np.int64)
        expected[3:7] += delta
        assert np.array_equal(ta.astype(np.int64), expected)

    @pytest.mark.parametrize("fn", [fh.inc_lits, fh.dec_lits])
    def test_offset_selects_the_other_half(self, fn):
        half = DET._n_literals // 2
        ta = fh.states(DET, MID)
        fn(DET, ta, 0, 3, offset=half)
        assert np.all(ta[:3] == MID)
        assert np.all(ta[half : half + 3] != MID)

    @pytest.mark.parametrize("fn", [fh.inc_lits, fh.dec_lits, fh.prob_inc_lits, fh.prob_dec_lits])
    def test_an_empty_or_inverted_range_does_nothing(self, fn):
        ta = fh.states(DET, MID)
        args = (DET, ta, 1.0, 5, 5) if "prob" in fn.__name__ else (DET, ta, 5, 5)
        fn(*args)
        args = (DET, ta, 1.0, 7, 3) if "prob" in fn.__name__ else (DET, ta, 7, 3)
        fn(*args)
        assert np.all(ta == MID)

    def test_increment_clamps_at_the_top_state(self):
        top = DET.n_states - 1
        ta = fh.states(DET, top)
        fh.inc_lits(DET, ta, 0, DET._n_literals)
        assert np.all(ta == top)

    def test_decrement_clamps_at_zero(self):
        ta = fh.states(DET, 0)
        fh.dec_lits(DET, ta, 0, DET._n_literals)
        assert np.all(ta == 0)


class TestProbabilisticPrimitives:
    @pytest.mark.parametrize("prob", [0.1, 0.25, 0.5])
    def test_hit_rate_matches_the_probability(self, prob):
        n, trials = STOCHASTIC._n_literals, 4000
        hits = np.zeros(n)
        for key in range(trials):
            ta = fh.states(STOCHASTIC, MID)
            fh.prob_inc_lits(STOCHASTIC, ta, prob, 0, n, key=key)
            hits += ta > MID
        assert abs(hits.mean() / trials - prob) < 0.02

    @pytest.mark.parametrize("prob", [1.0, 1.5])
    def test_certain_probability_touches_everything(self, prob):
        ta = fh.states(STOCHASTIC, MID)
        fh.prob_inc_lits(STOCHASTIC, ta, prob, 0, STOCHASTIC._n_literals)
        assert np.all(ta == MID + 1)

    @pytest.mark.parametrize("prob", [0.0, -0.5, float("nan")])
    def test_impossible_probability_touches_nothing(self, prob):
        """`geom_sample` returns INFINITY here, so the loop must not run at all."""
        ta = fh.states(STOCHASTIC, MID)
        fh.prob_inc_lits(STOCHASTIC, ta, prob, 0, STOCHASTIC._n_literals)
        assert np.all(ta == MID)

    def test_stays_inside_the_range(self):
        n = STOCHASTIC._n_literals
        touched = np.zeros(n, dtype=bool)
        for key in range(500):
            ta = fh.states(STOCHASTIC, MID)
            fh.prob_inc_lits(STOCHASTIC, ta, 0.9, 2, n - 2, key=key)
            touched |= ta != MID
        assert not touched[:2].any() and not touched[-2:].any()


class TestBoostDispatch:
    """`t1a_incs` and `t1a_decs` pick deterministic or geometric from the BOOST_TP_* defines."""

    @pytest.mark.parametrize("boost, all_moved", [(True, True), (False, False)])
    def test_boost_inc_decides_whether_every_literal_moves(self, boost, all_moved):
        cfg = BaseTMConfig(**BASE, s=4.0, dim=(2, 2, 1), position_literals=False, boost_tp_inc=boost)
        ta = fh.states(cfg, MID)
        fh.t1a_incs(cfg, ta, 0, cfg._n_literals)
        assert bool(np.all(ta == MID + 1)) is all_moved

    @pytest.mark.parametrize("boost, all_moved", [(True, True), (False, False)])
    def test_boost_dec_decides_whether_every_literal_moves(self, boost, all_moved):
        cfg = BaseTMConfig(**BASE, s=4.0, dim=(2, 2, 1), position_literals=False, boost_tp_dec=boost)
        ta = fh.states(cfg, MID)
        fh.t1a_decs(cfg, ta, 0, cfg._n_literals)
        assert bool(np.all(ta == MID - 1)) is all_moved

    @pytest.mark.parametrize("s", [1.0, 0.5])
    def test_s_at_or_below_one_makes_the_unboosted_paths_degenerate(self, s):
        """These used to be an `if (S > 1.0f)` branch. Now `1 - S_INV <= 0` and `S_INV >= 1` reach
        geom_sample's guards instead, so the behaviour has to survive the branch being gone."""
        cfg = BaseTMConfig(**BASE, s=s, dim=(2, 2, 1), position_literals=False, boost_tp_inc=False, boost_tp_dec=False)
        n = cfg._n_literals

        inc = fh.states(cfg, MID)
        fh.t1a_incs(cfg, inc, 0, n)
        assert np.all(inc == MID), "p = 1 - 1/S is not positive, so nothing should move"

        dec = fh.states(cfg, MID)
        fh.t1a_decs(cfg, dec, 0, n)
        assert np.all(dec == MID - 1), "p = 1/S is at least 1, so everything should move"


class TestFeedbackTypes:
    """Asserted exactly, using S = 1 so every literal in range moves."""

    @pytest.mark.parametrize("cfg", [DET, DET_NO_POS, DET_NO_NEG], ids=["full", "no position", "no negated"])
    def test_type1a_reinforces_what_the_patch_satisfies(self, cfg):
        X = sample(cfg)
        py, px = cfg._n_patches_y - 1, cfg._n_patches_x - 1
        sat = satisfied_mask(cfg, X, py, px)

        ta = fh.states(cfg, MID)
        fh.type1a_fb(cfg, ta, X, py, px)

        expected = np.where(sat, MID + 1, MID - 1).astype(np.int64)
        assert np.array_equal(ta.astype(np.int64), expected)

    @pytest.mark.parametrize("cfg", [DET, DET_NO_POS, DET_NO_NEG], ids=["full", "no position", "no negated"])
    def test_type2_reinforces_only_what_the_patch_fails(self, cfg):
        X = sample(cfg)
        py, px = cfg._n_patches_y - 1, cfg._n_patches_x - 1
        sat = satisfied_mask(cfg, X, py, px)

        ta = fh.states(cfg, MID)
        fh.type2_fb(cfg, ta, X, py, px)

        expected = np.where(sat, MID, MID + 1).astype(np.int64)
        assert np.array_equal(ta.astype(np.int64), expected)

    def test_type1b_weakens_every_literal(self):
        ta = fh.states(DET, MID)
        fh.type1b_fb(DET, ta)
        assert np.all(ta == MID - 1)

    def test_type1a_and_type2_disagree_on_the_satisfied_side(self):
        """The two are opposites where the patch matches, which is what makes them push apart."""
        X = sample(DET)
        sat = satisfied_mask(DET, X, 0, 0)

        a, b = fh.states(DET, MID), fh.states(DET, MID)
        fh.type1a_fb(DET, a, X, 0, 0)
        fh.type2_fb(DET, b, X, 0, 0)

        assert np.all(a[sat] > b[sat])
        assert np.all(a[~sat] < b[~sat])

    def test_a_different_patch_gives_a_different_result(self):
        """Guards against the patch index being ignored, which a single position test would miss."""
        X = sample(DET)
        first, last = fh.states(DET, MID), fh.states(DET, MID)
        fh.type1a_fb(DET, first, X, 0, 0)
        fh.type1a_fb(DET, last, X, DET._n_patches_y - 1, DET._n_patches_x - 1)
        assert not np.array_equal(first, last)
