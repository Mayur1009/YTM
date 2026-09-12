import itertools

import numpy as np
import pytest

from ytm._core.backends.cpu import CPUDevice
from ytm._core.config import BaseTMConfig
from ytm._core.device_config import DeviceConfig

INCLUDE_P = 0.05  # sparse enough that most clauses stay satisfiable


class Device(CPUDevice):
    def fit_epoch(self, X, Y, clause_drop_p, batch_size): ...
    def fit_sample(self, rng_key, buf, e): ...
    def _fit_decide_fb(self, buf, e, rng_key): ...
    def _fit_apply_fb(self, buf, e, rng_key): ...
    def _fit_update_weights(self, buf): ...
    def _fit_update_bias(self, buf): ...


def make_device(n_feats: int = 2, feat_max: int = 3, n_clauses: int = 64, **kwargs) -> Device:
    cfg = BaseTMConfig(
        n_clauses=n_clauses,
        s=10.0,
        dim=(n_feats, 1, 1),
        n_classes=1,
        feat_maxs=feat_max,
        position_literals=False,
        seed=1,
        **kwargs,
    )
    return Device(cfg, DeviceConfig())


def sprinkle_includes(dev: Device, rng: np.random.Generator, p: float = INCLUDE_P) -> None:
    cfg = dev.config
    dev.ta_states[:] = cfg._include_state - 1
    dev.ta_states[rng.random(dev.ta_states.shape) < p] = cfg._include_state


def clause_accepts(dev: Device, clause: int, x: np.ndarray) -> bool:
    """The TM definition: a clause is a conjunction of its included literals.

    Thermometer literal `b` of feature `f` asserts `x[f] > feat_mins[f] + b`, its negation asserts
    `x[f] <= feat_mins[f] + b`. No intervals anywhere.
    """
    cfg = dev.config
    half = cfg._n_literals // 2 if cfg.negated_literals else 0
    included = dev.ta_states[clause] >= cfg._include_state

    for f in range(cfg._n_raw_patch_feats):
        start = cfg._n_position_feats + cfg._literal_offsets[f]
        n_bits = cfg._literal_offsets[f + 1] - cfg._literal_offsets[f]
        for b in range(n_bits):
            if included[start + b] and not (x[f] > cfg._feat_mins[f] + b):
                return False
            if half and included[start + b + half] and not (x[f] <= cfg._feat_mins[f] + b):
                return False
    return True


def packed_accepts(dev: Device, packed, clause: int, x: np.ndarray) -> bool:
    """What the packed form accepts, mirroring the bounds check the kernels do."""
    if packed.clause_density[clause] < 0:
        return False
    n = packed.n_bounded_feats[clause]
    for i in range(n):
        f = packed.bounded_feat_ids[clause, i]
        lo, hi = packed.clause_feat_bounds[clause, f]
        if not (lo <= x[f] <= hi):
            return False
    return True


def all_inputs(n_feats: int, feat_max: int) -> list[np.ndarray]:
    return [np.array(v, dtype=np.int32) for v in itertools.product(range(feat_max + 1), repeat=n_feats)]


class TestAgainstLiteralEvaluation:
    @pytest.mark.parametrize("trial", range(8))
    def test_packed_and_literal_evaluation_agree_on_every_input(self, trial):
        dev = make_device()
        cfg = dev.config
        rng = np.random.default_rng(trial)
        sprinkle_includes(dev, rng)
        dev.pack_clauses(force_repack=True)
        packed = dev.get_packed_clauses()

        inputs = all_inputs(cfg._n_raw_patch_feats, int(cfg._feat_maxs[0]))
        valid = packed.clause_density >= 0
        non_trivial = 0

        for clause in range(cfg._total_clauses):
            accepted = [clause_accepts(dev, clause, x) for x in inputs]
            if valid[clause] and 0 < sum(accepted) < len(inputs):
                non_trivial += 1
            for x, expected in zip(inputs, accepted):
                assert packed_accepts(dev, packed, clause, x) == expected, (clause, x.tolist())

        # Without this the whole test can pass on clauses that accept nothing.
        assert non_trivial >= cfg._total_clauses // 4, f"only {non_trivial} clauses were non trivial"

    @pytest.mark.parametrize("feat_max", [1, 3, 7])
    def test_agreement_holds_across_feature_ranges(self, feat_max):
        dev = make_device(feat_max=feat_max, n_clauses=32)
        cfg = dev.config
        sprinkle_includes(dev, np.random.default_rng(0), p=0.1 if feat_max == 1 else INCLUDE_P)
        dev.pack_clauses(force_repack=True)
        packed = dev.get_packed_clauses()

        for clause in range(cfg._total_clauses):
            for x in all_inputs(cfg._n_raw_patch_feats, feat_max):
                assert packed_accepts(dev, packed, clause, x) == clause_accepts(dev, clause, x)

    def test_agreement_holds_without_negated_literals(self):
        dev = make_device(negated_literals=False)
        cfg = dev.config
        sprinkle_includes(dev, np.random.default_rng(3), p=0.1)
        dev.pack_clauses(force_repack=True)
        packed = dev.get_packed_clauses()

        for clause in range(cfg._total_clauses):
            for x in all_inputs(cfg._n_raw_patch_feats, int(cfg._feat_maxs[0])):
                assert packed_accepts(dev, packed, clause, x) == clause_accepts(dev, clause, x)


class TestDensity:
    """`match_patch` never reads the count, only its sign, so check the value directly."""

    def test_counts_the_included_literals(self):
        dev = make_device()
        cfg = dev.config
        sprinkle_includes(dev, np.random.default_rng(1))
        dev.pack_clauses(force_repack=True)
        packed = dev.get_packed_clauses()

        expected = (dev.ta_states >= cfg._include_state).sum(axis=1)
        valid = packed.clause_density >= 0
        assert valid.any()
        assert np.array_equal(packed.clause_density[valid], expected[valid])

    def test_an_interval_inverted_by_one_is_still_a_contradiction(self):
        """`decide_feedback` reads `density <= MAX_INCLUDED_LITERALS` as "has space". A barely
        inverted clause left unflagged would be given Type Ia instead of Type Ib forever."""
        dev = make_device()
        cfg = dev.config
        half = cfg._n_literals // 2
        dev.ta_states[:] = cfg._include_state - 1
        dev.ta_states[0, cfg._literal_offsets[0] + 1] = cfg._include_state  # f0 > 1, so lb = 2
        dev.ta_states[0, cfg._literal_offsets[0] + 1 + half] = cfg._include_state  # f0 <= 1, so ub = 1
        dev.pack_clauses(force_repack=True, full=True)

        lb, ub = dev.get_packed_clauses().clause_feat_bounds[0, 0]
        assert lb == ub + 1, f"this test needs an interval inverted by exactly one, got [{lb}, {ub}]"
        assert dev.get_packed_clauses().clause_density[0] == -1

    def test_a_contradiction_is_flagged(self):
        dev = make_device()
        cfg = dev.config
        half = cfg._n_literals // 2
        dev.ta_states[:] = cfg._include_state - 1
        dev.ta_states[0, cfg._literal_offsets[0] + 2] = cfg._include_state  # x > 2
        dev.ta_states[0, cfg._literal_offsets[0] + 0 + half] = cfg._include_state  # x <= 0
        dev.pack_clauses(force_repack=True)

        assert dev.get_packed_clauses().clause_density[0] == -1

    def test_flagged_clauses_accept_nothing(self):
        dev = make_device()
        cfg = dev.config
        sprinkle_includes(dev, np.random.default_rng(2), p=0.3)
        dev.pack_clauses(force_repack=True)
        packed = dev.get_packed_clauses()

        invalid = np.flatnonzero(packed.clause_density < 0)
        assert len(invalid) > 0
        for clause in invalid:
            assert not any(clause_accepts(dev, int(clause), x) for x in all_inputs(cfg._n_raw_patch_feats, 3))


class TestBoundedFeats:
    def test_only_constrained_features_are_listed(self):
        dev = make_device()
        cfg = dev.config
        sprinkle_includes(dev, np.random.default_rng(4))
        dev.pack_clauses(force_repack=True)
        packed = dev.get_packed_clauses()

        for clause in np.flatnonzero(packed.clause_density >= 0):
            listed = set(packed.bounded_feat_ids[clause, : packed.n_bounded_feats[clause]].tolist())
            constrained = {
                f
                for f in range(cfg._n_raw_patch_feats)
                if tuple(packed.clause_feat_bounds[clause, f]) != (cfg._feat_mins[f], cfg._feat_maxs[f])
            }
            assert listed == constrained, clause


class TestProperties:
    def test_packing_is_idempotent(self):
        dev = make_device()
        sprinkle_includes(dev, np.random.default_rng(5))
        dev.pack_clauses(force_repack=True)
        first = dev.get_packed_clauses()
        dev.pack_clauses(force_repack=True)
        second = dev.get_packed_clauses()

        assert np.array_equal(first.clause_feat_bounds, second.clause_feat_bounds)
        assert np.array_equal(first.clause_density, second.clause_density)

    def test_adding_a_literal_can_only_narrow_the_accept_set(self):
        """A clause is a conjunction, so an extra condition never admits more inputs."""
        dev = make_device(n_clauses=8, feat_max=7)
        cfg = dev.config
        rng = np.random.default_rng(6)
        inputs = all_inputs(cfg._n_raw_patch_feats, 7)
        sprinkle_includes(dev, rng)

        dev.pack_clauses(force_repack=True)
        before = {c: {tuple(x) for x in inputs if clause_accepts(dev, c, x)} for c in range(cfg._total_clauses)}

        excluded = np.flatnonzero(dev.ta_states[0] < cfg._include_state)
        dev.ta_states[0, rng.choice(excluded)] = cfg._include_state
        dev.pack_clauses(force_repack=True)
        packed = dev.get_packed_clauses()

        after = {tuple(x) for x in inputs if packed_accepts(dev, packed, 0, x)}
        assert after <= before[0]


class TestPositionBounds:
    """`match_patch` never reads them, so they need their own oracle.

    Position literals are thermometer coded over patch indices: literal `l` asserts `patch > l`,
    its negation asserts `patch <= l`.
    """

    @staticmethod
    def _conv_device(n_clauses: int = 32) -> Device:
        cfg = BaseTMConfig(
            n_clauses=n_clauses, s=10.0, dim=(6, 6), n_classes=1, patch_dim=(3, 3), feat_maxs=1, seed=1
        )
        return Device(cfg, DeviceConfig())

    @staticmethod
    def _position_bounds_naive(dev: Device, clause: int) -> tuple[int, int, int, int]:
        cfg = dev.config
        half = cfg._n_literals // 2
        included = dev.ta_states[clause] >= cfg._include_state
        ny, nx = cfg._n_patches_y, cfg._n_patches_x

        lo_y, hi_y, lo_x, hi_x = 0, ny - 1, 0, nx - 1
        for lit in range(ny - 1):
            if included[lit]:
                lo_y = max(lo_y, lit + 1)
            if included[lit + half]:
                hi_y = min(hi_y, lit)
        for lit in range(nx - 1):
            if included[(ny - 1) + lit]:
                lo_x = max(lo_x, lit + 1)
            if included[(ny - 1) + lit + half]:
                hi_x = min(hi_x, lit)
        return lo_y, hi_y, lo_x, hi_x

    @pytest.mark.parametrize("trial", range(4))
    def test_bounds_match_the_literal_scan(self, trial):
        dev = self._conv_device()
        cfg = dev.config
        sprinkle_includes(dev, np.random.default_rng(trial), p=0.03)
        dev.pack_clauses(force_repack=True)
        packed = dev.get_packed_clauses()

        constrained = 0
        for clause in np.flatnonzero(packed.clause_density >= 0):
            expected = self._position_bounds_naive(dev, int(clause))
            assert tuple(packed.clause_position_bounds[clause]) == expected, clause
            if expected != (0, cfg._n_patches_y - 1, 0, cfg._n_patches_x - 1):
                constrained += 1

        assert constrained > 0, "no clause constrained a position, the test proved nothing"

    def test_an_impossible_position_invalidates_the_clause(self):
        dev = self._conv_device()
        cfg = dev.config
        half = cfg._n_literals // 2
        dev.ta_states[:] = cfg._include_state - 1
        dev.ta_states[0, 2] = cfg._include_state  # patch_y > 2
        dev.ta_states[0, 0 + half] = cfg._include_state  # patch_y <= 0
        dev.pack_clauses(force_repack=True)

        assert dev.get_packed_clauses().clause_density[0] == -1




class TestBoundedFeatIdList:
    def test_the_list_is_compact_and_ordered(self):
        """`match_patch` reads `ids[0 : n_bounded_feats]`. A gap or a stale tail entry would make it
        constrain a feature the clause never mentioned."""
        dev = make_device()
        sprinkle_includes(dev, np.random.default_rng(11))
        dev.pack_clauses(force_repack=True)
        packed = dev.get_packed_clauses()

        valid = np.flatnonzero(packed.clause_density >= 0)
        assert len(valid) > 0
        for clause in valid:
            n = packed.n_bounded_feats[clause]
            ids = packed.bounded_feat_ids[clause, :n].tolist()
            assert ids == sorted(set(ids)), clause
            assert all(0 <= f < dev.config._n_raw_patch_feats for f in ids), clause


class TestSyncFlag:
    def test_packing_marks_every_clause_synced(self):
        dev = make_device()
        sprinkle_includes(dev, np.random.default_rng(12))
        dev.pack_clauses(force_repack=True)
        assert np.all(dev.get_packed_clauses().is_clause_synced == 1)

    def test_a_synced_clause_is_skipped(self):
        """The flag is what keeps training from repacking every clause every sample. If it were
        ignored the cost would rise silently; if it were honoured wrongly the model would evaluate
        a stale packed form."""
        dev = make_device()
        cfg = dev.config
        sprinkle_includes(dev, np.random.default_rng(13))
        dev.pack_clauses(force_repack=True)
        before = dev.get_packed_clauses().clause_feat_bounds.copy()

        dev.ta_states[:] = cfg._include_state  # every literal included, a very different clause
        dev.pack_clauses()  # no force, so nothing is stale as far as the flag knows

        assert np.array_equal(dev.get_packed_clauses().clause_feat_bounds, before)

    def test_force_repack_picks_up_the_change(self):
        dev = make_device()
        cfg = dev.config
        sprinkle_includes(dev, np.random.default_rng(13))
        dev.pack_clauses(force_repack=True)
        before = dev.get_packed_clauses().clause_feat_bounds.copy()

        dev.ta_states[:] = cfg._include_state
        dev.pack_clauses(force_repack=True)

        assert not np.array_equal(dev.get_packed_clauses().clause_feat_bounds, before)


class TestFullScan:
    """`full` keeps scanning past a contradiction so the bounds show where the clause went wrong."""

    @staticmethod
    def _contradictory(dev: Device) -> None:
        cfg = dev.config
        half = cfg._n_literals // 2
        dev.ta_states[:] = cfg._include_state - 1
        dev.ta_states[0, cfg._literal_offsets[0] + 2] = cfg._include_state  # f0 > 2
        dev.ta_states[0, cfg._literal_offsets[0] + half] = cfg._include_state  # f0 <= 0
        dev.ta_states[0, cfg._literal_offsets[1] + 1] = cfg._include_state  # f1 > 1, fine on its own

    def test_density_is_negative_either_way(self):
        """Evaluation keys off the sign, so a full pack must not make a broken clause look usable."""
        for full in (False, True):
            dev = make_device()
            self._contradictory(dev)
            dev.pack_clauses(force_repack=True, full=full)
            assert dev.get_packed_clauses().clause_density[0] == -1

    def test_without_full_the_later_bounds_are_left_alone(self):
        dev = make_device()
        self._contradictory(dev)
        dev.get_packed_clauses()
        dev.packed_clauses.clause_feat_bounds[0] = -99  # poison, so an untouched slot is visible
        dev.pack_clauses(force_repack=True, full=False)

        assert np.all(dev.get_packed_clauses().clause_feat_bounds[0, 1] == -99)

    def test_full_computes_every_feature(self):
        dev = make_device()
        cfg = dev.config
        self._contradictory(dev)
        dev.packed_clauses.clause_feat_bounds[0] = -99
        dev.pack_clauses(force_repack=True, full=True)
        bounds = dev.get_packed_clauses().clause_feat_bounds[0]

        assert tuple(bounds[0]) == (3, 0), "the offending feature, inverted so it is visible"
        assert tuple(bounds[1]) == (2, int(cfg._feat_maxs[1])), "the feature after it, still real"

    def test_valid_clauses_are_unaffected_by_the_flag(self):
        """So `get_clauses(full=True)` can be called mid training without perturbing anything."""
        dev = make_device()
        sprinkle_includes(dev, np.random.default_rng(14))
        dev.pack_clauses(force_repack=True, full=False)
        cheap = dev.get_packed_clauses()
        valid = cheap.clause_density >= 0
        assert valid.any()
        bounds, density = cheap.clause_feat_bounds[valid].copy(), cheap.clause_density.copy()

        dev.pack_clauses(force_repack=True, full=True)
        after = dev.get_packed_clauses()
        assert np.array_equal(after.clause_feat_bounds[valid], bounds)
        assert np.array_equal(after.clause_density, density)

    def test_full_also_scans_past_a_position_contradiction(self):
        """The position scan runs first and bails out before the features. Without `full` honoured
        on that path too, a clause with a bad position window would report no feature bounds at all."""
        dev = TestPositionBounds._conv_device(n_clauses=4)
        cfg = dev.config
        half = cfg._n_literals // 2
        dev.ta_states[:] = cfg._include_state - 1
        dev.ta_states[0, 1] = cfg._include_state  # patch_y > 1
        dev.ta_states[0, half] = cfg._include_state  # patch_y <= 0, so the window is impossible
        dev.ta_states[0, cfg._n_position_feats + cfg._literal_offsets[0]] = cfg._include_state  # f0 > 0

        dev.packed_clauses.clause_feat_bounds[0] = -99
        dev.pack_clauses(force_repack=True, full=False)
        assert np.all(dev.get_packed_clauses().clause_feat_bounds[0] == -99), "cheap path stops at the position"

        dev.packed_clauses.clause_feat_bounds[0] = -99
        dev.pack_clauses(force_repack=True, full=True)
        packed = dev.get_packed_clauses()
        assert packed.clause_density[0] == -1
        assert tuple(packed.clause_position_bounds[0][:2]) == (2, 0), "the impossible y window is visible"
        assert tuple(packed.clause_feat_bounds[0, 0]) == (1, int(cfg._feat_maxs[0])), "features scanned anyway"
