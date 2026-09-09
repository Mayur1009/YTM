"""Differential tests for the clause packing.

`pack_clauses` compiles TA states into per feature intervals so evaluation is a bounds check
instead of a literal scan. The oracle here never computes an interval: it walks the literals one
by one straight from the TM definition, so a bug in the interval derivation shows up as a
disagreement rather than being reproduced.

The include probability matters more than the number of trials. At the density `ta_init="random"`
produces, every clause contradicts itself and both sides trivially agree on everything, so the
tests assert that the generated clauses were actually valid and non trivial.
"""

import itertools

import common_harness
import numpy as np
import pytest

from ytm._core.backends.cpu import CPUDevice
from ytm._core.config import BaseTMConfig
from ytm._core.device_config import DeviceConfig

INCLUDE_P = 0.05  # sparse enough that most clauses stay satisfiable


class Device(CPUDevice):
    def fit_epoch(self, X, Y, clause_drop_p, batch_size, **kwargs): ...
    def fit_sample(self, X, Y, e, **kwargs): ...
    def infer(self, X, batch_size): ...
    def transform(self, X, batch_size, force_repack=False): ...
    def transform_patchwise(self, X, batch_size, force_repack=False): ...
    def wic(self, class_id, polarity, pw_th=0.0, force_repack=False): ...
    def wac(self, X, target_classes, polarity, force_repack=False): ...


def make_device(n_feats: int = 2, feat_max: int = 3, n_clauses: int = 64, **kwargs) -> Device:
    """A model small enough to enumerate every possible input."""
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


class TestKnownBounds:
    """Direct assertions, so a bug shared between packing and the bounds check cannot hide."""

    @pytest.mark.parametrize("bit, expected", [(0, (1, 7)), (3, (4, 7)), (6, (7, 7))])
    def test_positive_literal_sets_the_lower_bound(self, bit, expected):
        dev = make_device(feat_max=7)
        cfg = dev.config
        dev.ta_states[:] = cfg._include_state - 1
        dev.ta_states[0, cfg._literal_offsets[0] + bit] = cfg._include_state
        dev.pack_clauses(force_repack=True)

        assert tuple(dev.get_packed_clauses().clause_feat_bounds[0, 0]) == expected

    @pytest.mark.parametrize("bit, expected", [(0, (0, 0)), (3, (0, 3)), (6, (0, 6))])
    def test_negated_literal_sets_the_upper_bound(self, bit, expected):
        dev = make_device(feat_max=7)
        cfg = dev.config
        half = cfg._n_literals // 2
        dev.ta_states[:] = cfg._include_state - 1
        dev.ta_states[0, cfg._literal_offsets[0] + bit + half] = cfg._include_state
        dev.pack_clauses(force_repack=True)

        assert tuple(dev.get_packed_clauses().clause_feat_bounds[0, 0]) == expected

    def test_both_sides_narrow_to_an_interval(self):
        dev = make_device(feat_max=7)
        cfg = dev.config
        half = cfg._n_literals // 2
        dev.ta_states[:] = cfg._include_state - 1
        dev.ta_states[0, cfg._literal_offsets[0] + 1] = cfg._include_state  # x > 1
        dev.ta_states[0, cfg._literal_offsets[0] + 5 + half] = cfg._include_state  # x <= 5
        dev.pack_clauses(force_repack=True)

        packed = dev.get_packed_clauses()
        assert tuple(packed.clause_feat_bounds[0, 0]) == (2, 5)
        assert packed.clause_density[0] == 2


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

    def test_a_positive_literal_raises_the_lower_bound(self):
        dev = self._conv_device()
        cfg = dev.config
        dev.ta_states[:] = cfg._include_state - 1
        dev.ta_states[0, 1] = cfg._include_state  # patch_y > 1
        dev.pack_clauses(force_repack=True)

        lo_y, hi_y, lo_x, hi_x = dev.get_packed_clauses().clause_position_bounds[0]
        assert (lo_y, hi_y) == (2, cfg._n_patches_y - 1)
        assert (lo_x, hi_x) == (0, cfg._n_patches_x - 1)

    def test_a_negated_literal_lowers_the_upper_bound(self):
        dev = self._conv_device()
        cfg = dev.config
        half = cfg._n_literals // 2
        dev.ta_states[:] = cfg._include_state - 1
        dev.ta_states[0, 1 + half] = cfg._include_state  # patch_y <= 1
        dev.pack_clauses(force_repack=True)

        lo_y, hi_y, _, _ = dev.get_packed_clauses().clause_position_bounds[0]
        assert (lo_y, hi_y) == (0, 1)

    def test_an_impossible_position_invalidates_the_clause(self):
        dev = self._conv_device()
        cfg = dev.config
        half = cfg._n_literals // 2
        dev.ta_states[:] = cfg._include_state - 1
        dev.ta_states[0, 2] = cfg._include_state  # patch_y > 2
        dev.ta_states[0, 0 + half] = cfg._include_state  # patch_y <= 0
        dev.pack_clauses(force_repack=True)

        assert dev.get_packed_clauses().clause_density[0] == -1


class TestCommonHelpers:
    """`match_patch` and `get_feature_value` are the consumers of the packed form.

    The oracle above restates the bounds check in Python, so these call the real C instead.
    """

    @staticmethod
    def _conv_config() -> BaseTMConfig:
        return BaseTMConfig(n_clauses=8, s=10.0, dim=(6, 6, 2), n_classes=1, patch_dim=(3, 3), feat_maxs=7, seed=1)

    def test_get_feature_value_indexes_the_patch(self):
        cfg = self._conv_config()
        h, w, d = cfg._dim
        X = np.arange(h * w * d, dtype=np.int32).reshape(h, w, d)

        for py, px, fid in [(0, 0, 0), (0, 0, 5), (1, 2, 0), (2, 2, 17), (3, 3, 11)]:
            rel_y, rem = divmod(fid, cfg._patch_dim[1] * d)
            rel_x, z = divmod(rem, d)
            expected = X[py * cfg._stride[0] + rel_y, px * cfg._stride[1] + rel_x, z]
            assert common_harness.get_feature_value(cfg, X, py, px, fid) == expected, (py, px, fid)

    def test_match_patch_accepts_only_within_every_bound(self):
        cfg = self._conv_config()
        h, w, d = cfg._dim
        rng = np.random.default_rng(0)
        X = rng.integers(0, 8, size=(h, w, d), dtype=np.int32)

        n_feats = cfg._n_raw_patch_feats
        bounds = np.stack([np.zeros(n_feats), np.full(n_feats, 7)], axis=1).astype(np.int32)
        ids = np.array([0, 4, 11], dtype=np.int32)

        # wide bounds accept everything
        assert common_harness.match_patch(cfg, X, 1, 1, bounds, ids)

        # narrow one listed feature to exclude the value actually present
        actual = common_harness.get_feature_value(cfg, X, 1, 1, 4)
        bounds[4] = (actual + 1, 7) if actual < 7 else (0, actual - 1)
        assert not common_harness.match_patch(cfg, X, 1, 1, bounds, ids)

        # the same violation on an unlisted feature is ignored
        bounds[4] = (0, 7)
        bounds[7] = (99, 99)
        assert common_harness.match_patch(cfg, X, 1, 1, bounds, ids)

    def test_match_patch_agrees_with_the_bounds_check_over_random_trials(self):
        cfg = self._conv_config()
        h, w, d = cfg._dim
        rng = np.random.default_rng(1)
        n_feats = cfg._n_raw_patch_feats
        agreed_true = agreed_false = 0

        for _ in range(200):
            X = rng.integers(0, 8, size=(h, w, d), dtype=np.int32)
            lo = rng.integers(0, 8, size=n_feats, dtype=np.int32)
            hi = np.minimum(lo + rng.integers(0, 5, size=n_feats), 7).astype(np.int32)
            bounds = np.stack([lo, hi], axis=1).astype(np.int32)
            ids = rng.choice(n_feats, size=4, replace=False).astype(np.int32)
            py, px = int(rng.integers(cfg._n_patches_y)), int(rng.integers(cfg._n_patches_x))

            expected = all(
                lo[f] <= common_harness.get_feature_value(cfg, X, py, px, int(f)) <= hi[f] for f in ids
            )
            assert common_harness.match_patch(cfg, X, py, px, bounds, ids) == expected
            agreed_true += expected
            agreed_false += not expected

        assert agreed_true > 0 and agreed_false > 0, "trials were one sided"
