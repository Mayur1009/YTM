from typing import ClassVar

import numpy as np
import pytest

from ytm._core.backends.cpu import CPUFitBuffers

from .conftest import CoreDevice, make_device, sprinkle_includes, thread_pair

CONV = {"dim": (6, 6, 1), "patch_dim": (3, 3), "n_clauses": 16, "n_classes": 3, "feat_maxs": 3}
FLAT = {"dim": (4, 4, 1), "n_clauses": 16, "n_classes": 3, "feat_maxs": 3}


def prepared(trial: int = 0, p: float = 0.03, shape: dict | None = None, **kwargs) -> tuple[CoreDevice, np.ndarray]:
    """A packed model with sparse includes, plus a few samples to run it on.

    `shape` picks the geometry. FLAT has a single patch, which takes the `N_PATCHES == 1` branch of
    `evaluate` rather than the convolutional one, so both need covering.
    """
    dev = make_device(**{**(shape or CONV), **kwargs})
    sprinkle_includes(dev, np.random.default_rng(trial), p)
    dev.pack_clauses(force_repack=True)
    X = np.random.default_rng(100 + trial).integers(0, 4, size=(4, *dev.config._dim), dtype=np.int32)
    return dev, X


def buffers(dev: CoreDevice, X: np.ndarray, drop: np.ndarray | None = None) -> CPUFitBuffers:
    cfg = dev.config
    return CPUFitBuffers(
        X=np.ascontiguousarray(X, dtype=np.int32),
        Y=np.zeros((X.shape[0], cfg.n_classes), dtype=np.float32),
        clause_drop_mask=np.zeros(cfg._total_clauses, dtype=np.int8) if drop is None else drop,
        selected_pids=np.empty(cfg._total_clauses, dtype=np.int32),
        votes=np.empty(cfg.n_classes, dtype=np.float32),
    )


def weighted_sum(dev: CoreDevice, fired: np.ndarray) -> np.ndarray:
    """Sum the weights of the clauses that fired, mapping clauses to classes as the header does."""
    cfg = dev.config
    per_class = cfg._total_clauses if cfg.coalesced else cfg._total_clauses // cfg.n_classes
    sums = np.zeros((fired.shape[0], cfg.n_classes), dtype=np.float64)

    for e in range(fired.shape[0]):
        for clause in np.flatnonzero(fired[e]):
            rel = int(clause) % per_class
            classes = range(cfg.n_classes) if cfg.coalesced else [int(clause) // per_class]
            for c in classes:
                sums[e, c] += dev.clause_weights[c, rel]
    return sums


class TestInferencePaths:
    """The three deterministic entry points, checked against each other and against numpy."""

    @pytest.mark.parametrize("shape", [CONV, FLAT], ids=["conv", "flat"])
    @pytest.mark.parametrize("trial", range(3))
    def test_clause_outputs_are_the_or_over_patches(self, shape, trial):
        """`calc_clause_outputs` and the patchwise scan are separate C loops. A clause fires exactly
        when some patch inside its window matches, so neither can be written from the other."""
        dev, X = prepared(trial, shape=shape)
        flat = dev.transform(X, -1).reshape(X.shape[0], -1)
        per_patch = dev._patch_outputs(X)
        assert np.array_equal(flat.astype(bool), per_patch.any(axis=-1))

    @pytest.mark.parametrize("coalesced", [True, False])
    @pytest.mark.parametrize("trial", range(2))
    def test_class_sums_are_the_weighted_vote_of_the_clauses_that_fired(self, trial, coalesced):
        dev, X = prepared(trial, coalesced=coalesced)
        fired = dev.transform(X, -1).reshape(X.shape[0], -1)
        assert np.allclose(dev.calc_class_sums(X), weighted_sum(dev, fired), atol=1e-4)

    def test_non_coalesced_clauses_only_vote_for_their_own_class(self):
        """A wrong clause to class mapping would still produce plausible sums, just for the wrong class."""
        dev, X = prepared(coalesced=False, n_clauses=9, n_classes=3)
        cfg = dev.config
        per_class = cfg._total_clauses // cfg.n_classes

        dev.clause_weights[:] = 0.0
        dev.clause_weights[1, :] = 5.0  # only class 1 has weight
        dev.ta_states[:] = cfg._include_state - 1  # every clause empty, so all of them fire
        dev.pack_clauses(force_repack=True)

        sums = dev.calc_class_sums(X)
        assert np.all(sums[:, 0] == 0.0) and np.all(sums[:, 2] == 0.0)
        assert np.all(sums[:, 1] == 5.0 * per_class)

    def test_class_sums_do_not_accumulate_across_calls(self):
        """The C adds into the caller's buffer, so a missing zero fill would double every second call."""
        dev, X = prepared()
        first = dev.calc_class_sums(X).copy()
        assert np.array_equal(dev.calc_class_sums(X), first)


class TestEvaluate:
    """The training path. Same firing decision as inference, but it also picks a patch at random."""

    @pytest.mark.parametrize("shape", [CONV, FLAT], ids=["conv", "flat"])
    @pytest.mark.parametrize("trial", range(3))
    def test_a_patch_is_selected_exactly_when_the_clause_fires(self, shape, trial):
        """`selected_pids >= 0` is the training time clause output, so it has to agree with transform."""
        dev, X = prepared(trial, shape=shape)
        fired = dev.transform(X, -1).reshape(X.shape[0], -1)

        for e in range(X.shape[0]):
            buf = buffers(dev, X)
            dev._fit_eval(buf, e, 7 + e)
            assert np.array_equal(buf.selected_pids >= 0, fired[e].astype(bool))

    def test_the_selected_patch_is_one_that_matches(self):
        """Feedback is applied to this patch, so picking a non matching one corrupts the update."""
        dev, X = prepared()
        per_patch = dev._patch_outputs(X)
        density = dev.get_packed_clauses().clause_density

        buf = buffers(dev, X)
        dev._fit_eval(buf, 0, 11)
        for clause in np.flatnonzero(buf.selected_pids >= 0):
            if density[clause] > 0:  # an unconstrained clause picks any patch, all of them match
                assert per_patch[0, clause, buf.selected_pids[clause]] == 1

    def test_the_choice_is_uniform_over_the_matching_patches(self):
        """Reservoir sampling. Picking the first or last match would bias every conv model toward a
        corner of the image, which a "the patch matches" check alone would not notice."""
        dev, X = prepared()
        per_patch = dev._patch_outputs(X)[0]
        density = dev.get_packed_clauses().clause_density

        clause = max(range(dev.config._total_clauses), key=lambda c: per_patch[c].sum() if density[c] > 0 else 0)
        matches = np.flatnonzero(per_patch[clause])
        assert len(matches) >= 4, "the fixture must give some clause a real choice"

        draws = 600
        counts = np.zeros(dev.config._n_patches, dtype=int)
        for seed in range(draws):
            buf = buffers(dev, X)
            dev._fit_eval(buf, 0, seed)
            counts[buf.selected_pids[clause]] += 1

        assert np.array_equal(np.flatnonzero(counts), matches)
        assert counts[matches].min() > 0.5 * draws / len(matches)

    @pytest.mark.parametrize("shape", [CONV, FLAT], ids=["conv", "flat"])
    def test_dropped_clauses_select_nothing(self, shape):
        dev, X = prepared(shape=shape)
        buf = buffers(dev, X, drop=np.ones(dev.config._total_clauses, dtype=np.int8))
        dev._fit_eval(buf, 0, 3)
        assert np.all(buf.selected_pids == -1)

    def test_the_seed_decides_which_matching_patch_is_picked(self):
        dev, X = prepared()
        picks = []
        for seed in (1, 1, 2):
            buf = buffers(dev, X)
            dev._fit_eval(buf, 0, seed)
            picks.append(buf.selected_pids.copy())

        assert np.array_equal(picks[0], picks[1])
        assert not np.array_equal(picks[0], picks[2])

    def test_patch_weights_count_only_the_selected_patch(self):
        dev, X = prepared(track_patch_weights=True)
        buf = buffers(dev, X)
        dev.patch_weights[:] = 0
        dev._fit_eval(buf, 0, 5)

        for clause in range(dev.config._total_clauses):
            pid = buf.selected_pids[clause]
            expected = np.zeros(dev.config._n_patches, dtype=np.int32)
            if pid >= 0:
                expected[pid] = 1
            assert np.array_equal(dev.patch_weights[clause], expected), clause

    def test_untracked_patch_weights_are_left_alone(self):
        dev, X = prepared(track_patch_weights=False)
        before = dev.patch_weights.copy()
        dev._fit_eval(buffers(dev, X), 0, 5)
        assert np.array_equal(dev.patch_weights, before)

    def test_transform_does_not_touch_the_patch_weights(self):
        """Inference goes through a different entry point precisely so it has no side effects."""
        dev, X = prepared(track_patch_weights=True)
        before = dev.patch_weights.copy()
        dev.transform(X, -1)
        dev.calc_class_sums(X)
        assert np.array_equal(dev.patch_weights, before)


class TestCountVotes:
    @pytest.mark.parametrize("coalesced", [True, False])
    def test_votes_match_the_weights_of_the_selected_clauses(self, coalesced):
        """`count_votes` and `calc_class_sums` sum the same weights over different clause sets, so
        restricting the latter to what evaluate selected must reproduce the former."""
        dev, X = prepared(coalesced=coalesced)
        buf = buffers(dev, X)
        dev._fit_eval(buf, 0, 5)
        dev._fit_voting(buf)

        expected = weighted_sum(dev, (buf.selected_pids >= 0)[None])[0]
        assert np.allclose(buf.votes, expected, atol=1e-4)

    def test_votes_do_not_accumulate_across_samples(self):
        """`votes` is reused every sample, so count_votes has to overwrite rather than add."""
        dev, X = prepared()
        buf = buffers(dev, X)
        dev._fit_eval(buf, 0, 5)
        dev._fit_voting(buf)
        first = buf.votes.copy()
        dev._fit_voting(buf)
        assert np.array_equal(buf.votes, first)


class TestThreadDeterminism:
    """Every entry point here is `#pragma omp parallel for`, and the default device is one thread.

    A race would look like a different random draw rather than a failure, since the algorithm is
    stochastic anyway. Identical seeds give identical starting arrays, so a difference is the
    parallelism and nothing else.
    """

    BIG: ClassVar[dict] = {"n_clauses": 512, "dim": (8, 8), "patch_dim": (3, 3), "n_classes": 6, "feat_maxs": 3}

    def _pair(self):
        single, many = thread_pair(**self.BIG)
        rng = np.random.default_rng(31)
        states = np.where(rng.random(single.ta_states.shape) < 0.04, single.config._include_state, single.config._include_state - 1).astype(
            np.uint32
        )
        for dev in (single, many):
            dev.ta_states[:] = states
            dev.pack_clauses(force_repack=True)
        X = np.random.default_rng(32).integers(0, 4, size=(4, *single.config._dim), dtype=np.int32)
        return single, many, X

    @pytest.mark.parametrize("call", ["transform", "patch_outputs", "class_sums"])
    def test_inference_paths_agree(self, call):
        single, many, X = self._pair()
        run = {
            "transform": lambda d: d.transform(X, -1),
            "patch_outputs": lambda d: d._patch_outputs(X),
            "class_sums": lambda d: d.calc_class_sums(X),
        }[call]

        expected = run(single).copy()
        for _ in range(5):  # a race is intermittent, so one agreement proves little
            assert np.array_equal(run(many), expected)

    def test_evaluate_and_count_votes_agree(self):
        single, many, X = self._pair()

        buf = buffers(single, X)
        single._fit_eval(buf, 0, 77)
        single._fit_voting(buf)
        pids, votes = buf.selected_pids.copy(), buf.votes.copy()

        for _ in range(5):
            buf = buffers(many, X)
            many._fit_eval(buf, 0, 77)
            many._fit_voting(buf)
            assert np.array_equal(buf.selected_pids, pids)
            assert np.array_equal(buf.votes, votes)

    def test_patch_weight_counting_agrees(self):
        """Read modify write on a shared array, so the most likely place for a lost update."""
        single, many, X = self._pair()

        for dev in (single, many):
            dev.patch_weights[:] = 0
            for seed in range(6):
                dev._fit_eval(buffers(dev, X), 0, seed)

        assert np.array_equal(many.patch_weights, single.patch_weights)
