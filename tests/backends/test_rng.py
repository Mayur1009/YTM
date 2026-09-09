"""Statistical checks on the shared C RNG.

Each test also runs against numpy's PCG64 as a control. The point is not that the two agree on
values, they are different generators, but that ours passes what numpy passes. A control failure
means the test itself is miscalibrated.

A serious verdict needs PractRand or TestU01 on a much longer stream, this is the fast tier that
catches a broken hash rather than a subtly biased one.
"""

import numpy as np
import pytest
import rng_harness as rng
from scipy import stats

N = 200_000
ALPHA = 1e-4  # loose, these run on every commit and must not flake


def numpy_uniforms(seed: int, n: int) -> np.ndarray:
    return np.random.default_rng(seed).random(n, dtype=np.float32)


def numpy_raw32(seed: int, n: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 2**32, size=n, dtype=np.uint64).astype(np.uint32)


class TestReproducible:
    def test_same_key_gives_the_same_stream(self):
        assert np.array_equal(rng.uniforms(42, 1000), rng.uniforms(42, 1000))

    def test_counter_indexes_the_stream(self):
        """Resuming at counter k must match the tail of a run started at 0."""
        whole = rng.uniforms(42, 100)
        tail = rng.uniforms(42, 90, start=10)
        assert np.array_equal(whole[10:], tail)

    def test_different_keys_give_different_streams(self):
        assert not np.array_equal(rng.uniforms(1, 1000), rng.uniforms(2, 1000))


class TestUniformity:
    @pytest.mark.parametrize("source", ["c", "numpy"])
    def test_kolmogorov_smirnov(self, source):
        u = rng.uniforms(7, N) if source == "c" else numpy_uniforms(7, N)
        assert stats.kstest(u, "uniform").pvalue > ALPHA

    @pytest.mark.parametrize("source", ["c", "numpy"])
    def test_chi_square_over_bins(self, source):
        u = rng.uniforms(8, N) if source == "c" else numpy_uniforms(8, N)
        counts, _ = np.histogram(u, bins=256, range=(0.0, 1.0))
        assert stats.chisquare(counts).pvalue > ALPHA

    def test_bounds(self):
        u = rng.uniforms(9, N)
        assert u.min() >= 0.0
        assert u.max() < 1.0

    def test_mean_and_variance(self):
        u = rng.uniforms(10, N)
        assert abs(u.mean() - 0.5) < 0.01
        assert abs(u.var() - 1 / 12) < 0.01


class TestBits:
    @pytest.mark.parametrize("source", ["c", "numpy"])
    def test_every_bit_is_balanced(self, source):
        x = rng.raw32(11, N) if source == "c" else numpy_raw32(11, N)
        freq = np.array([((x >> b) & 1).mean() for b in range(32)])
        assert np.abs(freq - 0.5).max() < 0.01

    @pytest.mark.parametrize("source", ["c", "numpy"])
    def test_bit_pairs_are_uncorrelated(self, source):
        """A weak finaliser usually shows up as structure between output bits."""
        x = rng.raw32(12, 50_000) if source == "c" else numpy_raw32(12, 50_000)
        bits = np.stack([((x >> b) & 1).astype(np.float64) for b in range(32)])
        corr = np.corrcoef(bits)
        off_diagonal = corr[~np.eye(32, dtype=bool)]
        assert np.abs(off_diagonal).max() < 0.02


class TestIndependence:
    @pytest.mark.parametrize("source", ["c", "numpy"])
    @pytest.mark.parametrize("lag", [1, 2, 3, 7, 32])
    def test_no_autocorrelation_within_a_stream(self, source, lag):
        u = rng.uniforms(13, N) if source == "c" else numpy_uniforms(13, N)
        r = np.corrcoef(u[:-lag], u[lag:])[0, 1]
        assert abs(r) < 0.01

    def test_sequential_keys_give_uncorrelated_streams(self):
        """Keys come from `rng_hash(seed, clause, e, salt)` with small sequential clause and e,
        so nearby keys are the realistic collision risk."""
        keys = [rng.rng_hash(1234, clause, 0, 0xDEADBEEF) for clause in range(200)]
        streams = np.stack([rng.uniforms(k, 500) for k in keys])

        off_diagonal = np.corrcoef(streams)[~np.eye(len(keys), dtype=bool)]
        control = np.corrcoef(np.random.default_rng(0).random((len(keys), 500)))[~np.eye(len(keys), dtype=bool)]

        # Compared against numpy on the same shape, since the noise floor is set by the stream length.
        assert np.abs(off_diagonal).mean() < 1.5 * np.abs(control).mean()
        assert np.abs(off_diagonal).max() < 1.5 * np.abs(control).max()

    def test_first_draw_across_keys_is_uniform(self):
        """The first draw per clause is what the kernels actually consume most of."""
        firsts = np.array([rng.uniforms(rng.rng_hash(99, c, 0, 0xDEADBEEF), 1)[0] for c in range(20_000)])
        assert stats.kstest(firsts, "uniform").pvalue > ALPHA

    def test_neighbouring_keys_differ_in_about_half_their_bits(self):
        """Avalanche: one bit of input change should flip ~32 of 64 output bits."""
        a = np.array([rng.rng_hash(1234, c, 0, 0xDEADBEEF) for c in range(2000)], dtype=np.uint64)
        b = np.array([rng.rng_hash(1234, c + 1, 0, 0xDEADBEEF) for c in range(2000)], dtype=np.uint64)
        flipped = np.array([bin(int(x) ^ int(y)).count("1") for x, y in zip(a, b)])
        assert 28 < flipped.mean() < 36


class TestHashCollisions:
    def test_no_collisions_over_the_realistic_key_space(self):
        keys = {rng.rng_hash(1234, clause, e, 0xDEADBEEF) for clause in range(300) for e in range(300)}
        assert len(keys) == 300 * 300

    def test_each_argument_position_matters(self):
        base = rng.rng_hash(1, 2, 3, 4)
        assert rng.rng_hash(9, 2, 3, 4) != base
        assert rng.rng_hash(1, 9, 3, 4) != base
        assert rng.rng_hash(1, 2, 9, 4) != base
        assert rng.rng_hash(1, 2, 3, 9) != base

    def test_mix64_is_a_bijection_on_a_sample(self):
        xs = np.arange(50_000, dtype=np.uint64)
        assert len({rng.mix64(int(x)) for x in xs} ) == len(xs)


class TestGeomSample:
    @pytest.mark.parametrize("p", [0.5, 0.2, 0.1, 0.02])
    def test_matches_the_geometric_distribution(self, p):
        draws = rng.geom(21, 100_000, p)
        assert draws.min() >= 1

        # bin the tail together, the chi-square needs adequate expected counts per bin
        top = int(stats.geom.ppf(0.99, p))
        observed = np.bincount(np.clip(draws, 1, top), minlength=top + 1)[1:]
        expected = np.diff(np.concatenate([[0.0], stats.geom.cdf(np.arange(1, top), p), [1.0]])) * len(draws)
        assert stats.chisquare(observed, expected).pvalue > ALPHA

    @pytest.mark.parametrize("p", [0.5, 0.1, 0.02])
    def test_mean_matches_one_over_p(self, p):
        draws = rng.geom(22, 200_000, p)
        assert abs(draws.mean() - 1 / p) < 0.05 / p

    def test_is_reproducible(self):
        assert np.array_equal(rng.geom(23, 500, 0.1), rng.geom(23, 500, 0.1))
