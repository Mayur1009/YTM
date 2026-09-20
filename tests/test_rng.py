import numpy as np
import pytest
from scipy import stats

from . import rng_harness as rng
from .conftest import DEVICES

ALPHA = 0.001
N = 200_000
SALTS = [0xDEADBEEF, 0xFEEDFACE, 0xCAFEBABE]  # the key-derivation salts used by evaluate and by the update kernels


def numpy_uniforms(seed: int, n: int) -> np.ndarray:
    return np.random.default_rng(seed).random(n).astype(np.float32)


class TestCounterStream:
    """The whole design is `key + counter -> value`, with nothing stored between calls."""

    def test_same_key_gives_the_same_stream(self):
        """Hidden state (a static counter, an uninitialised word) would make identical keys give different streams."""
        assert np.array_equal(rng.uniforms(42, 1000), rng.uniforms(42, 1000))

    def test_counter_indexes_the_stream(self):
        """Two call sites sharing a key must see the same values at the same counter."""
        full = rng.uniforms(7, 100)
        assert np.array_equal(rng.uniforms(7, 40, start=60), full[60:])


class TestUniform:
    @pytest.mark.statistical
    @pytest.mark.parametrize("source", ["c", "numpy"])
    def test_kolmogorov_smirnov(self, source):
        """A biased or clumped uniform would skew every probabilistic feedback decision; numpy is the sanity reference."""
        u = rng.uniforms(1234, N) if source == "c" else numpy_uniforms(1234, N)
        assert stats.kstest(u, "uniform").pvalue > ALPHA

    @pytest.mark.statistical
    def test_same_distribution_as_numpy(self):
        """Two sample, so it compares the empirical distributions directly rather than each against U(0,1)."""
        assert stats.ks_2samp(rng.uniforms(5, N), numpy_uniforms(5, N)).pvalue > ALPHA

    def test_stays_in_the_unit_range(self):
        """Random smoke check of the top of the range; the deterministic pin is `test_never_returns_exactly_one`."""
        u = rng.uniforms(99, N)
        assert u.min() >= 0.0
        assert u.max() < 1.0

    def test_uniform_is_the_raw_bits_scaled(self):
        """The wrapper in `rng_harness` re-implements the hash; this ties it to `rand_uniform` so the two cannot drift apart."""
        u, raw = rng.uniforms(31, 1000, start=5), rng.raw32(31, 1000, start=5)
        assert np.array_equal(u, (raw.astype(np.float32) * np.float32(2.0**-32)).astype(np.float32))

    def test_never_returns_exactly_one(self):
        """Key 12345, counter 6574832 has x >> 32 == 4294967237, which the float32 cast rounds up to 2^32, so u == 1.0.

        Callers gate on `rand < p` (p = 1 when s = 1 makes Type I always fire) and `rand > label_prob`; a draw of exactly 1.0 breaks both.
        """
        assert rng.uniforms(12345, 1, start=6574832)[0] < 1.0

    @pytest.mark.statistical
    def test_every_bit_is_balanced(self):
        """A stuck or biased bit in `mix64` that uniformity alone would not reveal."""
        raw = rng.raw32(2024, N)
        ones = ((raw[:, None] >> np.arange(32, dtype=np.uint32)) & 1).mean(axis=0)
        assert np.abs(ones - 0.5).max() < 0.01


@pytest.mark.statistical
class TestIndependence:
    @pytest.mark.parametrize("lag", [1, 2, 3, 7, 32])
    def test_no_autocorrelation_within_a_stream(self, lag):
        """Correlation between nearby counters would couple successive draws inside one clause update."""
        u = rng.uniforms(13, N)
        assert abs(np.corrcoef(u[:-lag], u[lag:])[0, 1]) < 0.01

    @pytest.mark.parametrize("salt", SALTS)
    def test_sequential_keys_give_uncorrelated_streams(self, salt):
        """Clause 0 and clause 1 get adjacent keys. Correlated streams would make them learn in lockstep."""
        keys = [rng.rng_hash(1234, clause, salt) for clause in range(200)]
        streams = np.stack([rng.uniforms(k, 500) for k in keys])

        off_diagonal = np.corrcoef(streams)[~np.eye(len(keys), dtype=bool)]
        control = np.corrcoef(np.random.default_rng(0).random((len(keys), 500)))[~np.eye(len(keys), dtype=bool)]

        # against numpy on the same shape, since the noise floor is set by the stream length
        assert np.abs(off_diagonal).mean() < 1.5 * np.abs(control).mean()
        assert np.abs(off_diagonal).max() < 1.5 * np.abs(control).max()

    @pytest.mark.parametrize("salt", SALTS)
    def test_first_draw_across_keys_is_uniform(self, salt):
        """Most clauses consume one draw per sample, so the first of each stream is what matters."""
        firsts = np.array([rng.uniforms(rng.rng_hash(99, c, salt), 1)[0] for c in range(20_000)])
        assert stats.kstest(firsts, "uniform").pvalue > ALPHA


class TestHash:
    @pytest.mark.parametrize("salt", SALTS)
    def test_no_collisions_over_the_realistic_key_space(self, salt):
        """A collision means two clauses share a stream for the whole sample."""
        keys = {rng.rng_hash(seed, clause, salt) for seed in range(300) for clause in range(300)}
        assert len(keys) == 300 * 300

    def test_each_argument_position_matters(self):
        """Catches a commutative or degenerate combine (swapped arguments must change the key), which once made rng_hash symmetric."""
        base = rng.rng_hash(1, 2, 3)
        assert rng.rng_hash(9, 2, 3) != base
        assert rng.rng_hash(1, 9, 3) != base
        assert rng.rng_hash(1, 2, 9) != base
        assert rng.rng_hash(2, 1, 3) != base
        assert rng.rng_hash(1, 3, 2) != base

    def test_mix64_is_a_bijection_on_a_sample(self):
        """A collision in the finaliser would make two distinct keys share a stream."""
        xs = np.arange(50_000, dtype=np.uint64)
        assert len({rng.mix64(int(x)) for x in xs}) == len(xs)


class TestGeomSample:
    """Returns the gap to the next success, so the support is {1, 2, 3, ...}."""

    @pytest.mark.statistical
    @pytest.mark.parametrize("p", [0.1, 0.25, 0.5, 0.9])
    def test_matches_the_geometric_distribution(self, p):
        """Chi square against the exact pmf, with a tail bin so the counts sum to n."""
        d = rng.geom(4242, N, p)

        edges, k = [], 1
        while stats.geom(p).pmf(k) * N > 30:
            edges.append(k)
            k += 1
        obs = np.array([(d == e).sum() for e in edges] + [(d > edges[-1]).sum()], float)
        exp = np.array([stats.geom(p).pmf(e) for e in edges] + [stats.geom(p).sf(edges[-1])]) * N

        assert stats.chi2.sf(((obs - exp) ** 2 / exp).sum(), len(obs) - 1) > ALPHA

    @pytest.mark.statistical
    @pytest.mark.parametrize("p", [0.1, 0.5, 0.9])
    def test_same_distribution_as_numpy(self, p):
        """A geometric off by one (support starting at 0) or a wrong log base would shift the whole gap distribution."""
        assert stats.ks_2samp(rng.geom(4242, N, p), np.random.default_rng(0).geometric(p, N)).pvalue > ALPHA

    @pytest.mark.parametrize("p", [0.05, 0.5, 0.99])
    def test_draws_are_whole_numbers_of_at_least_one(self, p):
        """The callers do `li += geom_sample(...)`. A zero would never advance and would hang."""
        d = rng.geom(7, 50_000, p)
        assert d.min() >= 1.0
        assert np.array_equal(d, np.floor(d))

    @pytest.mark.parametrize("p", [1e-10, 1e-30, 1e-38, 1e-45])
    def test_tiny_success_probability_never_yields_nan_or_below_one(self, p):
        """Callers do `li += geom_sample(...)` and cast `(int)li` only inside the loop guard; NaN or 0 would hang or corrupt."""
        d = rng.geom(3, 2000, p)
        assert not np.any(np.isnan(d))
        assert d.min() >= 1.0

    @pytest.mark.parametrize("p", [1.0, 1.5])
    def test_certain_success_is_always_the_next_trial(self, p):
        """p >= 1 must return exactly 1 (s=1 exact-state feedback relies on it), not a logarithm-derived value."""
        assert np.array_equal(rng.geom(11, 100, p), np.ones(100, dtype=np.float32))

    @pytest.mark.parametrize("p", [0.0, -0.5, float("nan")])
    def test_impossible_success_is_infinite(self, p):
        """No finite gap is correct here, so the callers skip the range instead of stepping."""
        assert np.all(np.isinf(rng.geom(11, 100, p)))

    @pytest.mark.parametrize("p, draws", [(0.5, 1), (1.0, 0), (0.0, 0)])
    def test_only_a_real_draw_advances_the_counter(self, p, draws):
        """The degenerate cases return early, so they must not perturb the stream for later calls."""
        assert rng.geom_counter(123, p) == draws


needs_cuda = pytest.mark.skipif("cuda" not in DEVICES, reason="no usable cuda device")


@needs_cuda
class TestCpuCudaParity:
    """The header is shared, so a difference means one build compiled it differently (types, float flags)."""

    def test_uniform_stream_is_bit_identical(self):
        """Integer mixing plus one exact multiply: any mismatch is a real divergence."""
        from . import rng_harness_cuda as cuda

        assert np.array_equal(rng.uniforms(1234, 5000), cuda.uniforms(1234, 5000))
        assert np.array_equal(rng.uniforms(7, 100, start=60), cuda.uniforms(7, 100, start=60))

    def test_hash_is_bit_identical(self):
        """A different 64-bit multiply or shift on the device would give different keys, so different streams per clause."""
        from . import rng_harness_cuda as cuda

        for seed, a, b in [(1, 2, 3), (2**40, 5, 0xDEADBEEF), (0, 0, 0)]:
            assert rng.rng_hash(seed, a, b) == cuda.rng_hash(seed, a, b)

    @pytest.mark.parametrize("p", [0.1, 0.5, 0.9])
    def test_geometric_draws_agree_up_to_libm_rounding(self, p):
        """`logf`/`ceilf` may round differently on host and device, which can move a draw by exactly 1, rarely."""
        from . import rng_harness_cuda as cuda

        cpu, gpu = rng.geom(99, 20_000, p), cuda.geom(99, 20_000, p)
        bad = cpu != gpu
        assert bad.mean() < 1e-3
        assert np.all(np.abs(cpu[bad] - gpu[bad]) == 1.0)
