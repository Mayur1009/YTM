import numpy as np
import pytest

from ytm._core.backends.base import BaseDevice
from ytm._core.config import BaseTMConfig
from ytm._core.device_config import DeviceConfig

DEFAULTS: dict = {"n_clauses": 8, "s": 10.0, "dim": (4, 4), "n_classes": 3, "seed": 1}


class Device(BaseDevice):
    def dev_init(self):
        self.xp = np
        self._init_clauses()
        self._init_weights()
        self._init_bias()
        self._init_patch_weights()
        self._init_packed_clauses()

    def _to_host(self, arr) -> np.ndarray:
        return arr.copy()

    def pack_clauses(self, force_repack: bool = False): ...
    def fit_epoch(self, X, Y, clause_drop_p, batch_size, **kwargs): ...
    def fit_sample(self, X, Y, e, **kwargs): ...
    def calc_class_sums(self, X, force_repack=False): ...
    def transform(self, X, batch_size, force_repack=False): ...
    def infer(self, X, batch_size): ...
    def transform_patchwise(self, X, batch_size, force_repack=False): ...
    def wic(self, class_id, polarity, pw_th=0.0, force_repack=False): ...
    def wac(self, X, target_classes, polarity, force_repack=False): ...


def make_device(device: str = "cpu:1", **kwargs) -> Device:
    return Device(BaseTMConfig(**{**DEFAULTS, **kwargs}), DeviceConfig(device=device))


class TestConstruction:
    def test_rng_is_offset_from_the_config_seed(self):
        dev = make_device(seed=7, ta_init="middle", weight_init=1.0, negative_clauses=False, bias=False)
        assert np.array_equal(dev._rng.random(3), np.random.default_rng(8).random(3))

    def test_dev_init_ran(self):
        dev = make_device()
        for name in ("ta_states", "clause_weights", "bias", "patch_weights"):
            assert hasattr(dev, name)

    def test_same_seed_gives_identical_arrays(self):
        a, b = make_device(seed=3, ta_init="random"), make_device(seed=3, ta_init="random")
        assert np.array_equal(a.ta_states, b.ta_states)
        assert np.array_equal(a.clause_weights, b.clause_weights)

    def test_different_seeds_give_different_arrays(self):
        a, b = make_device(seed=3, ta_init="random"), make_device(seed=4, ta_init="random")
        assert not np.array_equal(a.ta_states, b.ta_states)

    def test_set_threads_is_unsupported_by_default(self):
        with pytest.raises(NotImplementedError):
            make_device().set_threads(4)


class TestInitClauses:
    def test_shape_and_dtype(self):
        dev = make_device()
        assert dev.ta_states.shape == (dev.config._total_clauses, dev.config._n_literals)
        assert dev.ta_states.dtype == np.uint32

    def test_middle(self):
        dev = make_device(ta_init="middle")
        assert np.all(dev.ta_states == dev.config._include_state - 1)

    def test_explicit_state(self):
        assert np.all(make_device(ta_init=7).ta_states == 7)

    def test_random_spans_the_state_range(self):
        dev = make_device(ta_init="random", n_clauses=64)
        assert dev.ta_states.min() >= 0
        assert dev.ta_states.max() <= dev.config.n_states - 1
        assert len(np.unique(dev.ta_states)) > 2

    def test_random_include_uses_only_the_boundary_states(self):
        dev = make_device(ta_init="random_include", n_clauses=64)
        include = dev.config._include_state
        assert set(np.unique(dev.ta_states)) == {include - 1, include}

    @pytest.mark.parametrize("band", [0, 1, 5, 20])
    def test_random_band_stays_within_the_band(self, band):
        dev = make_device(ta_init=f"random:{band}", n_clauses=64)
        mid = dev.config._include_state - 1
        assert dev.ta_states.min() >= max(0, mid - band)
        assert dev.ta_states.max() <= min(dev.config.n_states - 1, mid + band)


class TestInitWeights:
    def test_shape_and_dtype(self):
        dev = make_device()
        assert dev.clause_weights.shape == (dev.config.n_classes, dev.config._n_clauses)
        assert dev.clause_weights.dtype == np.float32

    def test_constant_magnitude(self):
        assert np.all(np.abs(make_device(weight_init=2.5).clause_weights) == 2.5)

    @pytest.mark.parametrize("weight_init, high", [("random", 1.0), ("random:3", 3.0)])
    def test_random_magnitudes_stay_in_range(self, weight_init, high):
        w = np.abs(make_device(weight_init=weight_init, n_clauses=64).clause_weights)
        assert w.min() >= 0.0
        assert w.max() <= high

    def test_negative_clauses_split_each_class_in_half(self):
        w = make_device(n_clauses=8).clause_weights
        assert np.all((w > 0).sum(axis=1) == 4)
        assert np.all((w < 0).sum(axis=1) == 4)

    def test_no_negative_clauses_keeps_every_weight_positive(self):
        assert np.all(make_device(negative_clauses=False).clause_weights > 0)

    def test_coalesced_permutes_polarity_per_class(self):
        signs = np.sign(make_device(coalesced=True, n_clauses=64).clause_weights)
        assert not np.array_equal(signs[0], signs[1])

    def test_non_coalesced_gives_every_class_the_same_split(self):
        signs = np.sign(make_device(coalesced=False, n_clauses=8).clause_weights)
        assert np.array_equal(signs[0], signs[1])
        assert np.array_equal(signs[0], np.array([1, 1, 1, 1, -1, -1, -1, -1], dtype=np.float32))


class TestInitBias:
    def test_disabled_is_a_single_zero(self):
        dev = make_device(bias=False)
        assert dev.bias.shape == (1,)
        assert dev.bias.dtype == np.float32
        assert np.all(dev.bias == 0)

    def test_constant(self):
        dev = make_device(bias=True, bias_init=2.0)
        assert dev.bias.shape == (3,)
        assert np.all(dev.bias == 2.0)

    def test_random_stays_in_the_unit_range(self):
        dev = make_device(bias=True, bias_init="random")
        assert dev.bias.shape == (3,)
        assert dev.bias.min() >= 0.0
        assert dev.bias.max() <= 1.0


class TestInitPatchWeights:
    def test_tracked(self):
        dev = make_device(dim=(8, 8), patch_dim=(3, 3), track_patch_weights=True)
        assert dev.patch_weights.shape == (dev.config._total_clauses, dev.config._n_patches)
        assert dev.patch_weights.dtype == np.int32
        assert np.all(dev.patch_weights == 0)

    def test_untracked_is_a_placeholder(self):
        assert make_device(track_patch_weights=False).patch_weights.shape == (1, 1)


class TestGetters:
    def test_ta_states_are_reshaped_by_clause_bank(self):
        dev = make_device(coalesced=False)
        cfg = dev.config
        out = dev.get_ta_states()
        assert out.shape == (cfg._n_clause_banks, cfg._n_clauses, cfg._n_literals)
        assert np.array_equal(out.reshape(cfg._total_clauses, cfg._n_literals), dev.ta_states)

    def test_weights_are_returned_as_stored(self):
        dev = make_device()
        assert np.array_equal(dev.get_weights(), dev.clause_weights)

    def test_patch_weights_are_reshaped_by_patch_grid(self):
        dev = make_device(dim=(8, 8), patch_dim=(3, 3))
        cfg = dev.config
        out = dev.get_patch_weights()
        assert out.shape == (cfg._n_clause_banks, cfg._n_clauses, cfg._n_patches_y, cfg._n_patches_x)

    def test_bias_is_returned_when_enabled(self):
        dev = make_device(bias=True, bias_init=2.0)
        assert np.array_equal(dev.get_bias(), dev.bias)

    def test_bias_raises_when_disabled(self):
        with pytest.raises(RuntimeError):
            make_device(bias=False).get_bias()

    def test_patch_weights_raise_when_untracked(self):
        with pytest.raises(RuntimeError):
            make_device(track_patch_weights=False).get_patch_weights()

    @pytest.mark.parametrize(
        "getter, attr",
        [
            ("get_ta_states", "ta_states"),
            ("get_weights", "clause_weights"),
            ("get_patch_weights", "patch_weights"),
        ],
    )
    def test_getters_return_a_copy(self, getter, attr):
        """`_to_host` copies, so mutating the result must not touch device memory."""
        dev = make_device()
        before = getattr(dev, attr).copy()

        getattr(dev, getter)().fill(0)

        assert np.array_equal(getattr(dev, attr), before)


PACKED_SHAPES = [
    ("clause_feat_bounds", lambda c: (c._total_clauses, c._n_raw_patch_feats, 2), np.int32),
    ("clause_position_bounds", lambda c: (c._total_clauses, 4), np.int32),
    ("bounded_feat_ids", lambda c: (c._total_clauses, c._n_raw_patch_feats), np.int32),
    ("n_bounded_feats", lambda c: (c._total_clauses,), np.int32),
    ("clause_density", lambda c: (c._total_clauses,), np.int32),
    ("is_clause_synced", lambda c: (c._total_clauses,), np.int8),
]


class TestInitPackedClauses:
    @pytest.mark.parametrize("field, shape_of, dtype", PACKED_SHAPES, ids=[f[0] for f in PACKED_SHAPES])
    def test_shapes_and_dtypes(self, field, shape_of, dtype):
        """The kernels index these directly, so a wrong shape is an overrun in C."""
        dev = make_device(dim=(8, 8), patch_dim=(3, 3), coalesced=False)
        arr = getattr(dev.packed_clauses, field)
        assert arr.shape == shape_of(dev.config)
        assert arr.dtype == dtype

    def test_starts_unsynced(self):
        """Zero means "needs repacking", so every clause must be packed on the first call."""
        assert np.all(make_device().packed_clauses.is_clause_synced == 0)


class TestGetPackedClauses:
    @pytest.mark.parametrize("field", [f[0] for f in PACKED_SHAPES])
    def test_values_match_the_device_arrays(self, field):
        dev = make_device()
        assert np.array_equal(getattr(dev.get_packed_clauses(), field), getattr(dev.packed_clauses, field))

    @pytest.mark.parametrize("field", [f[0] for f in PACKED_SHAPES])
    def test_returns_copies(self, field):
        dev = make_device()
        before = getattr(dev.packed_clauses, field).copy()

        getattr(dev.get_packed_clauses(), field).fill(0)

        assert np.array_equal(getattr(dev.packed_clauses, field), before)
