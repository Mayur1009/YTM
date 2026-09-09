import importlib.util

import pytest

HAS_CUPY = importlib.util.find_spec("cupy") is not None

pytestmark = pytest.mark.skipif(not HAS_CUPY, reason="cupy is not installed")

if HAS_CUPY:
    import numpy as np

    from ytm._core.backends.cuda import CUDADevice
    from ytm._core.config import BaseTMConfig
    from ytm._core.device_config import DeviceConfig

    DEFAULTS: dict = {"n_clauses": 4, "s": 10.0, "dim": (8, 8), "n_classes": 2, "seed": 1, "feat_maxs": 7}

    class Device(CUDADevice):
        """Concrete only so the shared module build can be exercised."""

        def fit_epoch(self, X, Y, clause_drop_p, batch_size, **kwargs): ...
        def fit_sample(self, X, Y, e, **kwargs): ...
        def infer(self, X, batch_size): ...
        def transform(self, X, batch_size, force_repack=False): ...
        def transform_patchwise(self, X, batch_size, force_repack=False): ...
        def wic(self, class_id, polarity, pw_th=0.0, force_repack=False): ...
        def wac(self, X, target_classes, polarity, force_repack=False): ...

    def make_device(device: str = "cuda:0", **kwargs) -> Device:
        return Device(BaseTMConfig(**{**DEFAULTS, **kwargs}), DeviceConfig(device=device))


@pytest.fixture(scope="module")
def dev():
    """One nvrtc compile for every test that only reads."""
    return make_device()


class TestBuildCode:
    def test_starts_with_the_generated_header(self, dev):
        assert dev._build_code().startswith(dev.config._header)

    @pytest.mark.parametrize("marker", ["INLINE_FN", "S_INV", "mix64", "geom_sample", "pack_clauses", "wic"])
    def test_shared_sources_are_included(self, dev, marker):
        assert marker in dev._build_code()

    def test_uses_the_cuda_platform_header(self, dev):
        code = dev._build_code()
        assert "cooperative_groups" in code
        assert "__device__ inline" in code
        assert "omp.h" not in code


class TestKernels:
    @pytest.mark.parametrize("name", ["pack_clauses", "wic", "wac"])
    def test_entry_points_are_bound(self, dev, name):
        assert getattr(dev, f"k_{name}") is not None

    def test_one_module_for_the_whole_model(self, dev):
        assert dev.module is not None


class TestKernelConfig:
    def test_block_size_comes_from_the_device_config(self, dev):
        _, block = dev._kernel_config(1)
        assert block == (dev.device_config._block_size, 1, 1)

    def test_grid_covers_the_work(self, dev):
        bs = dev.device_config._block_size
        (gs, _, _), _ = dev._kernel_config(bs * 3)
        assert gs == min(3, dev.device_config._max_grid_size)

    def test_grid_is_capped(self, dev):
        (gs, _, _), _ = dev._kernel_config(10**12)
        assert gs == dev.device_config._max_grid_size

    def test_explicit_grid_size_overrides(self):
        d = make_device(device="cuda:0")
        d.device_config.__dict__["_grid_size"] = 7
        (gs, _, _), _ = d._kernel_config(10**9)
        assert gs == 7


class TestDeviceArrays:
    @pytest.mark.parametrize("name", ["feat_mins_gpu", "feat_maxs_gpu", "literal_offsets_gpu"])
    def test_host_arrays_are_mirrored(self, dev, name):
        gpu = getattr(dev, name)
        host = getattr(dev.config, "_" + name.removesuffix("_gpu"))
        assert gpu.dtype == np.int32
        assert np.array_equal(gpu.get(), host)


class TestToHost:
    def test_returns_numpy(self, dev):
        out = dev._to_host(dev.ta_states)
        assert isinstance(out, np.ndarray)
        assert out.shape == (dev.config._total_clauses, dev.config._n_literals)


class TestSetThreads:
    def test_is_unsupported(self, dev):
        with pytest.raises(NotImplementedError):
            dev.set_threads(4)


class TestPackClauses:
    def test_all_excluded_gives_empty_clauses(self, dev):
        dev.pack_clauses(force_repack=True)
        pc = dev.get_packed_clauses()
        assert np.all(pc.clause_density == 0)
        assert np.all(pc.n_bounded_feats == 0)

    def test_marks_every_clause_synced(self, dev):
        dev.packed_clauses.is_clause_synced.fill(0)
        dev.pack_clauses()
        assert np.all(dev.get_packed_clauses().is_clause_synced == 1)

    def test_an_included_literal_becomes_a_lower_bound(self):
        d = make_device()
        cfg = d.config
        d.ta_states[0, cfg._n_position_feats + 3] = cfg._include_state
        d.pack_clauses(force_repack=True)

        pc = d.get_packed_clauses()
        assert pc.clause_density[0] == 1
        assert pc.n_bounded_feats[0] == 1
        assert pc.clause_feat_bounds[0][0].tolist() == [4, 7]

    def test_a_contradiction_is_marked_invalid(self):
        d = make_device()
        cfg = d.config
        half = cfg._n_literals // 2
        d.ta_states[0, cfg._n_position_feats + 5] = cfg._include_state
        d.ta_states[0, cfg._n_position_feats + 2 + half] = cfg._include_state
        d.pack_clauses(force_repack=True)

        assert d.get_packed_clauses().clause_density[0] == -1
