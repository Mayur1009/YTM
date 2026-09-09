import importlib.util
from dataclasses import asdict, fields
from unittest import mock

import pytest

import ytm._core.device_config as dc
from ytm._core.device_config import DeviceConfig

HAS_CUPY = importlib.util.find_spec("cupy") is not None

FAKE_PROPS = {
    "max_threads_per_block": 1024,
    "multiprocessor_count": 84,
    "warp_size": 32,
    "max_grid_size": 2147483647,
}

DERIVED = (
    "_device_kind",
    "_n_threads",
    "_gpu_id",
    "_compiler",
    "_compiler_flags",
    "_omp_flags",
    "_cuda_props",
    "_grid_size",
    "_max_grid_size",
    "_block_size",
    "_warps_per_clause",
)


@pytest.fixture
def fake_cuda():
    """Resolve the cuda branch against fixed properties, so it runs without a gpu."""
    with (
        mock.patch.object(dc, "check_cuda_available", lambda: None),
        mock.patch.object(dc, "resolve_cuda_props", lambda gpu_id: dict(FAKE_PROPS)),
    ):
        yield


class TestDeviceString:
    @pytest.mark.parametrize(
        "device, kind, n_threads, gpu_id",
        [
            ("cpu", "cpu", 1, -1),
            ("cpu:1", "cpu", 1, -1),
            ("cpu:0", "cpu", 1, -1),
            ("cpu:-4", "cpu", 1, -1),
        ],
    )
    def test_cpu_forms(self, device, kind, n_threads, gpu_id):
        cfg = DeviceConfig(device=device)
        assert (cfg._device_kind, cfg._n_threads, cfg._gpu_id) == (kind, n_threads, gpu_id)

    @pytest.mark.parametrize("device", ["CPU", " cpu ", "Cpu:1"])
    def test_case_and_whitespace_are_normalised(self, device):
        assert DeviceConfig(device=device)._device_kind == "cpu"

    @pytest.mark.parametrize("device", ["gpu", "", "cuda:x", "cpu:abc", "cpu:1:2", "tpu:0"])
    def test_rejects_unknown_forms(self, device):
        with pytest.raises(ValueError):
            DeviceConfig(device=device)


class TestCpuResolution:
    def test_compiler_is_resolved(self):
        cfg = DeviceConfig(device="cpu")
        assert cfg._compiler in ("clang", "gcc")
        assert cfg._compiler_flags == dc.DEFAULT_COMPILE_FLAGS
        assert cfg._compiler_flags is not dc.DEFAULT_COMPILE_FLAGS

    def test_custom_compile_flags_are_used(self):
        flags = ["-shared", "-fPIC", "-O2"]
        assert DeviceConfig(device="cpu", compile_flags=flags)._compiler_flags == flags

    def test_single_thread_skips_the_openmp_probe(self):
        assert DeviceConfig(device="cpu:1")._omp_flags == []

    def test_threads_and_openmp_agree(self):
        """Either openmp resolved and the thread count stands, or it did not and we fell back to 1."""
        cfg = DeviceConfig(device="cpu:4")
        assert (cfg._n_threads == 4) == bool(cfg._omp_flags)
        assert cfg._n_threads in (1, 4)

    def test_cuda_params_are_inert(self):
        cfg = DeviceConfig(device="cpu:2", block_size=512, grid_size=64, warps_per_clause=4)
        assert cfg._cuda_props == {}
        assert cfg._gpu_id == -1
        assert (cfg._block_size, cfg._grid_size, cfg._max_grid_size, cfg._warps_per_clause) == (0, None, 0, 0)


class TestCudaResolution:
    def test_toolchain_params_are_inert(self, fake_cuda):
        cfg = DeviceConfig(device="cuda:0")
        assert cfg._compiler is None
        assert cfg._compiler_flags == []
        assert cfg._omp_flags == []
        assert cfg._n_threads == 1

    @pytest.mark.parametrize("device, gpu_id", [("cuda", 0), ("cuda:0", 0), ("cuda:3", 3), ("cuda:-2", 0)])
    def test_gpu_id(self, fake_cuda, device, gpu_id):
        assert DeviceConfig(device=device)._gpu_id == gpu_id

    def test_props_are_captured(self, fake_cuda):
        assert DeviceConfig(device="cuda:0")._cuda_props == FAKE_PROPS

    @pytest.mark.parametrize(
        "block_size, expected",
        [(256, 256), (512, 512), (100, 96), (33, 32), (31, 32), (1, 32), (0, 32), (1023, 992), (4096, 1024)],
    )
    def test_block_size_is_capped_and_warp_aligned(self, fake_cuda, block_size, expected):
        cfg = DeviceConfig(device="cuda:0", block_size=block_size)
        assert cfg._block_size == expected
        assert cfg._block_size % FAKE_PROPS["warp_size"] == 0
        assert cfg._block_size <= FAKE_PROPS["max_threads_per_block"]

    @pytest.mark.parametrize("grid_size, expected", [(None, None), (8, 8), (2688, 2688), (999999, 2688), (0, 1)])
    def test_grid_size_is_capped(self, fake_cuda, grid_size, expected):
        assert DeviceConfig(device="cuda:0", grid_size=grid_size)._grid_size == expected

    def test_max_grid_size_is_the_smaller_of_the_two_limits(self, fake_cuda):
        assert DeviceConfig(device="cuda:0")._max_grid_size == 84 * 32

    @pytest.mark.parametrize("warps, expected", [(1, 1), (4, 4), (0, 1), (-2, 1)])
    def test_warps_per_clause_clamped(self, fake_cuda, warps, expected):
        assert DeviceConfig(device="cuda:0", warps_per_clause=warps)._warps_per_clause == expected


@pytest.mark.skipif(HAS_CUPY, reason="cupy is installed, so the missing-cupy path cannot be exercised")
class TestCudaUnavailable:
    @pytest.mark.parametrize("device", ["cuda", "cuda:0", "cuda:2"])
    def test_raises_without_cupy(self, device):
        with pytest.raises(ImportError):
            DeviceConfig(device=device)


class TestConventions:
    """Fields are user input, underscore attributes are the resolved machine-local state."""

    def test_no_field_is_underscore_prefixed(self):
        assert [f.name for f in fields(DeviceConfig) if f.name.startswith("_")] == []

    def test_asdict_has_no_derived_keys(self):
        assert [k for k in asdict(DeviceConfig(device="cpu")) if k.startswith("_")] == []

    def test_every_derived_attr_is_set_on_cpu(self):
        cfg = DeviceConfig(device="cpu:2")
        assert [name for name in DERIVED if not hasattr(cfg, name)] == []

    def test_every_derived_attr_is_set_on_cuda(self, fake_cuda):
        cfg = DeviceConfig(device="cuda:0")
        assert [name for name in DERIVED if not hasattr(cfg, name)] == []

    def test_user_input_is_not_mutated(self):
        cfg = DeviceConfig(device="CPU:8", block_size=100, grid_size=0, warps_per_clause=0)
        assert cfg.device == "CPU:8"
        assert cfg.block_size == 100
        assert cfg.grid_size == 0
        assert cfg.warps_per_clause == 0
