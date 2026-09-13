from dataclasses import dataclass
from typing import Literal, TypedDict

from ._device_checks import (
    DEFAULT_COMPILE_FLAGS,
    check_cuda_available,
    parse_device,
    resolve_cuda_props,
    resolve_link_flags,
    resolve_openmp_flags,
    select_compiler,
)


@dataclass()
class DeviceConfig:
    device: Literal["cpu", "cuda"] | str = "cpu:1"

    # cpu
    compile_flags: None | list[str] = None

    # cuda
    grid_size: int | None = None
    block_size: int = 256
    warps_per_clause: int = 1

    def __post_init__(self):
        self._check_device()

    def _check_device(self):
        """Resolve the toolchain and the launch params of the selected device."""
        self._device_kind, n = parse_device(self.device)

        if self._device_kind == "cuda":
            check_cuda_available()
            self._gpu_id = n
            self._cuda_props = resolve_cuda_props(n)
            self._warps_per_clause = max(1, int(self.warps_per_clause))

            warp_size = self._cuda_props["warp_size"]
            block_size = min(max(1, int(self.block_size)), self._cuda_props["max_threads_per_block"])
            self._block_size = max(warp_size, (block_size // warp_size) * warp_size)

            self._max_grid_size = min(self._cuda_props["multiprocessor_count"] * 32, self._cuda_props["max_grid_size"])
            self._grid_size = None if self.grid_size is None else min(max(1, int(self.grid_size)), self._max_grid_size)

            self._n_threads = 1
            self._compiler = None
            self._compiler_flags = []
            self._omp_flags = []
            return

        self._n_threads = n
        self._compiler = select_compiler()
        link_flags = resolve_link_flags(self._compiler)
        if self.compile_flags is None:
            self._compiler_flags = list(DEFAULT_COMPILE_FLAGS) + link_flags
        else:
            self._compiler_flags = list(self.compile_flags)
        self._omp_flags = resolve_openmp_flags(self._compiler, link_flags) if self._n_threads > 1 else []

        if not self._omp_flags:
            self._n_threads = 1

        self._gpu_id = -1
        self._cuda_props = {}
        self._grid_size = None
        self._max_grid_size = 0
        self._block_size = 0
        self._warps_per_clause = 0


class T_DeviceConfig(TypedDict, total=False):
    device: Literal["cpu", "cuda"] | str

    # cpu
    compile_flags: None | list[str]

    # cuda
    grid_size: int | None
    block_size: int
    warps_per_clause: int
