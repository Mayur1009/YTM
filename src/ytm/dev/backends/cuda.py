import numpy as np
import pathlib
from pycuda.driver import Context as ctx, mem_alloc, memcpy_dtoh, memcpy_htod, memset_d32  # pyright: ignore[reportAttributeAccessIssue]
import pycuda.curandom as curandom
from pycuda.gpuarray import to_gpu
from pycuda.compiler import SourceModule
from pycuda.driver import device_attribute, DeviceAllocation  # pyright: ignore[reportAttributeAccessIssue]
from ..utils import read_file
from .base import Backend


class CUDABackend(Backend):
    def __init__(
        self,
        header: str = "",
        grid_size: int | None = None,
        block_size: int = 128,
        seed: int | None = None,
    ):
        self.header = header
        self.grid_size = grid_size
        self.block_size = block_size
        self.seed = seed
        self._pycuda_init()
        self._set_rng(self.seed)
        self._init_kernels()

    def _pycuda_init(self):
        import pycuda.autoprimaryctx  # pyright: ignore[reportUnusedImport] # noqa: F401

        self.device = ctx.get_device()
        self.ctx = ctx.get_current()
        attrs = self.device.get_attributes()
        self.dev_props = {
            "max_threads_per_block": attrs[device_attribute.MAX_THREADS_PER_BLOCK],
            "max_block_dim_x": attrs[device_attribute.MAX_BLOCK_DIM_X],
            "max_grid_dim_x": attrs[device_attribute.MAX_GRID_DIM_X],
            "multiprocessor_count": attrs[device_attribute.MULTIPROCESSOR_COUNT],
        }

    def _kernel_config(self, data_size):
        block_size = min(self.block_size, self.dev_props["max_threads_per_block"])
        warp_size = 32
        block_size = (block_size + warp_size - 1) // warp_size * warp_size

        if self.grid_size is None:
            # Calculate grid size
            grid_size = (data_size + block_size - 1) // block_size
            max_blocks = min(65535, self.dev_props["multiprocessor_count"] * 8)
            grid_size = min(grid_size, max_blocks)
        else:
            grid_size = self.grid_size

        return (grid_size, 1, 1), (block_size, 1, 1)

    def _init_kernels(self):
        current_dir = pathlib.Path(__file__).parent

        kernel_str = read_file("impl.cu", current_dir)
        mod_kernels = SourceModule(
            self.header + kernel_str,
            options=["-O3", "--use_fast_math"],
            no_extern_c=True,
        )
        self._kernel_encode_batch = mod_kernels.get_function("encode_batch")
        self._kernel_encode_batch.prepare("PPi")

        self._kernel_pack_clauses = mod_kernels.get_function("pack_clauses")
        self._kernel_pack_clauses.prepare("PPP")

        self._kernel_fast_eval = mod_kernels.get_function("fast_eval")
        self._kernel_fast_eval.prepare("PPPPPPi")

        self._kernel_select_active = mod_kernels.get_function("select_active")
        self._kernel_select_active.prepare("PPPPPPP")

        self._kernel_calc_class_sums_infer_batch = mod_kernels.get_function("calc_class_sums_infer_batch")
        self._kernel_calc_class_sums_infer_batch.prepare("PPPPiPP")

        self._kernel_evidence_to_update_prob = mod_kernels.get_function("evidence_to_update_prob")
        self._kernel_evidence_to_update_prob.prepare("PPPPPiPP")

        self._kernel_clause_update = mod_kernels.get_function("clause_update")
        self._kernel_clause_update.prepare("PPPPPPPPPPPi")

        self._kernel_transform = mod_kernels.get_function("transform")
        self._kernel_transform.prepare("PPPiPP")

        self._kernel_transform_patchwise = mod_kernels.get_function("transform_patchwise")
        self._kernel_transform_patchwise.prepare("PPPiPP")

    def _set_rng(self, seed: int | None):
        self.rng_dev = (
            curandom.XORWOWRandomNumberGenerator()
            if seed is None
            else curandom.XORWOWRandomNumberGenerator(
                lambda count: to_gpu(np.array([(seed + i) for i in range(1, count + 1)], dtype=np.int32))  # pyright: ignore[reportOptionalOperand]
            )
        )

    def _get_rng(self):
        return self.rng_dev

    def allocate(self, size: int) -> DeviceAllocation:
        return mem_alloc(size)

    def to_device(self, dev, host: np.ndarray) -> None:
        memcpy_htod(dev, host)

    def to_host(self, host: np.ndarray, dev) -> None:
        memcpy_dtoh(host, dev)

    def memset(self, dev, value: int, size: int) -> None:
        memset_d32(dev, value, size)

    def encode_batch(self, X: DeviceAllocation, encoded_X: DeviceAllocation, N: int, n_patches: int) -> None:
        kconf = self._kernel_config(N * n_patches)
        self._kernel_encode_batch.prepared_call(*kconf, X, encoded_X, np.int32(N))
        self.ctx.synchronize()
