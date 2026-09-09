import pathlib

import cupy as cp
import numpy as np

from .base import BaseDevice


def read_file(path: pathlib.Path) -> str:
    with open(path) as f:
        return f.read()


class CUDADevice(BaseDevice):
    cuda_dev: cp.cuda.Device
    module: cp.RawModule

    def dev_init(self): ...

    def _build_code(self) -> str:
        """Generated header plus the shared sources. Subclasses append their own via `super()`."""
        core = pathlib.Path(__file__).parent
        return self.config._header + "".join(read_file(core / name) for name in ("cuda.h", "common.h", "rng.h", "pack_clauses.cu", "interpret.cu"))

    def _kernel_names(self) -> tuple[str, ...]:
        """Entry points to pull out of the module. Subclasses append their own via `super()`."""
        return ("pack_clauses", "wic", "wac")

    def _init_kernels(self):
        """One module for the whole model, so nvrtc runs once instead of per source group."""
        with self.cuda_dev:
            self.module = cp.RawModule(code=self._build_code(), backend="nvrtc", options=())
            for name in self._kernel_names():
                setattr(self, f"k_{name}", self.module.get_function(name))

    def _init_device_arrays(self):
        cfg = self.config
        with self.cuda_dev:
            self.feat_mins_gpu = cp.asarray(cfg._feat_mins, dtype=np.int32)
            self.feat_maxs_gpu = cp.asarray(cfg._feat_maxs, dtype=np.int32)
            self.literal_offsets_gpu = cp.asarray(cfg._literal_offsets, dtype=np.int32)

    def _kernel_config(self, n: int) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        dev = self.device_config
        bs = dev._block_size
        gs = dev._grid_size if dev._grid_size is not None else min((n + bs - 1) // bs, dev._max_grid_size)
        return (gs, 1, 1), (bs, 1, 1)

    def _to_host(self, arr) -> np.ndarray:
        with self.cuda_dev:
            return arr.get()

    def pack_clauses(self, force_repack: bool = False):
        with self.cuda_dev:
            if force_repack:
                self.packed_clauses.is_clause_synced.fill(0)

            self.k_pack_clauses(
                *self._kernel_config(self.config._total_clauses * self.device_config._cuda_props["warp_size"]),
                (
                    self.ta_states,
                    self.feat_mins_gpu,
                    self.feat_maxs_gpu,
                    self.literal_offsets_gpu,
                    self.packed_clauses.clause_position_bounds,
                    self.packed_clauses.clause_feat_bounds,
                    self.packed_clauses.bounded_feat_ids,
                    self.packed_clauses.n_bounded_feats,
                    self.packed_clauses.clause_density,
                    self.packed_clauses.is_clause_synced,
                ),
            )
