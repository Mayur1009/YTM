import pathlib
import platform
import subprocess
import tempfile
from ctypes import CDLL, POINTER, c_float, c_int, c_int8, c_int32, c_uint32

import numpy as np

from .._device_checks import run_compiler
from .base import BaseDevice

int8_p = POINTER(c_int8)
int32_p = POINTER(c_int32)
uint32_p = POINTER(c_uint32)
float_p = POINTER(c_float)


def read_file(path: pathlib.Path) -> str:
    with open(path) as f:
        return f.read()


class CPUDevice(BaseDevice):
    lib: CDLL

    def dev_init(self):
        self.xp = np

        self._init_clauses()
        self._init_weights()
        self._init_bias()
        self._init_patch_weights()
        self._init_packed_clauses()

        self._init_lib()
        self._init_pointers()

        self.set_threads(self.device_config._n_threads)

    def _to_host(self, arr) -> np.ndarray:
        return arr.copy()

    def _compile_code(self, code: str) -> CDLL:
        dev = self.device_config
        assert dev._compiler is not None, "a cpu device always resolves a compiler"

        with tempfile.NamedTemporaryFile(suffix=".c", mode="w", delete=False) as f:
            f.write(code)
            c_file = f.name

        so_file = c_file.replace(".c", ".dll" if platform.system() == "Windows" else ".so")

        try:
            run_compiler([dev._compiler] + dev._compiler_flags + dev._omp_flags + [c_file, "-o", so_file])
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to compile. Compiler output:\n{e.stdout.decode()}\nError: {e.stderr.decode()}") from None

        return CDLL(so_file)

    def _build_code(self) -> str:
        """Generated header plus the shared sources. Subclasses append their own via `super()`."""
        core = pathlib.Path(__file__).parent
        return self.config._header + "".join(
            read_file(core / name) for name in ("cpu.h", "common.h", "rng.h", "pack_clauses.c", "inference.c", "interpret.c")
        )

    def _init_lib(self):
        self.lib = self._compile_code(self._build_code())

    def _init_pointers(self):
        cfg = self.config
        pc = self.packed_clauses

        self.p_clause_feat_bounds = pc.clause_feat_bounds.ctypes.data_as(int32_p)
        self.p_clause_position_bounds = pc.clause_position_bounds.ctypes.data_as(int32_p)
        self.p_bounded_feat_ids = pc.bounded_feat_ids.ctypes.data_as(int32_p)
        self.p_n_bounded_feats = pc.n_bounded_feats.ctypes.data_as(int32_p)
        self.p_clause_density = pc.clause_density.ctypes.data_as(int32_p)
        self.p_is_clause_synced = pc.is_clause_synced.ctypes.data_as(int8_p)

        self.p_ta_states = self.ta_states.ctypes.data_as(uint32_p)
        self.p_clause_weights = self.clause_weights.ctypes.data_as(float_p)
        self.p_bias = self.bias.ctypes.data_as(float_p)
        self.p_patch_weights = self.patch_weights.ctypes.data_as(int32_p)

        self.p_feat_mins = cfg._feat_mins.ctypes.data_as(int32_p)
        self.p_feat_maxs = cfg._feat_maxs.ctypes.data_as(int32_p)
        self.p_literal_offsets = cfg._literal_offsets.ctypes.data_as(int32_p)

    def set_threads(self, n: int):
        self.lib.set_num_threads(c_int(max(1, n)))

    def pack_clauses(self, force_repack: bool = False):
        if force_repack:
            self.packed_clauses.is_clause_synced.fill(0)

        self.lib.pack_clauses(
            self.p_ta_states,
            self.p_feat_mins,
            self.p_feat_maxs,
            self.p_literal_offsets,
            self.p_clause_position_bounds,
            self.p_clause_feat_bounds,
            self.p_bounded_feat_ids,
            self.p_n_bounded_feats,
            self.p_clause_density,
            self.p_is_clause_synced,
        )
