import pathlib
import platform
import subprocess
import tempfile
from ctypes import CDLL, POINTER, c_float, c_int, c_int8, c_int32, c_uint64
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .._device_checks import run_compiler
from ..utils import FitBuffers, tqdm_bar
from .base import BaseDevice

int8_p = POINTER(c_int8)
int32_p = POINTER(c_int32)
float_p = POINTER(c_float)


@dataclass(kw_only=True)
class CPUFitBuffers(FitBuffers):
    """Adds the ctypes pointers, derived once per epoch rather than per sample."""

    p_X: Any = field(init=False)
    p_Y: Any = field(init=False)
    p_clause_drop_mask: Any = field(init=False)
    p_selected_pids: Any = field(init=False)
    p_votes: Any = field(init=False)

    def __post_init__(self):
        self.p_X = self.X.ctypes.data_as(int32_p)
        self.p_Y = self.Y.ctypes.data_as(float_p)
        self.p_clause_drop_mask = self.clause_drop_mask.ctypes.data_as(int8_p)
        self.p_selected_pids = self.selected_pids.ctypes.data_as(int32_p)
        self.p_votes = self.votes.ctypes.data_as(float_p)


def read_file(path: pathlib.Path) -> str:
    with open(path) as f:
        return f.read()


class CPUDevice(BaseDevice):
    lib: CDLL

    def dev_init(self):
        self.xp = np

        self._init_clauses()
        self._init_weights()
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

    def _code_sections(self) -> dict[str, str]:
        core = pathlib.Path(__file__).parent
        names = ("cpu.h", "common.h", "rng.h", "feedback.h", "feedback.c", "pack_clauses.c", "evaluate.c", "interpret.c")
        return {name: read_file(core / name) for name in names}

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

        ta_state_p = POINTER(np.ctypeslib.as_ctypes_type(cfg._ta_dtype))
        self.p_ta_states = self.ta_states.ctypes.data_as(ta_state_p)
        self.p_clause_weights = self.clause_weights.ctypes.data_as(float_p)
        self.p_patch_weights = self.patch_weights.ctypes.data_as(int32_p)

        self.p_feat_mins = cfg._feat_mins.ctypes.data_as(int32_p)
        self.p_feat_maxs = cfg._feat_maxs.ctypes.data_as(int32_p)
        self.p_literal_offsets = cfg._literal_offsets.ctypes.data_as(int32_p)

    def set_threads(self, n: int):
        self.lib.set_num_threads(c_int(max(1, n)))

    def pack_clauses(self, force_repack: bool = False, full: bool = False):
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
            c_int(full),
        )

    # == fit steps ==
    def _fit_eval(self, buf: CPUFitBuffers, e: int, rng_key):
        self.lib.evaluate(
            c_uint64(rng_key),
            buf.p_X,
            c_int(e),
            buf.p_clause_drop_mask,
            self.p_clause_position_bounds,
            self.p_clause_feat_bounds,
            self.p_bounded_feat_ids,
            self.p_n_bounded_feats,
            self.p_clause_density,
            buf.p_selected_pids,
            self.p_patch_weights,
        )

    def _fit_voting(self, buf: CPUFitBuffers):
        self.lib.count_votes(buf.p_selected_pids, self.p_clause_weights, buf.p_votes)

    def calc_class_sums(self, X: np.ndarray, batch_size: int = -1, force_repack: bool = False) -> np.ndarray:
        cfg = self.config
        X = np.asarray(X, dtype=np.int32, order="C")
        class_sums = np.zeros((X.shape[0], cfg.n_classes), dtype=np.float32)
        self.pack_clauses(force_repack)

        self.lib.calc_class_sums(
            self.p_clause_weights,
            self.p_clause_position_bounds,
            self.p_clause_feat_bounds,
            self.p_bounded_feat_ids,
            self.p_n_bounded_feats,
            self.p_clause_density,
            X.ctypes.data_as(int32_p),
            c_int(X.shape[0]),
            class_sums.ctypes.data_as(float_p),
        )
        return class_sums

    def transform(self, X: np.ndarray, batch_size: int, force_repack: bool = False) -> np.ndarray:
        """Whether each clause fired on each sample."""
        cfg = self.config
        X = np.ascontiguousarray(X, dtype=np.int32)
        p_X = X.ctypes.data_as(int32_p)
        out = np.zeros((X.shape[0], cfg._total_clauses), dtype=np.int8)
        self.pack_clauses(force_repack)

        for e in tqdm_bar(range(X.shape[0]), desc="Transform"):
            self.lib.calc_clause_outputs(
                self.p_clause_position_bounds,
                self.p_clause_feat_bounds,
                self.p_bounded_feat_ids,
                self.p_n_bounded_feats,
                self.p_clause_density,
                p_X,
                c_int(e),
                out.ctypes.data_as(int8_p),
            )
        return out.reshape(X.shape[0], cfg._n_clause_banks, cfg._n_clauses)

    def _patch_outputs(self, X: np.ndarray, force_repack: bool = False, desc: str = "Transform") -> np.ndarray:
        cfg = self.config
        X = np.ascontiguousarray(X, dtype=np.int32)
        p_X = X.ctypes.data_as(int32_p)
        out = np.zeros((X.shape[0], cfg._total_clauses, cfg._n_patches), dtype=np.int8)
        self.pack_clauses(force_repack)

        for e in tqdm_bar(range(X.shape[0]), desc=desc):
            self.lib.calc_clause_outputs_patchwise(
                self.p_clause_position_bounds,
                self.p_clause_feat_bounds,
                self.p_bounded_feat_ids,
                self.p_n_bounded_feats,
                self.p_clause_density,
                p_X,
                c_int(e),
                out.ctypes.data_as(int8_p),
            )
        return out

    def transform_patchwise(self, X: np.ndarray, batch_size: int, force_repack: bool = False) -> np.ndarray:
        cfg = self.config
        out = self._patch_outputs(X, force_repack)
        return out.reshape(out.shape[0], cfg._n_clause_banks, cfg._n_clauses, cfg._n_patches_y, cfg._n_patches_x)

    def wic(self, class_id: int, polarity: int, pw_th: float = 0.0, force_repack: bool = False) -> np.ndarray:
        cfg = self.config
        self.pack_clauses(force_repack)

        # Per clause max normalisation, on a copy so `patch_weights` is left alone.
        pw = self._to_host(self.patch_weights).astype(np.float32)
        pw_norm = np.ascontiguousarray(pw / (pw.max(axis=-1, keepdims=True) + 1e-7))

        output = np.zeros(cfg._dim, dtype=np.float32)
        self.lib.wic(
            c_int(class_id),
            c_int(polarity),
            self.p_clause_weights,
            self.p_clause_feat_bounds,
            self.p_clause_position_bounds,
            self.p_clause_density,
            pw_norm.ctypes.data_as(float_p),
            self.p_feat_mins,
            self.p_feat_maxs,
            c_float(pw_th),
            output.ctypes.data_as(float_p),
        )
        return output

    def wac(self, X: np.ndarray, target_classes: np.ndarray, polarity: int, force_repack: bool = False) -> np.ndarray:
        cfg = self.config
        N = X.shape[0]

        patch_output = self._patch_outputs(X, force_repack, desc="WAC activations")
        p_patch_output = patch_output.ctypes.data_as(int8_p)

        output = np.zeros((N, *cfg._dim), dtype=np.float32)
        p_output = output.ctypes.data_as(float_p)
        for e in tqdm_bar(range(N), desc="WAC"):
            self.lib.wac_sample(
                c_int(int(target_classes[e])),
                c_int(polarity),
                p_patch_output,
                c_int(e),
                self.p_clause_weights,
                self.p_clause_feat_bounds,
                self.p_clause_density,
                self.p_feat_mins,
                self.p_feat_maxs,
                p_output,
            )
        return output
