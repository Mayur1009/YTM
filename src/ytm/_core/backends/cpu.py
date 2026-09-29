import pathlib
import warnings
from ctypes import CDLL, POINTER, c_float, c_int, c_int8, c_int32, c_uint64
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..utils import FitBuffers, read_file, tqdm_bar
from .base import BaseDevice
from .toolchain import Toolchain

int8_p = POINTER(c_int8)
int32_p = POINTER(c_int32)
float_p = POINTER(c_float)


@dataclass(kw_only=True)
class CPUFitBuffers(FitBuffers):
    """Adds the ctypes pointers, derived once per epoch rather than per sample."""

    p_X: Any = field(init=False)
    p_Y: Any = field(init=False)
    p_clause_drop_mask: Any = field(init=False)
    p_clause_output: Any = field(init=False)
    p_selected_pids: Any = field(init=False)
    p_votes: Any = field(init=False)

    def __post_init__(self):
        self.p_X = self.X.ctypes.data_as(POINTER(np.ctypeslib.as_ctypes_type(self.X.dtype)))
        self.p_Y = self.Y.ctypes.data_as(float_p)
        self.p_clause_drop_mask = self.clause_drop_mask.ctypes.data_as(int8_p)
        self.p_clause_output = self.clause_output.ctypes.data_as(int8_p)
        self.p_selected_pids = self.selected_pids.ctypes.data_as(POINTER(np.ctypeslib.as_ctypes_type(self.selected_pids.dtype)))
        self.p_votes = self.votes.ctypes.data_as(float_p)


class CPUDevice(BaseDevice):
    xp = np

    def _setup(self):
        self._n_threads = self.device_config.n
        self.toolchain = Toolchain(openmp=self._n_threads > 1)
        if self._n_threads > 1 and not self.toolchain.is_openmp_working:
            warnings.warn(
                f"OpenMP is not usable with {' '.join(self.toolchain.comp)}, continuing with single thread. "
                f"Install an OpenMP runtime (`pixi add llvm-openmp`, `conda install llvm-openmp`, `brew install libomp`), point $OMP_PREFIX "
                f"at one, or set $CC to a compiler that has it. Tried:\n" + "\n".join(self.toolchain.omp_failures),
                stacklevel=2,
            )
            self._n_threads = 1

        self.lib: CDLL = self.toolchain.compile(self._build_code())
        self.set_threads(self._n_threads)

    def _to_host(self, arr) -> np.ndarray:
        return arr.copy()

    def _to_dev(self, arr: np.ndarray) -> np.ndarray:
        return np.asarray(arr, order="C")

    def _code_sections(self) -> dict[str, str]:
        core = pathlib.Path(__file__).parent
        names = ("cpu.h", "common.h", "rng.h", "feedback.h", "pack_clauses.h", "pack_clauses.c", "evaluate.c", "interpret.c")
        return {name: read_file(core / name) for name in names}

    def _bind(self):
        cfg = self.config
        pc = self.packed_clauses

        ta_state_p = POINTER(np.ctypeslib.as_ctypes_type(cfg._ta_dtype))
        fbound_p = POINTER(np.ctypeslib.as_ctypes_type(cfg._fbound_dtype))
        pbound_p = POINTER(np.ctypeslib.as_ctypes_type(cfg._pbound_dtype))
        nfeat_p = POINTER(np.ctypeslib.as_ctypes_type(cfg._nfeat_dtype))
        nlits_p = POINTER(np.ctypeslib.as_ctypes_type(cfg._nlits_dtype))

        self.p_clause_feat_bounds = pc.clause_feat_bounds.ctypes.data_as(fbound_p)
        self.p_clause_position_bounds = pc.clause_position_bounds.ctypes.data_as(pbound_p)
        self.p_clause_feat_ids = pc.clause_feat_ids.ctypes.data_as(nfeat_p)
        self.p_clause_n_feats = pc.clause_n_feats.ctypes.data_as(nfeat_p)
        self.p_has_contra = pc.has_contra.ctypes.data_as(int8_p)
        self.p_clause_len = pc.clause_len.ctypes.data_as(nlits_p)
        self.p_is_clause_synced = pc.is_clause_synced.ctypes.data_as(int8_p)

        self.p_ta_states = self.ta_states.ctypes.data_as(ta_state_p)
        self.p_clause_weights = self.clause_weights.ctypes.data_as(float_p)
        self.p_patch_weights = self.patch_weights.ctypes.data_as(int32_p)

        self.p_therm_bits = self.therm_bits.ctypes.data_as(fbound_p)
        self.p_literal_offsets = self.literal_offsets.ctypes.data_as(nlits_p)

    def set_threads(self, n: int):
        self.lib.set_num_threads(c_int(max(1, n)))

    def pack_clauses(self, force_repack: bool = False, full: bool = False):
        if force_repack:
            self.packed_clauses.is_clause_synced.fill(0)

        self.lib.pack_clauses(
            self.p_ta_states,
            self.p_therm_bits,
            self.p_literal_offsets,
            self.p_clause_position_bounds,
            self.p_clause_feat_bounds,
            self.p_clause_feat_ids,
            self.p_clause_n_feats,
            self.p_has_contra,
            self.p_clause_len,
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
            self.p_clause_feat_ids,
            self.p_clause_n_feats,
            self.p_has_contra,
            self.p_clause_len,
            buf.p_clause_output,
            buf.p_selected_pids,
            self.p_patch_weights,
        )

    def _fit_voting(self, buf: CPUFitBuffers):
        self.lib.count_votes(buf.p_clause_output, self.p_clause_weights, buf.p_votes)

    def calc_class_sums(self, X: np.ndarray, batch_size: int = -1, force_repack: bool = False) -> np.ndarray:
        cfg = self.config
        X = np.ascontiguousarray(X, dtype=cfg._fbound_dtype)
        class_sums = np.zeros((X.shape[0], cfg.n_classes), dtype=np.float32)
        self.pack_clauses(force_repack)

        self.lib.calc_class_sums(
            self.p_clause_weights,
            self.p_clause_position_bounds,
            self.p_clause_feat_bounds,
            self.p_clause_feat_ids,
            self.p_clause_n_feats,
            self.p_has_contra,
            self.p_clause_len,
            X.ctypes.data_as(POINTER(np.ctypeslib.as_ctypes_type(cfg._fbound_dtype))),
            c_int(X.shape[0]),
            class_sums.ctypes.data_as(float_p),
        )
        return class_sums

    def transform(self, X: np.ndarray, batch_size: int = -1, force_repack: bool = False) -> np.ndarray:
        """Whether each clause fired on each sample."""
        cfg = self.config
        X = np.ascontiguousarray(X, dtype=cfg._fbound_dtype)
        p_X = X.ctypes.data_as(POINTER(np.ctypeslib.as_ctypes_type(cfg._fbound_dtype)))
        out = np.zeros((X.shape[0], cfg._total_clauses), dtype=np.int8)
        self.pack_clauses(force_repack)

        for e in tqdm_bar(range(X.shape[0]), desc="Transform"):
            self.lib.calc_clause_outputs(
                self.p_clause_position_bounds,
                self.p_clause_feat_bounds,
                self.p_clause_feat_ids,
                self.p_clause_n_feats,
                self.p_has_contra,
                self.p_clause_len,
                p_X,
                c_int(e),
                out.ctypes.data_as(int8_p),
            )
        return out.reshape(X.shape[0], cfg._n_clause_banks, cfg._n_clauses)

    def _patch_outputs(self, X: np.ndarray, batch_size: int = -1, force_repack: bool = False, desc: str = "Transform") -> np.ndarray:
        cfg = self.config
        X = np.ascontiguousarray(X, dtype=cfg._fbound_dtype)
        p_X = X.ctypes.data_as(POINTER(np.ctypeslib.as_ctypes_type(cfg._fbound_dtype)))
        out = np.zeros((X.shape[0], cfg._total_clauses, cfg._n_patches), dtype=np.int8)
        self.pack_clauses(force_repack)

        for e in tqdm_bar(range(X.shape[0]), desc=desc):
            self.lib.calc_clause_outputs_patchwise(
                self.p_clause_position_bounds,
                self.p_clause_feat_bounds,
                self.p_clause_feat_ids,
                self.p_clause_n_feats,
                self.p_has_contra,
                self.p_clause_len,
                p_X,
                c_int(e),
                out.ctypes.data_as(int8_p),
            )
        return out

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
            self.p_clause_feat_ids,
            self.p_clause_n_feats,
            self.p_clause_position_bounds,
            self.p_has_contra,
            pw_norm.ctypes.data_as(float_p),
            self.p_therm_bits,
            c_float(pw_th),
            output.ctypes.data_as(float_p),
        )
        return output

    def wac(self, X: np.ndarray, target_classes: np.ndarray, polarity: int, batch_size: int = -1, force_repack: bool = False) -> np.ndarray:
        cfg = self.config
        N = X.shape[0]
        output = np.zeros((N, *cfg._dim), dtype=np.float32)

        for i, end, bs in self._batches(N, batch_size, "WAC"):
            patch_output = self._patch_outputs(X[i:end], -1, force_repack and i == 0, desc="WAC activations")
            p_patch_output = patch_output.ctypes.data_as(int8_p)
            out_b = output[i:end]
            p_out_b = out_b.ctypes.data_as(float_p)

            for e in range(bs):
                self.lib.wac_sample(
                    c_int(int(target_classes[i + e])),
                    c_int(polarity),
                    p_patch_output,
                    c_int(e),
                    self.p_clause_weights,
                    self.p_clause_feat_bounds,
                    self.p_clause_feat_ids,
                    self.p_clause_n_feats,
                    self.p_has_contra,
                    self.p_therm_bits,
                    p_out_b,
                )
        return output
