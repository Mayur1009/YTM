import pathlib
import warnings
from ctypes import CDLL, POINTER, c_float, c_int, c_int8, c_int32, c_uint32, c_uint64

import numpy as np

from ..utils import read_file
from .base import BaseDevice, BuffersT, ConfigT
from .toolchain import Toolchain

int8_p = POINTER(c_int8)
int32_p = POINTER(c_int32)
uint32_p = POINTER(c_uint32)
float_p = POINTER(c_float)


class CPUDevice(BaseDevice[BuffersT, ConfigT]):
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
        names = ("cpu.h", "common.h", "rng.h", "feedback.h", "pack_clauses.h", "pack_clauses.c", "evaluate.c", "interpret.c", "feedback.c")
        return {name: read_file(core / name) for name in names}

    def _fit_apply_fb(self, e: int) -> None:
        """T1b first, then T1a and T2, so a full clause's decrements land before its T2 increments."""
        buf = self.buf
        rng_key = buf.rng_keys[e]
        self.lib.update_t1b(
            c_uint64(rng_key),
            buf.p_t1b_n,
            buf.p_t1b_cid,
            buf.p_t1b_cnt,
            buf.p_lit_drop_mask,
            self.p_ta_states,
            self.p_is_clause_synced,
        )
        self.lib.update_t1a_t2(
            c_uint64(rng_key),
            c_int(e),
            buf.p_selected_pids,
            buf.p_X,
            self.p_literal_offsets,
            buf.p_lit_drop_mask,
            buf.p_t1a_t2_n,
            buf.p_t1a_t2_cid,
            buf.p_t1a_cnt,
            buf.p_t2_cnt,
            self.p_ta_states,
            self.p_is_clause_synced,
        )

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
    def _fit_eval(self, e: int):
        buf = self.buf
        rng_key = buf.rng_keys[e]
        self.lib.eval_fit(
            c_uint64(rng_key),
            buf.p_X,
            c_int(e),
            buf.p_clause_drop_mask,
            buf.p_class_active,
            buf.p_lit_drop_mask,
            self.p_literal_offsets,
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

    def _fit_voting(self):
        buf = self.buf
        self.lib.vote_sum(buf.p_clause_output, self.p_clause_weights, c_int(1), buf.p_votes)

    def _clause_outputs(self, Xb: np.ndarray, bs: int) -> np.ndarray:
        cfg = self.config
        Xb = np.ascontiguousarray(Xb, dtype=cfg._fbound_dtype)
        clause_outputs = np.empty((bs, cfg._total_clauses), dtype=np.int8)
        self.lib.eval_infer(
            Xb.ctypes.data_as(POINTER(np.ctypeslib.as_ctypes_type(cfg._fbound_dtype))),
            c_int(bs),
            self.p_clause_position_bounds,
            self.p_clause_feat_bounds,
            self.p_clause_feat_ids,
            self.p_clause_n_feats,
            self.p_has_contra,
            self.p_clause_len,
            clause_outputs.ctypes.data_as(int8_p),
        )
        return clause_outputs

    def _infer_batch(self, Xb: np.ndarray, bs: int) -> np.ndarray:
        class_sums = np.empty((bs, self.config.n_classes), dtype=np.float32)
        clause_outputs = self._clause_outputs(Xb, bs)
        self.lib.vote_sum(clause_outputs.ctypes.data_as(int8_p), self.p_clause_weights, c_int(bs), class_sums.ctypes.data_as(float_p))
        return class_sums

    def transform(self, X: np.ndarray, batch_size: int = -1, force_repack: bool = False) -> np.ndarray:
        """Whether each clause fired on each sample."""
        cfg = self.config
        N = X.shape[0]
        out = np.empty((N, cfg._total_clauses), dtype=np.int8)
        self.pack_clauses(force_repack)

        for start, end in self._batches(N, batch_size, "Transform"):
            out[start:end] = self._clause_outputs(X[start:end], end - start)

        return out.reshape(N, cfg._n_clause_banks, cfg._n_clauses)

    def _patch_outputs(self, X: np.ndarray, batch_size: int = -1, force_repack: bool = False, desc: str = "Transform") -> np.ndarray:
        cfg = self.config
        N = X.shape[0]
        out = np.empty((N, cfg._total_clauses, cfg._n_patches), dtype=np.int8)
        self.pack_clauses(force_repack)

        for start, end in self._batches(N, batch_size, desc):
            Xb = np.ascontiguousarray(X[start:end], dtype=cfg._fbound_dtype)
            self.lib.eval_patchwise(
                Xb.ctypes.data_as(POINTER(np.ctypeslib.as_ctypes_type(cfg._fbound_dtype))),
                c_int(end - start),
                self.p_clause_position_bounds,
                self.p_clause_feat_bounds,
                self.p_clause_feat_ids,
                self.p_clause_n_feats,
                self.p_has_contra,
                self.p_clause_len,
                out[start:end].ctypes.data_as(int8_p),
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

        for start, end in self._batches(N, batch_size, "WAC"):
            patch_output = self._patch_outputs(X[start:end], -1, force_repack and start == 0, desc="WAC activations")
            p_patch_output = patch_output.ctypes.data_as(int8_p)
            out_b = output[start:end]
            p_out_b = out_b.ctypes.data_as(float_p)

            for e in range(end - start):
                self.lib.wac_sample(
                    c_int(int(target_classes[start + e])),
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
