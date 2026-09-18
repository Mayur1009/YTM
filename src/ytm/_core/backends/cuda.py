import pathlib

import cupy as cp
import numpy as np

from ..utils import tqdm_bar
from .base import BaseDevice


def read_file(path: pathlib.Path) -> str:
    with open(path) as f:
        return f.read()


class CUDADevice(BaseDevice):
    cuda_dev: cp.cuda.Device
    cu_mod: cp.RawModule

    def dev_init(self):
        self.xp = cp
        self.cuda_dev = cp.cuda.Device(self.device_config._gpu_id)
        self.cuda_dev.use()

        with self.cuda_dev:
            self._init_clauses()
            self._init_weights()
            self._init_patch_weights()
            self._init_packed_clauses()
            self._init_device_arrays()
            self._init_kernels()

    def _code_sections(self) -> dict[str, str]:
        """The sources to concatenate, in order. Subclasses can add, replace or drop entries."""
        core = pathlib.Path(__file__).parent
        names = ("cuda.h", "common.h", "rng.h", "feedback.h", "feedback.cu", "pack_clauses.h", "pack_clauses.cu", "evaluate.cu", "interpret.cu")
        return {name: read_file(core / name) for name in names}

    def _init_kernels(self):
        with self.cuda_dev:
            self.cu_mod = cp.RawModule(code=self._build_code(), backend="nvrtc", options=())

            self.cu_pack_clauses = self.cu_mod.get_function("pack_clauses")
            self.cu_calc_clause_outputs = self.cu_mod.get_function("calc_clause_outputs")
            self.cu_calc_clause_outputs_patchwise = self.cu_mod.get_function("calc_clause_outputs_patchwise")
            self.cu_sum_votes = self.cu_mod.get_function("sum_votes")
            self.cu_evaluate = self.cu_mod.get_function("evaluate")
            self.cu_wic = self.cu_mod.get_function("wic")
            self.cu_wac = self.cu_mod.get_function("wac")

    def _init_device_arrays(self):
        cfg = self.config
        with self.cuda_dev:
            self.therm_bits_gpu = cp.asarray(cfg._therm_bits, dtype=cfg._fbound_dtype)
            self.literal_offsets_gpu = cp.asarray(cfg._literal_offsets, dtype=cfg._nlits_dtype)

    def _kernel_config(self, n: int) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        dev = self.device_config
        bs = dev._block_size
        gs = dev._grid_size if dev._grid_size is not None else min((n + bs - 1) // bs, dev._max_grid_size)
        return (gs, 1, 1), (bs, 1, 1)

    def _to_host(self, arr) -> np.ndarray:
        with self.cuda_dev:
            return arr.get()

    def pack_clauses(self, force_repack: bool = False, full: bool = False):
        with self.cuda_dev:
            if force_repack:
                self.packed_clauses.is_clause_synced.fill(0)

            self.cu_pack_clauses(
                *self._kernel_config(self.config._total_clauses * self.device_config._cuda_props["warp_size"]),
                (
                    self.ta_states,
                    self.therm_bits_gpu,
                    self.literal_offsets_gpu,
                    self.packed_clauses.clause_position_bounds,
                    self.packed_clauses.clause_feat_bounds,
                    self.packed_clauses.clause_feat_ids,
                    self.packed_clauses.clause_n_feats,
                    self.packed_clauses.has_contra,
                    self.packed_clauses.clause_len,
                    self.packed_clauses.is_clause_synced,
                    np.int32(full),
                ),
            )

    # == fit steps ==
    def _fit_eval(self, buf, e: int, rng_key):
        cfg = self.config
        pc = self.packed_clauses

        self.cu_evaluate(
            *self._kernel_config(cfg._total_clauses * self.device_config._cuda_props["warp_size"]),
            (
                np.uint64(rng_key),
                buf.X,
                np.int32(e),
                buf.clause_drop_mask,
                pc.clause_position_bounds,
                pc.clause_feat_bounds,
                pc.clause_feat_ids,
                pc.clause_n_feats,
                pc.has_contra,
                pc.clause_len,
                buf.clause_output,
                buf.selected_pids,
                self.patch_weights,
            ),
        )

    def _fit_voting(self, buf):
        self.cu_sum_votes(
            *self._kernel_config(self.config.n_classes * self.device_config._cuda_props["warp_size"]),
            (buf.clause_output, self.clause_weights, buf.votes, np.int32(1)),
        )

    def _batches(self, N: int, batch_size: int, desc: str):
        if batch_size == -1:
            batch_size = N
        for i in tqdm_bar(range(0, N, batch_size), desc=desc):
            end = min(i + batch_size, N)
            yield i, end, end - i

    def calc_class_sums(self, X: np.ndarray, batch_size: int = -1, force_repack: bool = False) -> np.ndarray:
        cfg = self.config
        N = X.shape[0]
        warp_size = self.device_config._cuda_props["warp_size"]
        pc = self.packed_clauses

        with self.cuda_dev:
            self.pack_clauses(force_repack)
            class_sums = cp.zeros((N, cfg.n_classes), dtype=np.float32)

            for i, end, bs in self._batches(N, batch_size, "Infer"):
                Xb = cp.asarray(X[i:end], dtype=cfg._fbound_dtype)
                clause_outputs = cp.empty((bs, cfg._total_clauses), dtype=np.int8)

                self.cu_calc_clause_outputs(
                    *self._kernel_config(bs * cfg._total_clauses * warp_size),
                    (
                        Xb,
                        clause_outputs,
                        np.int32(bs),
                        pc.clause_position_bounds,
                        pc.clause_feat_bounds,
                        pc.clause_feat_ids,
                        pc.clause_n_feats,
                        pc.has_contra,
                        pc.clause_len,
                    ),
                )

                self.cu_sum_votes(
                    *self._kernel_config(bs * cfg.n_classes * warp_size),
                    (clause_outputs, self.clause_weights, class_sums[i:end], np.int32(bs)),
                )

            return class_sums.get()

    def transform(self, X: np.ndarray, batch_size: int, force_repack: bool = False) -> np.ndarray:
        cfg = self.config
        N = X.shape[0]
        warp_size = self.device_config._cuda_props["warp_size"]
        pc = self.packed_clauses
        out = np.empty((N, cfg._total_clauses), dtype=np.int8)

        with self.cuda_dev:
            self.pack_clauses(force_repack)

            for i, end, bs in self._batches(N, batch_size, "Transform"):
                Xb = cp.asarray(X[i:end], dtype=cfg._fbound_dtype)
                clause_outputs = cp.empty((bs, cfg._total_clauses), dtype=np.int8)

                self.cu_calc_clause_outputs(
                    *self._kernel_config(bs * cfg._total_clauses * warp_size),
                    (
                        Xb,
                        clause_outputs,
                        np.int32(bs),
                        pc.clause_position_bounds,
                        pc.clause_feat_bounds,
                        pc.clause_feat_ids,
                        pc.clause_n_feats,
                        pc.has_contra,
                        pc.clause_len,
                    ),
                )
                out[i:end] = clause_outputs.get()

        return out.reshape(N, cfg._n_clause_banks, cfg._n_clauses)

    def _patch_outputs(self, X: np.ndarray, batch_size: int, force_repack: bool = False, desc: str = "Transform") -> np.ndarray:
        cfg = self.config
        N = X.shape[0]
        pc = self.packed_clauses
        out = np.empty((N, cfg._total_clauses, cfg._n_patches), dtype=np.int8)

        with self.cuda_dev:
            self.pack_clauses(force_repack)

            for i, end, bs in self._batches(N, batch_size, desc):
                Xb = cp.asarray(X[i:end], dtype=cfg._fbound_dtype)
                patch_output = cp.empty((bs, cfg._total_clauses, cfg._n_patches), dtype=np.int8)

                self.cu_calc_clause_outputs_patchwise(
                    *self._kernel_config(bs * cfg._total_clauses * cfg._n_patches),
                    (
                        Xb,
                        patch_output,
                        np.int32(bs),
                        pc.clause_position_bounds,
                        pc.clause_feat_bounds,
                        pc.clause_feat_ids,
                        pc.clause_n_feats,
                        pc.has_contra,
                        pc.clause_len,
                    ),
                )
                out[i:end] = patch_output.get()

        return out

    def transform_patchwise(self, X: np.ndarray, batch_size: int, force_repack: bool = False) -> np.ndarray:
        cfg = self.config
        out = self._patch_outputs(X, batch_size, force_repack)
        return out.reshape(out.shape[0], cfg._n_clause_banks, cfg._n_clauses, cfg._n_patches_y, cfg._n_patches_x)

    def wic(self, class_id: int, polarity: int, pw_th: float = 0.0, force_repack: bool = False) -> np.ndarray:
        cfg = self.config
        pc = self.packed_clauses

        with self.cuda_dev:
            self.pack_clauses(force_repack)

            pw = self.patch_weights.astype(np.float32)
            pw_norm = cp.ascontiguousarray(pw / (pw.max(axis=-1, keepdims=True) + 1e-7))
            output = cp.zeros(cfg._dim, dtype=np.float32)

            self.cu_wic(
                *self._kernel_config(cfg._total_clauses * cfg._n_patches),
                (
                    np.int32(class_id),
                    np.int32(polarity),
                    self.clause_weights,
                    pc.clause_feat_bounds,
                    pc.clause_feat_ids,
                    pc.clause_n_feats,
                    pc.clause_position_bounds,
                    pc.has_contra,
                    pw_norm,
                    self.therm_bits_gpu,
                    np.float32(pw_th),
                    output,
                ),
            )
            return output.get()

    def wac(self, X: np.ndarray, target_classes: np.ndarray, polarity: int, force_repack: bool = False) -> np.ndarray:
        cfg = self.config
        pc = self.packed_clauses
        N = X.shape[0]

        patch_output = self._patch_outputs(X, -1, force_repack, desc="WAC activations")

        with self.cuda_dev:
            patch_output_gpu = cp.asarray(patch_output)
            target_classes_gpu = cp.asarray(target_classes, dtype=np.int32)
            output = cp.zeros((N, *cfg._dim), dtype=np.float32)

            self.cu_wac(
                *self._kernel_config(N * cfg._total_clauses * cfg._n_patches),
                (
                    target_classes_gpu,
                    np.int32(polarity),
                    np.int32(N),
                    self.clause_weights,
                    pc.clause_feat_bounds,
                    pc.clause_feat_ids,
                    pc.clause_n_feats,
                    patch_output_gpu,
                    pc.has_contra,
                    self.therm_bits_gpu,
                    output,
                ),
            )
            return output.get()
