try:
    import cupy as cp
except ImportError as e:
    raise ImportError("`device='cuda'` requires `cupy` to be installed. But `cupy` is not installed.") from e

import pathlib

import numpy as np

from ..utils import read_file
from .base import BaseDevice, BuffersT, ConfigT


class Launcher:
    def __init__(self, fn: cp.RawKernel, dev: cp.cuda.Device, block_size: int, n_sms: int):
        self.fn = fn
        self.block_size = block_size
        with dev:
            per_sm = cp.cuda.driver.occupancyMaxActiveBlocksPerMultiprocessor(fn.kernel.ptr, block_size, 0)
        self.max_grid_size = max(1, n_sms * per_sm)

    def __call__(self, n: int, args: tuple):
        grid_size = min(max(1, (n + self.block_size - 1) // self.block_size), self.max_grid_size)
        self.fn((grid_size, 1, 1), (self.block_size, 1, 1), args)


class Kernel:
    def __init__(self, code: str, dev: cp.cuda.Device, block_size: int, n_sms: int):
        self._dev = dev
        self._block_size = block_size
        self._n_sms = n_sms
        with dev:
            self._mod = cp.RawModule(code=code)
            self._mod.compile()

    def __getattr__(self, name: str) -> Launcher:
        if name.startswith("_"):
            raise AttributeError(name)
        with self._dev:
            try:
                fn = self._mod.get_function(name)
            except cp.cuda.driver.CUDADriverError as e:
                raise AttributeError(name) from e
        launcher = Launcher(fn, self._dev, self._block_size, self._n_sms)
        setattr(self, name, launcher)
        return launcher


class CUDADevice(BaseDevice[BuffersT, ConfigT]):
    _VOTING_MIN_CHUNK_SIZE = 256  # smallest clause chunk one warp sums alone

    xp = cp

    def _setup(self):
        dev = self.device_config

        self.cuda_dev = cp.cuda.Device(dev.n)
        self.cuda_dev.use()

        props = cp.cuda.runtime.getDeviceProperties(dev.n)
        self._warp_size = props["warpSize"]
        block_size = min(max(1, int(dev.block_size)), props["maxThreadsPerBlock"])
        block_size = max(self._warp_size, (block_size // self._warp_size) * self._warp_size)

        self.cu = Kernel(self._build_code(), self.cuda_dev, block_size, props["multiProcessorCount"])

        cfg = self.config
        self._fit_vote_chunks = -(-cfg._n_clauses // self._VOTING_MIN_CHUNK_SIZE)
        self._chunk_sums = cp.empty(cfg.n_classes * self._fit_vote_chunks, dtype=np.float32)

    def _code_sections(self) -> dict[str, str]:
        """The sources to concatenate, in order. Subclasses can add, replace or drop entries."""
        core = pathlib.Path(__file__).parent
        names = ("cuda.h", "common.h", "rng.h", "feedback.h", "pack_clauses.h", "pack_clauses.cu", "evaluate.cu", "interpret.cu", "feedback.cu")
        return {name: read_file(core / name) for name in names}

    def _fit_apply_fb(self, e: int) -> None:
        """T1b first, then T1a and T2, so a full clause's decrements land before its T2 increments."""
        buf = self.buf
        self.cu.update_t1b(
            self.config._total_clauses * self._warp_size,
            (
                buf.rng_keys,
                np.int32(e),
                buf.t1b_n,
                buf.t1b_cid,
                buf.t1b_cnt,
                buf.lit_drop_mask,
                self.ta_states,
                self.packed_clauses.is_clause_synced,
            ),
        )
        self.cu.update_t1a_t2(
            self.config._total_clauses * self._warp_size,
            (
                buf.rng_keys,
                np.int32(e),
                buf.selected_pids,
                buf.X,
                self.literal_offsets,
                buf.lit_drop_mask,
                buf.t1a_t2_n,
                buf.t1a_t2_cid,
                buf.t1a_cnt,
                buf.t2_cnt,
                self.ta_states,
                self.packed_clauses.is_clause_synced,
            ),
        )

    def _ctx(self):
        return self.cuda_dev

    def _bind(self):
        pass

    def _to_host(self, arr) -> np.ndarray:
        with self.cuda_dev:
            return arr.get()

    def _to_dev(self, arr: np.ndarray) -> cp.ndarray:
        with self.cuda_dev:
            return cp.asarray(arr)


    def _chunk_size(self, N, bytes_per_smp):
        with self.cuda_dev:
            cp.get_default_memory_pool().free_all_blocks()
            free = self.cuda_dev.mem_info[0]
            return min(N, max(1, free // bytes_per_smp))

    def pack_clauses(self, force_repack: bool = False, full: bool = False):
        with self.cuda_dev:
            if force_repack:
                self.packed_clauses.is_clause_synced.fill(0)

            self.cu.pack_clauses(
                self.config._total_clauses * self._warp_size,
                (
                    self.ta_states,
                    self.therm_bits,
                    self.literal_offsets,
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
    def _fit_eval(self, e: int):
        buf = self.buf
        cfg = self.config
        pc = self.packed_clauses

        self.cu.eval_fit(
            cfg._total_clauses * self._warp_size,
            (
                buf.rng_keys,
                buf.X,
                np.int32(e),
                buf.clause_drop_mask,
                buf.class_active,
                buf.lit_drop_mask,
                self.literal_offsets,
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

    def _vote_sum(self, clause_outputs, N: int, class_sums, chunks: int = 1) -> None:
        """Class sums of N samples, calculated using chunks"""
        n_sums = N * self.config.n_classes

        # If one chunk per class sum, then write the class sums directly.
        out = class_sums if chunks == 1 else self._chunk_sums
        self.cu.vote_over_clause_chunks(
            n_sums * chunks * self._warp_size,
            (clause_outputs, self.clause_weights, np.int32(N), np.int32(chunks), out),
        )
        if chunks > 1:
            self.cu.vote_combine_chunks(n_sums, (self._chunk_sums, np.int32(N), np.int32(chunks), class_sums))

    def _fit_voting(self):
        buf = self.buf
        self._vote_sum(buf.clause_output, 1, buf.votes, self._fit_vote_chunks)

    def _clause_outputs(self, Xb, bs: int):
        cfg = self.config
        pc = self.packed_clauses
        clause_outputs = cp.empty((bs, cfg._total_clauses), dtype=np.int8)
        self.cu.eval_infer(
            bs * cfg._total_clauses * self._warp_size,
            (
                Xb,
                np.int32(bs),
                pc.clause_position_bounds,
                pc.clause_feat_bounds,
                pc.clause_feat_ids,
                pc.clause_n_feats,
                pc.has_contra,
                pc.clause_len,
                clause_outputs,
            ),
        )
        return clause_outputs

    def _infer_batch(self, Xb, bs: int):
        class_sums = cp.zeros((bs, self.config.n_classes), dtype=np.float32)
        self._vote_sum(self._clause_outputs(Xb, bs), bs, class_sums)
        return class_sums

    def transform(self, X: np.ndarray, batch_size: int = -1, force_repack: bool = False) -> np.ndarray:
        cfg = self.config
        N = X.shape[0]
        out = np.empty((N, cfg._total_clauses), dtype=np.int8)

        with self.cuda_dev:
            self.pack_clauses(force_repack)

            for start, end in self._batches(N, batch_size, "Transform"):
                out[start:end] = self._clause_outputs(cp.asarray(X[start:end], dtype=cfg._fbound_dtype), end - start).get()

        return out.reshape(N, cfg._n_clause_banks, cfg._n_clauses)

    def _patch_outputs(self, X: np.ndarray, batch_size: int = -1, force_repack: bool = False, desc: str = "Transform") -> np.ndarray:
        cfg = self.config
        N = X.shape[0]
        pc = self.packed_clauses
        out = np.empty((N, cfg._total_clauses, cfg._n_patches), dtype=np.int8)

        with self.cuda_dev:
            self.pack_clauses(force_repack)

            for start, end in self._batches(N, batch_size, desc):
                bs = end - start
                Xb = cp.asarray(X[start:end], dtype=cfg._fbound_dtype)
                patch_output = cp.empty((bs, cfg._total_clauses, cfg._n_patches), dtype=np.int8)

                self.cu.eval_patchwise(
                    bs * cfg._total_clauses * cfg._n_patches,
                    (
                        Xb,
                        np.int32(bs),
                        pc.clause_position_bounds,
                        pc.clause_feat_bounds,
                        pc.clause_feat_ids,
                        pc.clause_n_feats,
                        pc.has_contra,
                        pc.clause_len,
                        patch_output,
                    ),
                )
                out[start:end] = patch_output.get()

        return out

    def wic(self, class_id: int, polarity: int, pw_th: float = 0.0, force_repack: bool = False) -> np.ndarray:
        cfg = self.config
        pc = self.packed_clauses

        with self.cuda_dev:
            self.pack_clauses(force_repack)

            pw = self.patch_weights.astype(np.float32)
            pw_norm = cp.ascontiguousarray(pw / (pw.max(axis=-1, keepdims=True) + 1e-7))
            output = cp.zeros(cfg._dim, dtype=np.float32)

            self.cu.wic(
                cfg._total_clauses * cfg._n_patches,
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
                    self.therm_bits,
                    np.float32(pw_th),
                    output,
                ),
            )
            return output.get()

    def wac(self, X: np.ndarray, target_classes: np.ndarray, polarity: int, batch_size: int = -1, force_repack: bool = False) -> np.ndarray:
        cfg = self.config
        pc = self.packed_clauses
        N = X.shape[0]
        output = np.zeros((N, *cfg._dim), dtype=np.float32)

        for start, end in self._batches(N, batch_size, "WAC"):
            bs = end - start
            patch_output = self._patch_outputs(X[start:end], -1, force_repack and start == 0, desc="WAC activations")

            with self.cuda_dev:
                patch_output_gpu = cp.asarray(patch_output)
                target_classes_gpu = cp.asarray(target_classes[start:end], dtype=np.int32)
                out_b = cp.zeros((bs, *cfg._dim), dtype=np.float32)

                self.cu.wac(
                    bs * cfg._total_clauses * cfg._n_patches,
                    (
                        target_classes_gpu,
                        np.int32(polarity),
                        np.int32(bs),
                        self.clause_weights,
                        pc.clause_feat_bounds,
                        pc.clause_feat_ids,
                        pc.clause_n_feats,
                        patch_output_gpu,
                        pc.has_contra,
                        self.therm_bits,
                        out_b,
                    ),
                )
                output[start:end] = out_b.get()
        return output
