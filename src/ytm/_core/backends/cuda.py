try:
    import cupy as cp
except ImportError as e:
    raise ImportError("`device='cuda'` requires `cupy` to be installed. But `cupy` is not installed.") from e

import pathlib

import numpy as np

from ..utils import read_file
from .base import BaseDevice


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


class CUDADevice(BaseDevice):
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

    def _code_sections(self) -> dict[str, str]:
        """The sources to concatenate, in order. Subclasses can add, replace or drop entries."""
        core = pathlib.Path(__file__).parent
        names = ("cuda.h", "common.h", "rng.h", "feedback.h", "pack_clauses.h", "pack_clauses.cu", "evaluate.cu", "interpret.cu")
        return {name: read_file(core / name) for name in names}

    def _bind(self):
        pass

    def _to_host(self, arr) -> np.ndarray:
        with self.cuda_dev:
            return arr.get()

    def _to_dev(self, arr: np.ndarray) -> cp.ndarray:
        with self.cuda_dev:
            return cp.asarray(arr)

    def _init_consts(self):
        with self.cuda_dev:
            super()._init_consts()

    def _init_params(self):
        with self.cuda_dev:
            super()._init_params()

    def _restore_params(self, state: dict):
        with self.cuda_dev:
            super()._restore_params(state)

    def load_state_dict(self, state: dict) -> None:
        with self.cuda_dev:
            super().load_state_dict(state)

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
    def _fit_eval(self, buf, e: int, rng_key):
        cfg = self.config
        pc = self.packed_clauses

        self.cu.evaluate(
            cfg._total_clauses * self._warp_size,
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
        self.cu.sum_votes(
            self.config.n_classes * self._warp_size,
            (buf.clause_output, self.clause_weights, buf.votes, np.int32(1)),
        )

    def calc_class_sums(self, X: np.ndarray, batch_size: int = -1, force_repack: bool = False) -> np.ndarray:
        cfg = self.config
        N = X.shape[0]
        pc = self.packed_clauses

        with self.cuda_dev:
            self.pack_clauses(force_repack)
            class_sums = cp.zeros((N, cfg.n_classes), dtype=np.float32)

            for i, end, bs in self._batches(N, batch_size, "Infer"):
                Xb = cp.asarray(X[i:end], dtype=cfg._fbound_dtype)
                clause_outputs = cp.empty((bs, cfg._total_clauses), dtype=np.int8)

                self.cu.calc_clause_outputs(
                    bs * cfg._total_clauses * self._warp_size,
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

                self.cu.sum_votes(
                    bs * cfg.n_classes * self._warp_size,
                    (clause_outputs, self.clause_weights, class_sums[i:end], np.int32(bs)),
                )

            return class_sums.get()

    def transform(self, X: np.ndarray, batch_size: int = -1, force_repack: bool = False) -> np.ndarray:
        cfg = self.config
        N = X.shape[0]
        pc = self.packed_clauses
        out = np.empty((N, cfg._total_clauses), dtype=np.int8)

        with self.cuda_dev:
            self.pack_clauses(force_repack)

            for i, end, bs in self._batches(N, batch_size, "Transform"):
                Xb = cp.asarray(X[i:end], dtype=cfg._fbound_dtype)
                clause_outputs = cp.empty((bs, cfg._total_clauses), dtype=np.int8)

                self.cu.calc_clause_outputs(
                    bs * cfg._total_clauses * self._warp_size,
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

    def _patch_outputs(self, X: np.ndarray, batch_size: int = -1, force_repack: bool = False, desc: str = "Transform") -> np.ndarray:
        cfg = self.config
        N = X.shape[0]
        pc = self.packed_clauses
        out = np.empty((N, cfg._total_clauses, cfg._n_patches), dtype=np.int8)

        with self.cuda_dev:
            self.pack_clauses(force_repack)

            for i, end, bs in self._batches(N, batch_size, desc):
                Xb = cp.asarray(X[i:end], dtype=cfg._fbound_dtype)
                patch_output = cp.empty((bs, cfg._total_clauses, cfg._n_patches), dtype=np.int8)

                self.cu.calc_clause_outputs_patchwise(
                    bs * cfg._total_clauses * cfg._n_patches,
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

        for i, end, bs in self._batches(N, batch_size, "WAC"):
            patch_output = self._patch_outputs(X[i:end], -1, force_repack and i == 0, desc="WAC activations")

            with self.cuda_dev:
                patch_output_gpu = cp.asarray(patch_output)
                target_classes_gpu = cp.asarray(target_classes[i:end], dtype=np.int32)
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
                output[i:end] = out_b.get()
        return output
