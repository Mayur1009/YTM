import pathlib
from dataclasses import dataclass
from typing import Any

import cupy as cp
import numpy as np

from ..._core.backends.cuda import CUDADevice as CoreCUDADevice
from ..._core.utils import FitBuffers, read_file, tqdm_bar
from .base import BaseDevice as DiscreteBaseDevice


@dataclass(kw_only=True)
class DiscreteFitBuffers(FitBuffers):
    feedback_type: Any  # (n_clauses, n_classes) one of FB_NONE/T1A/T1B/T2, indexed [rel_clause, class]
    prob: Any  # (n_classes,) signed update probability for the current sample
    label_probs: Any  # (n_samples, n_classes) chance a class gives feedback at all


class CUDADevice(DiscreteBaseDevice, CoreCUDADevice):
    def _code_sections(self) -> dict[str, str]:
        sections = super()._code_sections()
        sections["update.cu"] = read_file(pathlib.Path(__file__).parent / "update.cu")
        return sections

    def _init_kernels(self):
        super()._init_kernels()

        with self.cuda_dev:
            self.cu_calc_update_prob = self.cu_mod.get_function("calc_update_prob")
            self.cu_decide_feedback = self.cu_mod.get_function("decide_feedback")
            self.cu_update_clauses = self.cu_mod.get_function("update_clauses")
            self.cu_update_weights = self.cu_mod.get_function("update_weights")

    def _fit_allocs(self, clause_drop_mask) -> DiscreteFitBuffers:
        cfg = self.config
        return DiscreteFitBuffers(
            X=None,
            Y=None,
            clause_drop_mask=clause_drop_mask,
            clause_output=cp.empty(cfg._total_clauses, dtype=np.int8),
            selected_pids=cp.empty(cfg._total_clauses if cfg._n_patches > 1 else 1, dtype=cfg._npatches_dtype),
            votes=cp.empty(cfg.n_classes, dtype=np.float32),
            feedback_type=cp.zeros((cfg._n_clauses, cfg.n_classes), dtype=np.uint8),
            prob=cp.zeros(cfg.n_classes, dtype=np.float32),
            label_probs=None,
        )

    def fit_epoch(self, X: np.ndarray, Y: np.ndarray, clause_drop_p: float, batch_size: int, label_probs: np.ndarray) -> None:
        cfg = self.config
        N = X.shape[0]
        bs = N if batch_size == -1 else batch_size

        with self.cuda_dev:
            buf = self._fit_allocs(self._fit_drop_mask(clause_drop_p))
            samples = self._fit_samples(tqdm_bar(range(N), desc="Fit"))

            for i in range(0, N, bs):
                end = min(i + bs, N)
                buf.X = cp.asarray(X[i:end], dtype=cfg._fbound_dtype)
                buf.Y = cp.asarray(Y[i:end], dtype=np.float32)
                buf.label_probs = cp.asarray(label_probs[i:end], dtype=np.float32)

                for e in range(end - i):
                    _, rng_key = next(samples)
                    self.fit_sample(rng_key, buf, e)

    def fit_sample(self, rng_key: int, buf: DiscreteFitBuffers, e: int) -> None:
        with self.cuda_dev:
            self.pack_clauses()
            self._fit_eval(buf, e, rng_key)
            self._fit_voting(buf)
            self._fit_decide_fb(buf, e, rng_key)
            self._fit_apply_fb(buf, e, rng_key)
            self._fit_update_weights(buf)

    def _fit_decide_fb(self, buf: DiscreteFitBuffers, e: int, rng_key: int) -> None:
        cfg = self.config

        self.cu_calc_update_prob(
            *self._kernel_config(cfg.n_classes),
            (buf.votes, buf.Y, np.int32(e), buf.prob),
        )

        self.cu_decide_feedback(
            *self._kernel_config(cfg._total_clauses),
            (
                np.uint64(rng_key),
                buf.clause_output,
                self.packed_clauses.clause_len,
                buf.clause_drop_mask,
                buf.prob,
                buf.label_probs,
                np.int32(e),
                self.clause_weights,
                buf.feedback_type,
            ),
        )

    def _fit_apply_fb(self, buf: DiscreteFitBuffers, e: int, rng_key: int) -> None:
        self.cu_update_clauses(
            *self._kernel_config(self.config._total_clauses * self.device_config._cuda_props["warp_size"]),
            (
                np.uint64(rng_key),
                buf.clause_output,
                buf.selected_pids,
                buf.X,
                np.int32(e),
                self.literal_offsets_gpu,
                buf.feedback_type,
                self.ta_states,
                self.packed_clauses.is_clause_synced,
            ),
        )

    def _fit_update_weights(self, buf: DiscreteFitBuffers) -> None:
        self.cu_update_weights(
            *self._kernel_config(self.config._total_clauses),
            (buf.feedback_type, self.clause_weights),
        )
