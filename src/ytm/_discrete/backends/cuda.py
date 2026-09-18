import pathlib
from dataclasses import dataclass
from typing import Any

import cupy as cp
import numpy as np

from ..._core.backends.cuda import CUDADevice as CoreCUDADevice
from ..._core.backends.cuda import read_file
from ..._core.utils import FitBuffers, tqdm_bar
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

    def _kernel_names(self) -> tuple[str, ...]:
        return super()._kernel_names() + ("calc_update_prob", "decide_feedback", "update_clauses", "update_weights")

    def _fit_allocs(self, X: np.ndarray, Y: np.ndarray, clause_drop_mask, label_probs: np.ndarray) -> DiscreteFitBuffers:
        cfg = self.config
        return DiscreteFitBuffers(
            X=cp.asarray(X, dtype=np.int32),
            Y=cp.asarray(Y, dtype=np.float32),
            clause_drop_mask=clause_drop_mask,
            selected_pids=cp.empty(cfg._total_clauses, dtype=np.int32),
            votes=cp.empty(cfg.n_classes, dtype=np.float32),
            feedback_type=cp.zeros((cfg._n_clauses, cfg.n_classes), dtype=np.uint8),
            prob=cp.zeros(cfg.n_classes, dtype=np.float32),
            label_probs=cp.asarray(label_probs, dtype=np.float32),
        )

    def fit_epoch(self, X: np.ndarray, Y: np.ndarray, clause_drop_p: float, batch_size: int, label_probs: np.ndarray) -> None:
        with self.cuda_dev:
            buf = self._fit_allocs(X, Y, self._fit_drop_mask(clause_drop_p), label_probs)

            for e, rng_key in self._fit_samples(tqdm_bar(range(X.shape[0]), desc="Fit")):
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

        self.k_calc_update_prob(
            *self._kernel_config(cfg.n_classes),
            (buf.votes, buf.Y, np.int32(e), buf.prob),
        )

        self.k_decide_feedback(
            *self._kernel_config(cfg._total_clauses),
            (
                np.uint64(rng_key),
                buf.selected_pids,
                self.packed_clauses.clause_density,
                buf.clause_drop_mask,
                buf.prob,
                buf.label_probs,
                np.int32(e),
                self.clause_weights,
                buf.feedback_type,
            ),
        )

    def _fit_apply_fb(self, buf: DiscreteFitBuffers, e: int, rng_key: int) -> None:
        self.k_update_clauses(
            *self._kernel_config(self.config._total_clauses * self.device_config._cuda_props["warp_size"]),
            (
                np.uint64(rng_key),
                buf.selected_pids,
                buf.X,
                np.int32(e),
                self.feat_mins_gpu,
                self.literal_offsets_gpu,
                buf.feedback_type,
                self.ta_states,
                self.packed_clauses.is_clause_synced,
            ),
        )

    def _fit_update_weights(self, buf: DiscreteFitBuffers) -> None:
        self.k_update_weights(
            *self._kernel_config(self.config._total_clauses),
            (buf.feedback_type, self.clause_weights),
        )
