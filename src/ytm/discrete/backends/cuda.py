import pathlib
from dataclasses import dataclass
from typing import Any

import cupy as cp
import numpy as np

from ..._core.backends.base import FitBuffers
from ..._core.backends.cuda import CUDADevice as CoreCUDADevice
from ..._core.utils import read_file, tqdm_bar
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
            **self._fb_list_allocs(),
        )

    def fit_epoch(self, X: np.ndarray, Y: np.ndarray, clause_drop_p: float, batch_size: int, label_probs: np.ndarray) -> None:
        cfg = self.config
        N = X.shape[0]
        bs = N if batch_size == -1 else batch_size

        with self.cuda_dev:
            # The fit stream does not wait on the default stream, so finish earlier work first.
            self.cuda_dev.synchronize()
            with self._fit_stream:
                buf = self._fit_allocs(self._fit_drop_mask(clause_drop_p))
                keys = self._fit_keys(N)
                pbar = tqdm_bar(None, total=N, desc="Fit")

                for i in range(0, N, bs):
                    end = min(i + bs, N)
                    buf.X = cp.asarray(X[i:end], dtype=cfg._fbound_dtype)
                    buf.Y = cp.asarray(Y[i:end], dtype=np.float32)
                    buf.label_probs = cp.asarray(label_probs[i:end], dtype=np.float32)

                    self._run_batch(buf, keys[i:end], lambda: self._fit_sample_kernels(buf), pbar)
                pbar.close()
            self._fit_stream.synchronize()

    def fit_sample(self, rng_key: int, buf: DiscreteFitBuffers, e: int) -> None:
        with self.cuda_dev:
            self._set_sample(buf, e, rng_key)
            self._fit_sample_kernels(buf)

    def _fit_sample_kernels(self, buf: DiscreteFitBuffers) -> None:
        """One sample's kernels. They read the sample index and rng key on the device, so the stage arguments are unused."""
        self.pack_clauses()
        self._fit_eval(buf, 0, 0)
        self._fit_voting(buf)
        self._fit_decide_fb(buf, 0, 0)
        self._fit_apply_fb(buf, 0, 0)
        if self.config.weighted:
            self._fit_update_weights(buf)

    def _fit_decide_fb(self, buf: DiscreteFitBuffers, e: int, rng_key: int) -> None:
        cfg = self.config
        self.cu.calc_update_prob(
            cfg.n_classes,
            (buf.votes, buf.Y, buf.e_dev, buf.prob),
        )

        self.cu.decide_feedback(
            cfg._total_clauses,
            (
                buf.keys,
                buf.e_dev,
                buf.clause_output,
                self.packed_clauses.clause_len,
                buf.clause_drop_mask,
                buf.prob,
                buf.label_probs,
                self.clause_weights,
                buf.feedback_type,
                buf.t1b_n,
                buf.t1b_cid,
                buf.t1b_cnt,
                buf.t1a_t2_n,
                buf.t1a_t2_cid,
                buf.t1a_cnt,
                buf.t2_cnt,
            ),
        )

    def _fit_update_weights(self, buf: DiscreteFitBuffers) -> None:
        self.cu.update_weights(
            self.config._total_clauses,
            (buf.feedback_type, buf.t1a_t2_n, buf.t1a_t2_cid, self.clause_weights),
        )
