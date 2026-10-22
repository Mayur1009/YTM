import pathlib

import cupy as cp
import numpy as np

from ..._core.backends.cuda import CUDADevice as CoreCUDADevice
from ..._core.utils import read_file, tqdm_bar
from .buffers import DiscreteFitBuffers


class CUDADevice(CoreCUDADevice):
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
            lit_drop_mask=cp.zeros(cfg._n_literals, dtype=np.uint8),
            class_active=cp.ones(cfg.n_classes, dtype=np.uint8),
            clause_output=cp.empty(cfg._total_clauses, dtype=np.int8),
            selected_pids=cp.empty(cfg._total_clauses if cfg._n_patches > 1 else 1, dtype=cfg._npatches_dtype),
            votes=cp.empty(cfg.n_classes, dtype=np.float32),
            feedback_type=cp.zeros((cfg._n_clauses, cfg.n_classes), dtype=np.uint8),
            prob=cp.zeros(cfg.n_classes, dtype=np.float32),
            label_probs=None,
            **self._fb_list_allocs(),
        )

    def fit_epoch(self, X: np.ndarray, Y: np.ndarray, clause_drop_p: float, label_probs: np.ndarray) -> None:
        cfg = self.config

        with self.cuda_dev:
            buf = self._fit_allocs(self._fit_drop_mask(clause_drop_p))
            buf.X = cp.asarray(X, dtype=cfg._fbound_dtype)
            buf.Y = cp.asarray(Y, dtype=np.float32)
            buf.label_probs = cp.asarray(label_probs, dtype=np.float32)

            for e, rng_key in self._fit_iter(tqdm_bar(range(X.shape[0]), desc="Fit")):
                self.fit_sample(rng_key, buf, e)

    def _fit_eval(self, buf: DiscreteFitBuffers, e: int, rng_key: int) -> None:
        if not self.config.coalesced:
            self.cu.sample_classes(self.config.n_classes, (np.uint64(rng_key), buf.label_probs, np.int32(e), buf.class_active))
        super()._fit_eval(buf, e, rng_key)

    def _fit_decide_fb(self, buf: DiscreteFitBuffers, e: int, rng_key: int) -> None:
        cfg = self.config
        buf.t1b_n.data.memset_async(0, buf.t1b_n.nbytes)
        buf.t1a_t2_n.data.memset_async(0, buf.t1a_t2_n.nbytes)

        self.cu.calc_update_prob(
            cfg.n_classes,
            (buf.votes, buf.Y, np.int32(e), buf.prob),
        )

        self.cu.decide_feedback(
            cfg._total_clauses,
            (
                np.uint64(rng_key),
                buf.clause_output,
                self.packed_clauses.clause_len,
                buf.clause_drop_mask,
                buf.prob,
                buf.label_probs,
                buf.class_active,
                np.int32(e),
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
