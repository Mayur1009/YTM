import pathlib
from ctypes import c_int, c_uint64

import numpy as np

from ..._core.backends.cpu import CPUDevice as CoreCPUDevice
from ..._core.backends.cpu import bind_ptrs
from ..._core.utils import read_file, tqdm_bar
from .buffers import DiscreteFitBuffers


class CPUDevice(CoreCPUDevice):
    def _code_sections(self) -> dict[str, str]:
        sections = super()._code_sections()
        sections["update.c"] = read_file(pathlib.Path(__file__).parent / "update.c")
        return sections

    def _fit_allocs(self, X: np.ndarray, Y: np.ndarray, clause_drop_mask: np.ndarray, label_probs: np.ndarray) -> DiscreteFitBuffers:
        cfg = self.config
        for name, arr, dtype in (("X", X, cfg._fbound_dtype), ("Y", Y, np.float32), ("label_probs", label_probs, np.float32)):
            assert arr.dtype == dtype and arr.flags.c_contiguous, (
                f"`{name}` must be C contiguous {np.dtype(dtype)}, got {arr.dtype} contiguous={arr.flags.c_contiguous}"
            )

        buf = DiscreteFitBuffers(
            X=X,
            Y=Y,
            clause_drop_mask=clause_drop_mask,
            clause_output=np.empty(cfg._total_clauses, dtype=np.int8),
            selected_pids=np.empty(cfg._total_clauses if cfg._n_patches > 1 else 1, dtype=cfg._npatches_dtype),
            votes=np.empty(cfg.n_classes, dtype=np.float32),
            feedback_type=np.zeros((cfg._n_clauses, cfg.n_classes), dtype=np.uint8),
            prob=np.zeros(cfg.n_classes, dtype=np.float32),
            label_probs=label_probs,
            **self._fb_list_allocs(),
        )
        return bind_ptrs(buf)

    def fit_epoch(self, X: np.ndarray, Y: np.ndarray, clause_drop_p: float, batch_size: int, label_probs: np.ndarray) -> None:
        buf = self._fit_allocs(X, Y, self._fit_drop_mask(clause_drop_p), label_probs)

        for e, rng_key in self._fit_iter(tqdm_bar(range(X.shape[0]), desc="Fit")):
            self.fit_sample(rng_key, buf, e)

    def _fit_decide_fb(self, buf: DiscreteFitBuffers, e: int, rng_key: int) -> None:
        """Turn the votes into a per class update probability, then pick a feedback type per clause."""
        buf.t1b_n[0] = 0
        buf.t1a_t2_n[0] = 0
        self.lib.calc_update_prob(buf.p_votes, buf.p_Y, c_int(e), buf.p_prob)
        self.lib.decide_feedback(
            c_uint64(rng_key),
            buf.p_clause_output,
            self.p_clause_len,
            buf.p_clause_drop_mask,
            buf.p_prob,
            buf.p_label_probs,
            c_int(e),
            self.p_clause_weights,
            buf.p_feedback_type,
            buf.p_t1b_n,
            buf.p_t1b_cid,
            buf.p_t1b_cnt,
            buf.p_t1a_t2_n,
            buf.p_t1a_t2_cid,
            buf.p_t1a_cnt,
            buf.p_t2_cnt,
        )

    def _fit_update_weights(self, buf: DiscreteFitBuffers) -> None:
        self.lib.update_weights(buf.p_feedback_type, buf.p_t1a_t2_n, buf.p_t1a_t2_cid, self.p_clause_weights)
