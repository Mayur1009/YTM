import pathlib

import numpy as np

from ..._core.backends.cuda import CUDADevice as CoreCUDADevice
from ..config import TMConfig
from .buffers import TMFitBuffers


class CUDADevice(CoreCUDADevice[TMFitBuffers, TMConfig]):
    def _code_sections(self) -> dict[str, str]:
        sections = super()._code_sections()
        sections["update.cu"] = (pathlib.Path(__file__).parent / "update.cu").read_text()
        return sections

    @staticmethod
    def _fit_buffers_cls() -> type[TMFitBuffers]:
        return TMFitBuffers

    def _fit_eval(self, e: int) -> None:
        buf = self.buf
        if not self.config.coalesced:
            self.cu.sample_classes(self.config.n_classes, (buf.rng_keys, buf.label_probs, np.int32(e), buf.class_active))
        super()._fit_eval(e)

    def _fit_decide_fb(self, e: int) -> None:
        buf = self.buf
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
                buf.rng_keys,
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

    def _fit_update_weights(self) -> None:
        buf = self.buf
        self.cu.update_weights(
            self.config._total_clauses,
            (buf.feedback_type, buf.t1a_t2_n, buf.t1a_t2_cid, self.clause_weights),
        )
