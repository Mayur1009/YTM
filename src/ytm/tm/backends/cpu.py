import pathlib
from ctypes import c_int

from ..._core.backends.cpu import CPUDevice as CoreCPUDevice
from ..config import TMConfig
from .buffers import TMFitBuffers


class CPUDevice(CoreCPUDevice[TMFitBuffers, TMConfig]):
    def _code_sections(self) -> dict[str, str]:
        sections = super()._code_sections()
        sections["update.c"] = (pathlib.Path(__file__).parent / "update.c").read_text()
        return sections

    @staticmethod
    def _fit_buffers_cls() -> type[TMFitBuffers]:
        return TMFitBuffers

    def _fit_eval(self, e: int) -> None:
        buf = self.buf
        if not self.config.coalesced:
            self.lib.sample_classes(buf.p_rng_keys, buf.p_label_probs, c_int(e), buf.p_class_active)
        super()._fit_eval(e)

    def _fit_decide_fb(self, e: int) -> None:
        """Turn the votes into a per class update probability, then pick a feedback type per clause."""
        buf = self.buf
        buf.t1b_n[0] = 0
        buf.t1a_t2_n[0] = 0
        self.lib.calc_update_prob(buf.p_votes, buf.p_Y, c_int(e), buf.p_prob)
        self.lib.decide_feedback(
            buf.p_rng_keys,
            buf.p_clause_output,
            self.p_clause_len,
            buf.p_clause_drop_mask,
            buf.p_prob,
            buf.p_label_probs,
            buf.p_class_active,
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

    def _fit_update_weights(self) -> None:
        buf = self.buf
        self.lib.update_weights(buf.p_feedback_type, buf.p_t1a_t2_n, buf.p_t1a_t2_cid, self.p_clause_weights)
