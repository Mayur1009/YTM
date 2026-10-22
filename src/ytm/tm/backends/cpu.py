import pathlib
from ctypes import c_int, c_uint64

from ..._core.backends.cpu import CPUDevice as CoreCPUDevice
from ..._core.utils import read_file
from ..config import TMConfig
from .buffers import DiscreteFitBuffers


class CPUDevice(CoreCPUDevice[DiscreteFitBuffers, TMConfig]):
    def _code_sections(self) -> dict[str, str]:
        sections = super()._code_sections()
        sections["update.c"] = read_file(pathlib.Path(__file__).parent / "update.c")
        return sections

    @staticmethod
    def _fit_buffers_cls() -> type[DiscreteFitBuffers]:
        return DiscreteFitBuffers

    def _fit_eval(self, buf: DiscreteFitBuffers, e: int, rng_key: int) -> None:
        if not self.config.coalesced:
            self.lib.sample_classes(c_uint64(rng_key), buf.p_label_probs, c_int(e), buf.p_class_active)
        super()._fit_eval(buf, e, rng_key)

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

    def _fit_update_weights(self, buf: DiscreteFitBuffers) -> None:
        self.lib.update_weights(buf.p_feedback_type, buf.p_t1a_t2_n, buf.p_t1a_t2_cid, self.p_clause_weights)
