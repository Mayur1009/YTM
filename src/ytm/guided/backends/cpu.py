import pathlib
from ctypes import c_float, c_int

import numpy as np

from ..._core.backends.cpu import CPUDevice as CoreCPUDevice
from ..._core.backends.cpu import float_p
from ..config import FbSignal, TMConfig
from .buffers import GuidedFitBuffers


class CPUDevice(CoreCPUDevice[GuidedFitBuffers, TMConfig]):
    def _code_sections(self) -> dict[str, str]:
        sections = super()._code_sections()
        here = pathlib.Path(__file__).parent
        sections["act.h"] = (here / "act.h").read_text()
        sections["loss"] = self.config.act_loss.src
        sections["update.c"] = (here / "update.c").read_text()
        return sections

    @staticmethod
    def _fit_buffers_cls() -> type[GuidedFitBuffers]:
        return GuidedFitBuffers

    def post_sample(self, e: int) -> float:
        buf = self.buf
        return buf.loss[0].astype(np.float64)

    def post_epoch(self, total) -> float:
        buf = self.buf
        return float(total) / buf.N

    def _fit_decide_fb(self, e: int) -> None:
        buf = self.buf
        cfg = self.config
        buf.t1b_n[0] = 0
        buf.t1a_t2_n[0] = 0
        p_Y_e = buf.Y[e].ctypes.data_as(float_p)

        self.lib.votes_activation(buf.p_votes, buf.p_y_hat)
        self.lib.loss_gradient(buf.p_y_hat, p_Y_e, buf.p_grad, buf.p_loss)

        if cfg._fb_signal == FbSignal.GRAD:
            self.lib.decide_feedback_grad(
                buf.p_rng_keys,
                c_int(e),
                buf.p_grad,
                self.p_clause_weights,
                self.p_clause_len,
                buf.p_clause_output,
                buf.p_clause_drop_mask,
                c_float(buf.lambda_),
                buf.p_t1b_n,
                buf.p_t1b_cid,
                buf.p_t1b_cnt,
                buf.p_t1a_t2_n,
                buf.p_t1a_t2_cid,
                buf.p_t1a_cnt,
                buf.p_t2_cnt,
            )
        else:
            self.lib.compute_votes_neg_ck(buf.p_votes, self.p_clause_weights, buf.p_clause_output, buf.p_votes_neg_ck)
            self.lib.votes_activation_batch(buf.p_votes_neg_ck, c_int(cfg._total_clauses), buf.p_y_hat_neg_ck)
            self.lib.compute_loss_neg_ck(buf.p_y_hat_neg_ck, p_Y_e, buf.p_loss_neg_ck)
            self.lib.decide_feedback_delta_l(
                buf.p_rng_keys,
                c_int(e),
                buf.p_loss,
                buf.p_loss_neg_ck,
                self.p_clause_len,
                buf.p_clause_output,
                buf.p_clause_drop_mask,
                c_float(buf.lambda_),
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
        self.lib.update_weights(buf.p_grad, c_float(buf.lr), buf.p_clause_output, buf.p_clause_drop_mask, self.p_clause_weights)

    def _infer_batch(self, Xb: np.ndarray, bs: int) -> np.ndarray:
        votes = super()._infer_batch(Xb, bs)
        y_hat = np.empty_like(votes)
        self.lib.votes_activation_batch(votes.ctypes.data_as(float_p), c_int(bs), y_hat.ctypes.data_as(float_p))
        return y_hat
