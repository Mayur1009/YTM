import pathlib

import cupy as cp
import numpy as np

from ..._core.backends.cuda import CUDADevice as CoreCUDADevice
from ..._core.utils import read_file
from ..config import FbSignal, TMConfig
from .buffers import GuidedFitBuffers


class CUDADevice(CoreCUDADevice[GuidedFitBuffers, TMConfig]):
    def _code_sections(self) -> dict[str, str]:
        sections = super()._code_sections()
        here = pathlib.Path(__file__).parent
        sections["act.h"] = read_file(here / "act.h")
        sections["loss"] = self.config.act_loss.src
        sections["update.cu"] = read_file(here / "update.cu")
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
        buf.t1b_n.data.memset_async(0, buf.t1b_n.nbytes)
        buf.t1a_t2_n.data.memset_async(0, buf.t1a_t2_n.nbytes)

        self.cu.votes_activation(1, (buf.votes, buf.y_hat))
        self.cu.loss_gradient(1, (buf.y_hat, buf.Y[e], buf.grad, buf.loss))

        if cfg._fb_signal == FbSignal.GRAD:
            self.cu.decide_feedback_grad(
                cfg._total_clauses,
                (
                    buf.rng_keys,
                    np.int32(e),
                    buf.grad,
                    self.clause_weights,
                    self.packed_clauses.clause_len,
                    buf.clause_output,
                    buf.clause_drop_mask,
                    np.float32(buf.lambda_),
                    buf.t1b_n,
                    buf.t1b_cid,
                    buf.t1b_cnt,
                    buf.t1a_t2_n,
                    buf.t1a_t2_cid,
                    buf.t1a_cnt,
                    buf.t2_cnt,
                ),
            )
        else:
            self.cu.compute_votes_neg_ck(
                cfg._total_clauses * cfg.n_classes,
                (buf.votes, self.clause_weights, buf.clause_output, buf.votes_neg_ck),
            )
            self.cu.votes_activation_batch(
                cfg._total_clauses,
                (buf.votes_neg_ck, np.int32(cfg._total_clauses), buf.y_hat_neg_ck),
            )
            self.cu.compute_loss_neg_ck(cfg._total_clauses, (buf.y_hat_neg_ck, buf.Y[e], buf.loss_neg_ck))
            self.cu.decide_feedback_delta_l(
                cfg._total_clauses,
                (
                    buf.rng_keys,
                    np.int32(e),
                    buf.loss,
                    buf.loss_neg_ck,
                    self.packed_clauses.clause_len,
                    buf.clause_output,
                    buf.clause_drop_mask,
                    np.float32(buf.lambda_),
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
            (buf.grad, np.float32(buf.lr), buf.clause_output, buf.clause_drop_mask, self.clause_weights),
        )

    def _infer_batch(self, Xb: cp.ndarray, bs: int) -> cp.ndarray:
        votes = super()._infer_batch(Xb, bs)
        y_hat = cp.empty_like(votes)
        self.cu.votes_activation_batch(bs, (votes, np.int32(bs), y_hat))
        return y_hat
