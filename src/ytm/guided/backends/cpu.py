import pathlib
from ctypes import c_float, c_int, c_uint64

import numpy as np

from ..._core.backends.cpu import CPUDevice as CoreCPUDevice
from ..._core.backends.cpu import bind_ptrs, float_p
from ..._core.utils import read_file, tqdm_bar
from ..config import FbSignal
from .buffers import GuidedFitBuffers


class CPUDevice(CoreCPUDevice):
    def _code_sections(self) -> dict[str, str]:
        sections = super()._code_sections()
        here = pathlib.Path(__file__).parent
        sections["act.h"] = read_file(here / "act.h")
        sections["loss"] = self.config.act_loss.src
        sections["update.c"] = read_file(here / "update.c")
        return sections

    def _fit_allocs(self, X: np.ndarray, Y: np.ndarray, clause_drop_mask: np.ndarray, lr: float, lambda_: float) -> GuidedFitBuffers:
        cfg = self.config
        for name, arr, dtype in (("X", X, cfg._fbound_dtype), ("Y", Y, np.float32)):
            assert arr.dtype == dtype and arr.flags.c_contiguous, (
                f"`{name}` must be C contiguous {np.dtype(dtype)}, got {arr.dtype} contiguous={arr.flags.c_contiguous}"
            )

        extra: dict[str, np.ndarray] = {}
        if cfg._fb_signal == FbSignal.DELTA_L:
            extra["votes_neg_ck"] = np.empty((cfg._total_clauses, cfg.n_classes), dtype=np.float32)
            extra["y_hat_neg_ck"] = np.empty((cfg._total_clauses, cfg.n_classes), dtype=np.float32)
            extra["loss_neg_ck"] = np.empty(cfg._total_clauses, dtype=np.float32)

        buf = GuidedFitBuffers(
            X=X,
            Y=Y,
            clause_drop_mask=clause_drop_mask,
            clause_output=np.empty(cfg._total_clauses, dtype=np.int8),
            selected_pids=np.empty(cfg._total_clauses if cfg._n_patches > 1 else 1, dtype=cfg._npatches_dtype),
            votes=np.empty(cfg.n_classes, dtype=np.float32),
            grad=np.empty(cfg.n_classes, dtype=np.float32),
            y_hat=np.empty(cfg.n_classes, dtype=np.float32),
            loss=np.empty(1, dtype=np.float32),
            lr=lr,
            lambda_=lambda_,
            **extra,
            **self._fb_list_allocs(),
        )
        return bind_ptrs(buf)

    def fit_epoch(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        clause_drop_p: float,
        batch_size: int,
        lr: float | None,
        lambda_: float | None,
    ) -> float:
        cfg = self.config
        buf = self._fit_allocs(
            X,
            Y,
            self._fit_drop_mask(clause_drop_p),
            lr=cfg.lr if lr is None else lr,
            lambda_=cfg.lambda_ if lambda_ is None else lambda_,
        )

        running_loss = 0.0
        pbar = tqdm_bar(range(X.shape[0]), desc="Fit")
        for e, rng_key in self._fit_iter(pbar):
            self.fit_sample(rng_key, buf, e)
            running_loss += float(buf.loss[0])
            pbar.set_postfix(loss=f"{running_loss / (e + 1):.4f}")

        return running_loss / X.shape[0]

    def _fit_decide_fb(self, buf: GuidedFitBuffers, e: int, rng_key: int) -> None:
        cfg = self.config
        buf.t1b_n[0] = 0
        buf.t1a_t2_n[0] = 0
        p_Y_e = buf.Y[e].ctypes.data_as(float_p)

        self.lib.votes_activation(buf.p_votes, buf.p_y_hat)
        self.lib.loss_gradient(buf.p_y_hat, p_Y_e, buf.p_grad, buf.p_loss)

        if cfg._fb_signal == FbSignal.GRAD:
            self.lib.decide_feedback_grad(
                c_uint64(rng_key),
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
                c_uint64(rng_key),
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

    def _fit_update_weights(self, buf: GuidedFitBuffers) -> None:
        self.lib.update_weights(buf.p_grad, c_float(buf.lr), buf.p_clause_output, buf.p_clause_drop_mask, self.p_clause_weights)

    def calc_class_sums(self, X: np.ndarray, batch_size: int = -1, force_repack: bool = False) -> np.ndarray:
        votes = super().calc_class_sums(X, batch_size, force_repack)
        y_hat = np.empty_like(votes)
        self.lib.votes_activation_batch(votes.ctypes.data_as(float_p), c_int(votes.shape[0]), y_hat.ctypes.data_as(float_p))
        return y_hat

    def raw_votes(self, X: np.ndarray, batch_size: int = -1, force_repack: bool = False) -> np.ndarray:
        return super().calc_class_sums(X, batch_size, force_repack)
