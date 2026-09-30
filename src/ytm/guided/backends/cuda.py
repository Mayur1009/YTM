import pathlib
from dataclasses import dataclass
from typing import Any

import cupy as cp
import numpy as np

from ..._core.backends.cuda import CUDADevice as CoreCUDADevice
from ..._core.utils import FitBuffers, read_file, tqdm_bar
from ..config import FbSignal
from .base import BaseDevice as GuidedBaseDevice


@dataclass(kw_only=True)
class GuidedFitBuffers(FitBuffers):
    grad: Any  # (n_classes,) loss gradient w.r.t. votes for the current sample
    y_hat: Any  # (n_classes,) activated votes for the current sample
    loss: Any  # (1,) loss for the current sample
    feedback_type: Any  # (total_clauses, n_classes) for fb_signal="grad", (total_clauses,) for "delta_l"
    lr: float  # resolved once per epoch: the call's override, or config.lr
    lambda_: float  # resolved once per epoch: the call's override, or config.lambda_
    votes_neg_ck: Any = None  # (total_clauses, n_classes) counterfactual votes, delta_l only
    y_hat_neg_ck: Any = None  # (total_clauses, n_classes) counterfactual activations, delta_l only
    loss_neg_ck: Any = None  # (total_clauses,) counterfactual loss, delta_l only


class CUDADevice(GuidedBaseDevice, CoreCUDADevice):
    def _code_sections(self) -> dict[str, str]:
        sections = super()._code_sections()
        here = pathlib.Path(__file__).parent
        sections["act.h"] = read_file(here / "act.h")
        sections["loss"] = self.config.act_loss.src
        sections["update.cu"] = read_file(here / "update.cu")
        return sections

    def _fit_allocs(self, clause_drop_mask, lr: float, lambda_: float) -> GuidedFitBuffers:
        cfg = self.config

        extra: dict[str, Any] = {}
        if cfg._fb_signal == FbSignal.DELTA_L:
            feedback_type = cp.zeros(cfg._total_clauses, dtype=np.uint8)
            extra["votes_neg_ck"] = cp.empty((cfg._total_clauses, cfg.n_classes), dtype=np.float32)
            extra["y_hat_neg_ck"] = cp.empty((cfg._total_clauses, cfg.n_classes), dtype=np.float32)
            extra["loss_neg_ck"] = cp.empty(cfg._total_clauses, dtype=np.float32)
        else:
            feedback_type = cp.zeros((cfg._n_clauses, cfg.n_classes), dtype=np.uint8)

        return GuidedFitBuffers(
            X=None,
            Y=None,
            clause_drop_mask=clause_drop_mask,
            clause_output=cp.empty(cfg._total_clauses, dtype=np.int8),
            selected_pids=cp.empty(cfg._total_clauses if cfg._n_patches > 1 else 1, dtype=cfg._npatches_dtype),
            votes=cp.empty(cfg.n_classes, dtype=np.float32),
            fb_count=cp.zeros(1, dtype=np.uint32),
            fb_ids=cp.empty(cfg._total_clauses, dtype=np.uint32),
            grad=cp.empty(cfg.n_classes, dtype=np.float32),
            y_hat=cp.empty(cfg.n_classes, dtype=np.float32),
            loss=cp.empty(1, dtype=np.float32),
            feedback_type=feedback_type,
            lr=lr,
            lambda_=lambda_,
            **extra,
        )

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
        N = X.shape[0]
        bs = N if batch_size == -1 else batch_size

        with self.cuda_dev:
            buf = self._fit_allocs(
                self._fit_drop_mask(clause_drop_p),
                lr=cfg.lr if lr is None else lr,
                lambda_=cfg.lambda_ if lambda_ is None else lambda_,
            )

            running_loss = cp.zeros(1, dtype=np.float32)
            samples = self._fit_samples(tqdm_bar(range(N), desc="Fit"))

            for i in range(0, N, bs):
                end = min(i + bs, N)
                buf.X = cp.asarray(X[i:end], dtype=cfg._fbound_dtype)
                buf.Y = cp.asarray(Y[i:end], dtype=np.float32)

                for e in range(end - i):
                    _, rng_key = next(samples)
                    self.fit_sample(rng_key, buf, e)
                    running_loss += buf.loss

            return float(running_loss[0]) / N

    def fit_sample(self, rng_key: int, buf: GuidedFitBuffers, e: int) -> None:
        with self.cuda_dev:
            self.pack_clauses()
            self._fit_eval(buf, e, rng_key)
            self._fit_voting(buf)
            self._fit_decide_fb(buf, e, rng_key)
            self._fit_apply_fb(buf, e, rng_key)
            if self.config.weighted:
                self._fit_update_weights(buf)

    def _fit_decide_fb(self, buf: GuidedFitBuffers, e: int, rng_key: int) -> None:
        cfg = self.config
        buf.fb_count.data.memset_async(0, buf.fb_count.nbytes)

        self.cu.votes_activation(1, (buf.votes, buf.y_hat))
        self.cu.loss_gradient(1, (buf.y_hat, buf.Y[e], buf.grad, buf.loss))

        if cfg._fb_signal == FbSignal.GRAD:
            self.cu.decide_feedback_grad(
                cfg._total_clauses,
                (
                    np.uint64(rng_key),
                    buf.grad,
                    self.clause_weights,
                    self.packed_clauses.clause_len,
                    buf.clause_output,
                    buf.clause_drop_mask,
                    np.float32(buf.lambda_),
                    buf.feedback_type,
                    buf.fb_count,
                    buf.fb_ids,
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
                    np.uint64(rng_key),
                    buf.loss,
                    buf.loss_neg_ck,
                    self.packed_clauses.clause_len,
                    buf.clause_output,
                    buf.clause_drop_mask,
                    np.float32(buf.lambda_),
                    buf.feedback_type,
                    buf.fb_count,
                    buf.fb_ids,
                ),
            )

    def _fit_apply_fb(self, buf: GuidedFitBuffers, e: int, rng_key: int) -> None:
        self.cu.update_clauses(
            self.config._total_clauses * self._warp_size,
            (
                np.uint64(rng_key),
                buf.clause_output,
                buf.selected_pids,
                buf.X,
                np.int32(e),
                self.literal_offsets,
                buf.feedback_type,
                buf.fb_count,
                buf.fb_ids,
                self.ta_states,
                self.packed_clauses.is_clause_synced,
            ),
        )

    def _fit_update_weights(self, buf: GuidedFitBuffers) -> None:
        self.cu.update_weights(
            self.config._total_clauses,
            (buf.grad, np.float32(buf.lr), buf.clause_output, buf.clause_drop_mask, self.clause_weights),
        )

    def calc_class_sums(self, X: np.ndarray, batch_size: int = -1, force_repack: bool = False) -> np.ndarray:
        votes = super().calc_class_sums(X, batch_size, force_repack)
        with self.cuda_dev:
            votes_gpu = cp.asarray(votes, dtype=np.float32)
            y_hat = cp.empty_like(votes_gpu)
            self.cu.votes_activation_batch(
                votes.shape[0],
                (votes_gpu, np.int32(votes.shape[0]), y_hat),
            )
            return y_hat.get()

    def raw_votes(self, X: np.ndarray, batch_size: int = -1, force_repack: bool = False) -> np.ndarray:
        return super().calc_class_sums(X, batch_size, force_repack)
