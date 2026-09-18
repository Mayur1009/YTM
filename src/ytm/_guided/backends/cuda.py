import pathlib
from dataclasses import dataclass
from typing import Any

import cupy as cp
import numpy as np

from ..._core.backends.cuda import CUDADevice as CoreCUDADevice
from ..._core.backends.cuda import read_file
from ..._core.utils import FitBuffers, tqdm_bar
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

    def _kernel_names(self) -> tuple[str, ...]:
        names = super()._kernel_names() + (
            "votes_activation",
            "votes_activation_batch",
            "loss_gradient",
            "update_clauses",
            "update_weights",
        )
        if self.config._fb_signal == FbSignal.DELTA_L:
            return names + ("compute_votes_neg_ck", "compute_loss_neg_ck", "decide_feedback_delta_l")
        return names + ("decide_feedback_grad",)

    def _fit_allocs(self, X: np.ndarray, Y: np.ndarray, clause_drop_mask, lr: float, lambda_: float) -> GuidedFitBuffers:
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
            X=cp.asarray(X, dtype=np.int32),
            Y=cp.asarray(Y, dtype=np.float32),
            clause_drop_mask=clause_drop_mask,
            selected_pids=cp.empty(cfg._total_clauses, dtype=np.int32),
            votes=cp.empty(cfg.n_classes, dtype=np.float32),
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

        with self.cuda_dev:
            buf = self._fit_allocs(
                X,
                Y,
                self._fit_drop_mask(clause_drop_p),
                lr=cfg.lr if lr is None else lr,
                lambda_=cfg.lambda_ if lambda_ is None else lambda_,
            )

            running_loss = cp.zeros(1, dtype=np.float32)
            pbar = tqdm_bar(range(X.shape[0]), desc="Fit")
            for e, rng_key in self._fit_samples(pbar):
                self.fit_sample(rng_key, buf, e)
                running_loss += buf.loss

            return float(running_loss[0]) / X.shape[0]

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
        one = self._kernel_config(1)
        per_clause = self._kernel_config(cfg._total_clauses)

        self.k_votes_activation(*one, (buf.votes, buf.y_hat))
        self.k_loss_gradient(*one, (buf.y_hat, buf.Y[e], buf.grad, buf.loss))

        if cfg._fb_signal == FbSignal.GRAD:
            self.k_decide_feedback_grad(
                *per_clause,
                (
                    np.uint64(rng_key),
                    buf.grad,
                    self.clause_weights,
                    self.packed_clauses.clause_density,
                    buf.selected_pids,
                    buf.clause_drop_mask,
                    np.float32(buf.lambda_),
                    buf.feedback_type,
                ),
            )
        else:
            self.k_compute_votes_neg_ck(
                *self._kernel_config(cfg._total_clauses * cfg.n_classes),
                (buf.votes, self.clause_weights, buf.selected_pids, buf.votes_neg_ck),
            )
            self.k_votes_activation_batch(
                *per_clause,
                (buf.votes_neg_ck, np.int32(cfg._total_clauses), buf.y_hat_neg_ck),
            )
            self.k_compute_loss_neg_ck(*per_clause, (buf.y_hat_neg_ck, buf.Y[e], buf.loss_neg_ck))
            self.k_decide_feedback_delta_l(
                *per_clause,
                (
                    np.uint64(rng_key),
                    buf.loss,
                    buf.loss_neg_ck,
                    self.packed_clauses.clause_density,
                    buf.selected_pids,
                    buf.clause_drop_mask,
                    np.float32(buf.lambda_),
                    buf.feedback_type,
                ),
            )

    def _fit_apply_fb(self, buf: GuidedFitBuffers, e: int, rng_key: int) -> None:
        self.k_update_clauses(
            *self._kernel_config(self.config._total_clauses * self.device_config._cuda_props["warp_size"]),
            (
                np.uint64(rng_key),
                buf.selected_pids,
                buf.X,
                np.int32(e),
                self.feat_mins_gpu,
                self.literal_offsets_gpu,
                buf.feedback_type,
                self.ta_states,
                self.packed_clauses.is_clause_synced,
            ),
        )

    def _fit_update_weights(self, buf: GuidedFitBuffers) -> None:
        self.k_update_weights(
            *self._kernel_config(self.config._total_clauses),
            (buf.grad, np.float32(buf.lr), buf.selected_pids, buf.clause_drop_mask, self.clause_weights),
        )

    def calc_class_sums(self, X: np.ndarray, batch_size: int = -1, force_repack: bool = False) -> np.ndarray:
        votes = super().calc_class_sums(X, batch_size, force_repack)
        with self.cuda_dev:
            votes_gpu = cp.asarray(votes, dtype=np.float32)
            y_hat = cp.empty_like(votes_gpu)
            self.k_votes_activation_batch(
                *self._kernel_config(votes.shape[0]),
                (votes_gpu, np.int32(votes.shape[0]), y_hat),
            )
            return y_hat.get()

    def raw_votes(self, X: np.ndarray, batch_size: int = -1, force_repack: bool = False) -> np.ndarray:
        return super().calc_class_sums(X, batch_size, force_repack)
