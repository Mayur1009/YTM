import pathlib
from dataclasses import dataclass
from typing import Any

import cupy as cp
import numpy as np

from ..._core.backends.base import FitBuffers
from ..._core.backends.cuda import CUDADevice as CoreCUDADevice
from ..._core.utils import read_file, tqdm_bar
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
            grad=cp.empty(cfg.n_classes, dtype=np.float32),
            y_hat=cp.empty(cfg.n_classes, dtype=np.float32),
            loss=cp.empty(1, dtype=np.float32),
            feedback_type=feedback_type,
            lr=lr,
            lambda_=lambda_,
            **extra,
            **self._fb_list_allocs(),
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
            # The fit stream does not wait on the default stream, so finish earlier work first.
            self.cuda_dev.synchronize()
            with self._fit_stream:
                buf = self._fit_allocs(
                    self._fit_drop_mask(clause_drop_p),
                    lr=cfg.lr if lr is None else lr,
                    lambda_=cfg.lambda_ if lambda_ is None else lambda_,
                )

                running_loss = cp.zeros(1, dtype=np.float32)
                keys = self._fit_keys(N)
                pbar = tqdm_bar(None, total=N, desc="Fit")

                def sample_kernels():
                    self._fit_sample_kernels(buf)
                    cp.add(running_loss, buf.loss, out=running_loss)

                for i in range(0, N, bs):
                    end = min(i + bs, N)
                    buf.X = cp.asarray(X[i:end], dtype=cfg._fbound_dtype)
                    buf.Y = cp.asarray(Y[i:end], dtype=np.float32)

                    self._run_batch(buf, keys[i:end], sample_kernels, pbar)
                pbar.close()
            self._fit_stream.synchronize()
            return float(running_loss[0]) / N

    def fit_sample(self, rng_key: int, buf: GuidedFitBuffers, e: int) -> None:
        with self.cuda_dev:
            self._set_sample(buf, e, rng_key)
            self._fit_sample_kernels(buf)

    def _fit_sample_kernels(self, buf: GuidedFitBuffers) -> None:
        """One sample's kernels. They read the sample index and rng key on the device, so the stage arguments are unused."""
        self.pack_clauses()
        self._fit_eval(buf, 0, 0)
        self._fit_voting(buf)
        self._fit_decide_fb(buf, 0, 0)
        self._fit_apply_fb(buf, 0, 0)
        if self.config.weighted:
            self._fit_update_weights(buf)

    def _fit_decide_fb(self, buf: GuidedFitBuffers, e: int, rng_key: int) -> None:
        cfg = self.config
        self.cu.votes_activation(1, (buf.votes, buf.y_hat))
        self.cu.loss_gradient(1, (buf.y_hat, buf.Y, buf.e_dev, buf.grad, buf.loss))

        if cfg._fb_signal == FbSignal.GRAD:
            self.cu.decide_feedback_grad(
                cfg._total_clauses,
                (
                    buf.keys,
                    buf.e_dev,
                    buf.grad,
                    self.clause_weights,
                    self.packed_clauses.clause_len,
                    buf.clause_output,
                    buf.clause_drop_mask,
                    np.float32(buf.lambda_),
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
        else:
            self.cu.compute_votes_neg_ck(
                cfg._total_clauses * cfg.n_classes,
                (buf.votes, self.clause_weights, buf.clause_output, buf.votes_neg_ck),
            )
            self.cu.votes_activation_batch(
                cfg._total_clauses,
                (buf.votes_neg_ck, np.int32(cfg._total_clauses), buf.y_hat_neg_ck),
            )
            self.cu.compute_loss_neg_ck(cfg._total_clauses, (buf.y_hat_neg_ck, buf.Y, buf.e_dev, buf.loss_neg_ck))
            self.cu.decide_feedback_delta_l(
                cfg._total_clauses,
                (
                    buf.keys,
                    buf.e_dev,
                    buf.loss,
                    buf.loss_neg_ck,
                    self.packed_clauses.clause_len,
                    buf.clause_output,
                    buf.clause_drop_mask,
                    np.float32(buf.lambda_),
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
