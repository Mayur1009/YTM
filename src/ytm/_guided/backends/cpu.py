import pathlib
from ctypes import POINTER, c_float, c_int, c_uint8, c_uint64
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..._core.backends.cpu import CPUDevice as CoreCPUDevice
from ..._core.backends.cpu import CPUFitBuffers, float_p, read_file
from ..._core.utils import tqdm_bar
from ..config import FbSignal
from .base import BaseDevice as GuidedBaseDevice

uint8_p = POINTER(c_uint8)


@dataclass(kw_only=True)
class GuidedFitBuffers(CPUFitBuffers):
    grad: Any  # (n_classes,) loss gradient w.r.t. votes for the current sample
    y_hat: Any  # (n_classes,) activated votes for the current sample
    loss: Any  # (1,) loss for the current sample
    feedback_type: Any  # (total_clauses, n_classes) for fb_signal="grad", (total_clauses,) for "delta_l"
    lr: float  # resolved once per epoch: the call's override, or config.lr
    lambda_: float  # resolved once per epoch: the call's override, or config.lambda_
    votes_neg_ck: Any = None  # (total_clauses, n_classes) counterfactual votes, delta_l only
    y_hat_neg_ck: Any = None  # (total_clauses, n_classes) counterfactual activations, delta_l only
    loss_neg_ck: Any = None  # (total_clauses,) counterfactual loss, delta_l only

    p_grad: Any = field(init=False)
    p_y_hat: Any = field(init=False)
    p_loss: Any = field(init=False)
    p_feedback_type: Any = field(init=False)
    p_votes_neg_ck: Any = field(init=False, default=None)
    p_y_hat_neg_ck: Any = field(init=False, default=None)
    p_loss_neg_ck: Any = field(init=False, default=None)

    def __post_init__(self):
        super().__post_init__()
        self.p_grad = self.grad.ctypes.data_as(float_p)
        self.p_y_hat = self.y_hat.ctypes.data_as(float_p)
        self.p_loss = self.loss.ctypes.data_as(float_p)
        self.p_feedback_type = self.feedback_type.ctypes.data_as(uint8_p)

        if self.votes_neg_ck is not None:
            self.p_votes_neg_ck = self.votes_neg_ck.ctypes.data_as(float_p)
            self.p_y_hat_neg_ck = self.y_hat_neg_ck.ctypes.data_as(float_p)
            self.p_loss_neg_ck = self.loss_neg_ck.ctypes.data_as(float_p)


class CPUDevice(GuidedBaseDevice, CoreCPUDevice):
    def dev_init(self):
        super().dev_init()
        cfg = self.config
        self.loss_class_weights = self.xp.asarray(cfg.loss_fn_kwargs.get("class_weights", np.ones(cfg.n_classes)), dtype=np.float32)
        self.p_loss_class_weights = self.loss_class_weights.ctypes.data_as(float_p)

    def _code_sections(self) -> dict[str, str]:
        sections = super()._code_sections()
        here = pathlib.Path(__file__).parent
        sections["act_loss.h"] = read_file(here / "act_loss.h")
        sections["update.c"] = read_file(here / "update.c")
        return sections

    def _fit_allocs(self, X: np.ndarray, Y: np.ndarray, clause_drop_mask: np.ndarray, lr: float, lambda_: float) -> GuidedFitBuffers:
        cfg = self.config
        for name, arr, dtype in (("X", X, np.int32), ("Y", Y, np.float32)):
            assert arr.dtype == dtype and arr.flags.c_contiguous, (
                f"`{name}` must be C contiguous {np.dtype(dtype)}, got {arr.dtype} contiguous={arr.flags.c_contiguous}"
            )

        extra: dict[str, np.ndarray] = {}
        if cfg._fb_signal == FbSignal.DELTA_L:
            feedback_type = np.zeros(cfg._total_clauses, dtype=np.uint8)
            extra["votes_neg_ck"] = np.empty((cfg._total_clauses, cfg.n_classes), dtype=np.float32)
            extra["y_hat_neg_ck"] = np.empty((cfg._total_clauses, cfg.n_classes), dtype=np.float32)
            extra["loss_neg_ck"] = np.empty(cfg._total_clauses, dtype=np.float32)
        else:
            feedback_type = np.zeros((cfg._n_clauses, cfg.n_classes), dtype=np.uint8)

        return GuidedFitBuffers(
            X=X,
            Y=Y,
            clause_drop_mask=clause_drop_mask,
            selected_pids=np.empty(cfg._total_clauses, dtype=np.int32),
            votes=np.empty(cfg.n_classes, dtype=np.float32),
            grad=np.empty(cfg.n_classes, dtype=np.float32),
            y_hat=np.empty(cfg.n_classes, dtype=np.float32),
            loss=np.empty(1, dtype=np.float32),
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
        buf = self._fit_allocs(
            X,
            Y,
            self._fit_drop_mask(clause_drop_p),
            lr=cfg.lr if lr is None else lr,
            lambda_=cfg.lambda_ if lambda_ is None else lambda_,
        )

        running_loss = 0.0
        pbar = tqdm_bar(range(X.shape[0]), desc="Fit")
        for e, rng_key in self._fit_samples(pbar):
            self.fit_sample(rng_key, buf, e)
            running_loss += float(buf.loss[0])
            pbar.set_postfix(loss=f"{running_loss / (e + 1):.4f}")

        return running_loss / X.shape[0]

    def fit_sample(self, rng_key: int, buf: GuidedFitBuffers, e: int) -> None:
        self.pack_clauses()
        self._fit_eval(buf, e, rng_key)
        self._fit_voting(buf)
        self._fit_decide_fb(buf, e, rng_key)
        self._fit_apply_fb(buf, e, rng_key)
        if self.config.weighted:
            self._fit_update_weights(buf)

    def _fit_decide_fb(self, buf: GuidedFitBuffers, e: int, rng_key: int) -> None:
        cfg = self.config
        p_Y_e = buf.Y[e].ctypes.data_as(float_p)

        self.lib.votes_activation(buf.p_votes, buf.p_y_hat)
        self.lib.loss_gradient(buf.p_y_hat, p_Y_e, self.p_loss_class_weights, buf.p_grad, buf.p_loss)

        if cfg._fb_signal == FbSignal.GRAD:
            self.lib.decide_feedback_grad(
                c_uint64(rng_key),
                buf.p_grad,
                self.p_clause_weights,
                self.p_clause_density,
                buf.p_selected_pids,
                buf.p_clause_drop_mask,
                c_float(buf.lambda_),
                buf.p_feedback_type,
            )
        else:
            self.lib.compute_votes_neg_ck(buf.p_votes, self.p_clause_weights, buf.p_selected_pids, buf.p_votes_neg_ck)
            self.lib.votes_activation_batch(buf.p_votes_neg_ck, c_int(cfg._total_clauses), buf.p_y_hat_neg_ck)
            self.lib.compute_loss_neg_ck(buf.p_y_hat_neg_ck, p_Y_e, self.p_loss_class_weights, buf.p_loss_neg_ck)
            self.lib.decide_feedback_delta_l(
                c_uint64(rng_key),
                buf.p_loss,
                buf.p_loss_neg_ck,
                self.p_clause_density,
                buf.p_selected_pids,
                buf.p_clause_drop_mask,
                c_float(buf.lambda_),
                buf.p_feedback_type,
            )

    def _fit_apply_fb(self, buf: GuidedFitBuffers, e: int, rng_key: int) -> None:
        self.lib.update_clauses(
            c_uint64(rng_key),
            buf.p_selected_pids,
            buf.p_X,
            c_int(e),
            self.p_feat_mins,
            self.p_literal_offsets,
            buf.p_feedback_type,
            self.p_ta_states,
            self.p_is_clause_synced,
        )

    def _fit_update_weights(self, buf: GuidedFitBuffers) -> None:
        self.lib.update_weights(buf.p_grad, c_float(buf.lr), buf.p_selected_pids, buf.p_clause_drop_mask, self.p_clause_weights)

    def calc_class_sums(self, X: np.ndarray, force_repack: bool = False) -> np.ndarray:
        votes = super().calc_class_sums(X, force_repack)
        y_hat = np.empty_like(votes)
        self.lib.votes_activation_batch(votes.ctypes.data_as(float_p), c_int(votes.shape[0]), y_hat.ctypes.data_as(float_p))
        return y_hat

    def raw_votes(self, X: np.ndarray, force_repack: bool = False) -> np.ndarray:
        return super().calc_class_sums(X, force_repack)
