import numpy as np

from ..._core.backends.base import BaseDevice
from ..._core.backends.buffers import FitBuffers
from ..config import FbSignal
from .buffers import GuidedFitBuffers


class GuidedDevice(BaseDevice):
    @staticmethod
    def _fit_buffers_cls() -> type[FitBuffers]:
        return GuidedFitBuffers

    def pre_epoch(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        clause_drop_p: float,
        lr: float | None,
        lambda_: float | None,
    ) -> FitBuffers:
        cfg = self.config
        xp = self.xp

        extra = {}
        if cfg._fb_signal == FbSignal.DELTA_L:
            extra["votes_neg_ck"] = xp.empty((cfg._total_clauses, cfg.n_classes), dtype=np.float32)
            extra["y_hat_neg_ck"] = xp.empty((cfg._total_clauses, cfg.n_classes), dtype=np.float32)
            extra["loss_neg_ck"] = xp.empty(cfg._total_clauses, dtype=np.float32)

        return super().pre_epoch(
            X,
            Y,
            clause_drop_p,
            grad=xp.empty(cfg.n_classes, dtype=np.float32),
            y_hat=xp.empty(cfg.n_classes, dtype=np.float32),
            loss=xp.empty(1, dtype=np.float32),
            lr=cfg.lr if lr is None else lr,
            lambda_=cfg.lambda_ if lambda_ is None else lambda_,
            **extra,
        )

    def post_sample(self, buf: GuidedFitBuffers, e: int):
        return buf.loss[0].astype(np.float64)

    def post_epoch(self, buf: GuidedFitBuffers, total) -> float:
        return float(total) / buf.X.shape[0]
