import numpy as np

from ..._core.backends.base import BaseDevice
from ..._core.backends.buffers import FitBuffers
from .buffers import DiscreteFitBuffers


class TMDevice(BaseDevice):
    @staticmethod
    def _fit_buffers_cls() -> type[FitBuffers]:
        return DiscreteFitBuffers

    def pre_epoch(self, X: np.ndarray, Y: np.ndarray, clause_drop_p: float, label_probs: np.ndarray) -> FitBuffers:
        cfg = self.config
        xp = self.xp
        return super().pre_epoch(
            X,
            Y,
            clause_drop_p,
            feedback_type=xp.zeros((cfg._n_clauses, cfg.n_classes), dtype=np.uint8),
            prob=xp.zeros(cfg.n_classes, dtype=np.float32),
            label_probs=xp.ascontiguousarray(label_probs, dtype=np.float32),
        )
