from typing import ClassVar, Unpack

import numpy as np

from .backends.act_loss import SigmoidBCE, SoftmaxCE
from .base import BaseTM
from .config import T_Config


class Classifier(BaseTM):
    _DECISION_THRESHOLDS: ClassVar[dict[str, float]] = {"sigmoid": 0.5, "identity": 0.0, "softmax": 0.5}

    @property
    def decision_threshold(self) -> float | None:
        return getattr(self, "_decision_threshold", self._DECISION_THRESHOLDS.get(self.config.act_loss.act))

    @decision_threshold.setter
    def decision_threshold(self, value: float) -> None:
        self._decision_threshold = value

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        shuffle: bool = True,
        clause_drop_p: float = 0.0,
        lr: float | None = None,
        lambda_: float | None = None,
    ) -> float:
        assert X.shape[0] == Y.shape[0], "X and Y must have the same number of samples."
        return self._fit(X, self._prepare_Y(Y), shuffle, clause_drop_p, lr, lambda_)


class MultiClassTM(Classifier):
    def __init__(self, n_clauses: int, s: float, dim: int | tuple[int, ...], n_classes: int, **opt: Unpack[T_Config]):
        opt.setdefault("act_loss", SoftmaxCE())
        super().__init__(n_clauses, s, dim, n_classes, **opt)

    def predict(self, X: np.ndarray, batch_size: int = -1, force_repack: bool = False):
        class_sums = self.score(X, batch_size, force_repack=force_repack)
        return np.argmax(class_sums, axis=1), class_sums


class MultiOutputTM(Classifier):
    def __init__(self, n_clauses: int, s: float, dim: int | tuple[int, ...], n_classes: int, **opt: Unpack[T_Config]):
        opt.setdefault("act_loss", SigmoidBCE())
        super().__init__(n_clauses, s, dim, n_classes, **opt)

    def predict(self, X: np.ndarray, batch_size: int = -1, force_repack: bool = False):
        class_sums = self.score(X, batch_size, force_repack=force_repack)
        threshold = self.decision_threshold
        preds = class_sums if threshold is None else (class_sums > threshold).astype(np.uint32)
        return preds, class_sums


class BinaryTM(Classifier):
    def __init__(self, n_clauses: int, s: float, dim: int | tuple[int, ...], **opt: Unpack[T_Config]):
        opt.setdefault("act_loss", SigmoidBCE())
        opt.setdefault("coalesced", False)
        super().__init__(n_clauses, s, dim, 1, **opt)

    def predict(self, X: np.ndarray, batch_size: int = -1, force_repack: bool = False):
        class_sums = self.score(X, batch_size, force_repack=force_repack)
        threshold = self.decision_threshold
        preds = class_sums[:, 0] if threshold is None else (class_sums[:, 0] > threshold).astype(np.uint32)
        return preds, class_sums
