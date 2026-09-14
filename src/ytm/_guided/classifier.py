from typing import ClassVar, Unpack

import numpy as np

from .base import BaseTM
from .config import T_Config


class Classifier(BaseTM):
    _DECISION_THRESHOLDS: ClassVar[dict[str, float]] = {"sigmoid": 0.5, "identity": 0.0, "softmax": 0.5}

    @property
    def decision_threshold(self) -> float | None:
        return self._DECISION_THRESHOLDS.get(self.config.act_fn)

    @decision_threshold.setter
    def decision_threshold(self, value: float) -> None:
        self._DECISION_THRESHOLDS[self.config.act_fn] = value


class MultiClassTM(Classifier):
    def __init__(self, n_clauses: int, s: float, dim: int | tuple[int, ...], n_classes: int, **opt: Unpack[T_Config]):
        opt.setdefault("act_fn", "softmax")
        opt.setdefault("loss_fn", "ce")
        super().__init__(n_clauses, s, dim, n_classes, **opt)

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        shuffle: bool = True,
        clause_drop_p: float = 0.0,
        batch_size: int = -1,
        lr: float | None = None,
        lambda_: float | None = None,
    ) -> float:
        assert Y.ndim == 1, "Y must be 1D array (samples,)"
        assert X.shape[0] == Y.shape[0], "X and Y must have the same number of samples."

        one_hot_Y = np.zeros((Y.shape[0], self.config.n_classes), dtype=np.int8)
        for i in range(self.config.n_classes):
            one_hot_Y[:, i] = np.where(Y == i, 1, 0)

        return self._fit(X, one_hot_Y, shuffle, clause_drop_p, batch_size, lr, lambda_)

    def predict(self, X: np.ndarray, force_repack: bool = False):
        class_sums = self.score(X, force_repack)
        return np.argmax(class_sums, axis=1), class_sums


class MultiOutputTM(Classifier):
    def __init__(self, n_clauses: int, s: float, dim: int | tuple[int, ...], n_classes: int, **opt: Unpack[T_Config]):
        opt.setdefault("act_fn", "sigmoid")
        opt.setdefault("loss_fn", "ce")
        super().__init__(n_clauses, s, dim, n_classes, **opt)

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        shuffle: bool = True,
        clause_drop_p: float = 0.0,
        batch_size: int = -1,
        lr: float | None = None,
        lambda_: float | None = None,
    ) -> float:
        assert Y.ndim == 2, f"Y must be 2D array (samples, outputs), got {Y.ndim}D"
        assert X.shape[0] == Y.shape[0], "X and Y must have the same number of samples"
        return self._fit(X, Y, shuffle, clause_drop_p, batch_size, lr, lambda_)

    def predict(self, X: np.ndarray, force_repack: bool = False):
        class_sums = self.score(X, force_repack)
        threshold = self.decision_threshold
        preds = class_sums if threshold is None else (class_sums > threshold).astype(np.uint32)
        return preds, class_sums


class BinaryTM(Classifier):
    def __init__(self, n_clauses: int, s: float, dim: int | tuple[int, ...], **opt: Unpack[T_Config]):
        opt.setdefault("act_fn", "sigmoid")
        opt.setdefault("loss_fn", "ce")
        opt.setdefault("coalesced", False)
        super().__init__(n_clauses, s, dim, 1, **opt)

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        shuffle: bool = True,
        clause_drop_p: float = 0.0,
        batch_size: int = -1,
        lr: float | None = None,
        lambda_: float | None = None,
    ) -> float:
        assert Y.ndim == 1, "Y must be 1D array (samples,)"
        assert X.shape[0] == Y.shape[0], "X and Y must have the same number of samples."
        return self._fit(X, Y.reshape(-1, 1), shuffle, clause_drop_p, batch_size, lr, lambda_)

    def predict(self, X: np.ndarray, force_repack: bool = False):
        class_sums = self.score(X, force_repack)
        threshold = self.decision_threshold
        preds = class_sums[:, 0] if threshold is None else (class_sums[:, 0] > threshold).astype(np.uint32)
        return preds, class_sums
