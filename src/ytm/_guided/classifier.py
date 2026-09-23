import abc
from typing import ClassVar, Unpack

import numpy as np

from .backends.act_loss import SigmoidBCE, SoftmaxCE
from .base import BaseTM
from .config import T_Config


class Classifier(BaseTM):
    _DECISION_THRESHOLDS: ClassVar[dict[str, float]] = {"sigmoid": 0.5, "identity": 0.0, "softmax": 0.5}

    @property
    def decision_threshold(self) -> float | None:
        return self._DECISION_THRESHOLDS.get(self.config.act_loss.act)

    @decision_threshold.setter
    def decision_threshold(self, value: float) -> None:
        self._DECISION_THRESHOLDS[self.config.act_loss.act] = value

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
        assert X.shape[0] == Y.shape[0], "X and Y must have the same number of samples."
        return self._fit(X, self._prepare_Y(Y), shuffle, clause_drop_p, batch_size, lr, lambda_)

    @abc.abstractmethod
    def _prepare_Y(self, Y: np.ndarray) -> np.ndarray:
        """Check the Y shape this classifier accepts and return it as (samples, n_classes)."""
        ...


class MultiClassTM(Classifier):
    def __init__(self, n_clauses: int, s: float, dim: int | tuple[int, ...], n_classes: int, **opt: Unpack[T_Config]):
        opt.setdefault("act_loss", SoftmaxCE())
        super().__init__(n_clauses, s, dim, n_classes, **opt)

    def _prepare_Y(self, Y: np.ndarray) -> np.ndarray:
        n_classes = self.config.n_classes
        assert Y.ndim in (1, 2), f"Y must be 1D (samples,) or 2D (samples, {n_classes}) one hot, got {Y.ndim}D"

        if Y.ndim == 2:
            assert Y.shape[1] == n_classes, f"One-hot Y must have n_classes ({n_classes}) columns, got {Y.shape[1]}"
            assert np.isin(Y, (0, 1)).all(), "One-hot Y must be binary containing only {0, 1}."
            assert np.all(Y.sum(axis=1) == 1), "One-hot Y must have only one label per sample."
            return Y

        one_hot_Y = np.zeros((Y.shape[0], n_classes), dtype=np.int8)
        for i in range(n_classes):
            one_hot_Y[:, i] = np.where(Y == i, 1, 0)
        return one_hot_Y

    def predict(self, X: np.ndarray, batch_size: int = -1, force_repack: bool = False):
        class_sums = self.score(X, batch_size, force_repack=force_repack)
        return np.argmax(class_sums, axis=1), class_sums


class MultiOutputTM(Classifier):
    def __init__(self, n_clauses: int, s: float, dim: int | tuple[int, ...], n_classes: int, **opt: Unpack[T_Config]):
        opt.setdefault("act_loss", SigmoidBCE())
        super().__init__(n_clauses, s, dim, n_classes, **opt)

    def _prepare_Y(self, Y: np.ndarray) -> np.ndarray:
        assert Y.ndim == 2, f"Y must be 2D array (samples, outputs), got {Y.ndim}D"
        return Y

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

    def _prepare_Y(self, Y: np.ndarray) -> np.ndarray:
        assert Y.ndim in (1, 2), f"Y must be 1D (samples,) or 2D (samples, 1), got {Y.ndim}D"
        assert Y.ndim == 1 or Y.shape[1] == 1, f"Y must have a single column, got {Y.shape[1]}"
        assert np.isin(Y, (0, 1)).all(), "Y must be binary containing only {0, 1}."
        return Y.reshape(-1, 1)

    def predict(self, X: np.ndarray, batch_size: int = -1, force_repack: bool = False):
        class_sums = self.score(X, batch_size, force_repack=force_repack)
        threshold = self.decision_threshold
        preds = class_sums[:, 0] if threshold is None else (class_sums[:, 0] > threshold).astype(np.uint32)
        return preds, class_sums
