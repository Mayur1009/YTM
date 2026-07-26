import numpy as np
from typing import Unpack
from .base import BaseTM
from .args import T_args


class Classifier(BaseTM):
    _DECISION_THRESHOLDS = {"sigmoid": 0.5, "identity": 0.0, "softmax": 0.5}

    def __init__(self, n_clauses: int, s: float, dim: tuple, n_classes: int, **opt_args: Unpack[T_args]):
        super().__init__(n_clauses, s, dim, n_classes, **opt_args)

    @property
    def decision_threshold(self) -> float | None:
        key = self.args.act_fn if isinstance(self.args.act_fn, str) else "custom"
        return self._DECISION_THRESHOLDS.get(key)

    @decision_threshold.setter
    def decision_threshold(self, value: float) -> None:
        key = self.args.act_fn if isinstance(self.args.act_fn, str) else "custom"
        self._DECISION_THRESHOLDS[key] = value

    def _encode_Y(self, Y: np.ndarray) -> np.ndarray:
        assert Y.ndim == 2, f"Y must be 2D array (samples, outputs), got {Y.ndim}D"
        return Y.astype(np.float32)

class MultiClassTM(Classifier):
    def __init__(self, n_clauses: int, s: float, dim: tuple, n_classes: int, **opt_args: Unpack[T_args]):
        opt_args.setdefault("act_fn", "softmax")
        opt_args.setdefault("loss_fn", "ce")
        super().__init__(n_clauses, s, dim, n_classes, **opt_args)

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        shuffle: bool = True,
        clause_drop_p: float = 0.0,
        batch_size: int = -1,
        lr: float | None = None,
    ):
        assert Y.ndim == 1, "Y must be 1D array (samples,)"
        assert X.shape[0] == Y.shape[0], "X and Y must have the same number of samples."

        one_hot_Y = np.empty((Y.shape[0], self.args.n_classes), dtype=np.int8)
        for i in range(self.args.n_classes):
            one_hot_Y[:, i] = np.where(Y == i, 1, 0)

        return self._fit(X, one_hot_Y, shuffle, clause_drop_p, batch_size, lr=lr)

    def predict(self, X: np.ndarray, batch_size: int = -1):
        class_sums = self.score(X, batch_size)
        preds = np.argmax(class_sums, axis=1)
        return preds, class_sums


class MultiOutputTM(Classifier):
    def __init__(self, n_clauses: int, s: float, dim: tuple, n_classes: int, **opt_args: Unpack[T_args]):
        opt_args.setdefault("act_fn", "sigmoid")
        opt_args.setdefault("loss_fn", "ce")
        super().__init__(n_clauses, s, dim, n_classes, **opt_args)

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        shuffle: bool = True,
        clause_drop_p: float = 0.0,
        batch_size: int = -1,
        lr: float | None = None,
    ):
        assert Y.ndim == 2, f"Y must be 2D array (samples, outputs), got {Y.ndim}D"
        assert X.shape[0] == Y.shape[0], "X and Y must have the same number of samples"
        return self._fit(X, Y, shuffle, clause_drop_p, batch_size, lr=lr)

    def predict(self, X: np.ndarray, batch_size: int = -1):
        class_sums = self.score(X, batch_size)
        threshold = self.decision_threshold
        preds = class_sums if threshold is None else (class_sums >= threshold).astype(np.uint32)
        return preds, class_sums


class BinaryTM(Classifier):
    def __init__(self, n_clauses: int, s: float, dim: tuple, **opt_args: Unpack[T_args]):
        opt_args.setdefault("act_fn", "sigmoid")
        opt_args.setdefault("loss_fn", "ce")
        super().__init__(n_clauses, s, dim, 1, **opt_args)

    def _encode_Y(self, Y: np.ndarray) -> np.ndarray:
        return Y.astype(np.float32)

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        shuffle: bool = True,
        clause_drop_p: float = 0.0,
        batch_size: int = -1,
        lr: float | None = None,
    ):
        assert Y.ndim == 1, "Y must be 1D array (samples,)"
        assert X.shape[0] == Y.shape[0], "X and Y must have the same number of samples."
        return self._fit(X, Y.reshape(-1, 1), shuffle, clause_drop_p, batch_size, lr=lr)

    def predict(self, X: np.ndarray, batch_size: int = -1):
        class_sums = self.score(X, batch_size)
        threshold = self.decision_threshold
        preds = class_sums[:, 0] if threshold is None else (class_sums[:, 0] >= threshold).astype(np.uint32)
        return preds, class_sums
