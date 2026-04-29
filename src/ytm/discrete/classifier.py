import numpy as np
from typing import Unpack
from .base import BaseTM
from .args import T_args


class Classifier(BaseTM):
    def __init__(self, n_clauses: int, T: float, s: float, dim: tuple, n_classes: int, **opt_args: Unpack[T_args]):
        super().__init__(n_clauses, (-T, T), s, dim, n_classes, **opt_args)

    def _target_sampling(self, Y: np.ndarray) -> np.ndarray:
        assert Y.ndim == 2, f"Y must be 2D array (samples, outputs), got {Y.ndim}D"
        assert np.unique(Y).tolist() == [0, 1], "Y must be binary (0 or 1)"
        return ((np.copy(Y).astype(np.float32) * 2) - 1) * self.args.T_max  # Convert {0, 1} to {-T_max, T_max}


class MultiClassTM(Classifier):
    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        shuffle: bool = True,
        clause_drop_p: float = 0.0,
        batch_size: int = -1,
    ):
        assert Y.ndim == 1, "Y must be 1D array (samples,)"
        assert X.shape[0] == Y.shape[0], "X and Y must have the same number of samples."

        encoded_Y = np.empty((Y.shape[0], self.args.n_classes), dtype=np.int8)
        for i in range(self.args.n_classes):
            encoded_Y[:, i] = np.where(Y == i, 1, 0)

        return self._fit(X, encoded_Y, shuffle, clause_drop_p, batch_size)

    def predict(self, X: np.ndarray, batch_size: int = -1, clip_class_sums: bool = False):
        class_sums = self.score(X, batch_size, clip_class_sums)
        preds = np.argmax(class_sums, axis=1)
        return preds, class_sums


class MultiOutputTM(Classifier):
    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        shuffle: bool = True,
        clause_drop_p: float = 0.0,
        batch_size: int = -1,
    ):
        assert Y.ndim == 2, f"Y must be 2D array (samples, outputs), got {Y.ndim}D"
        assert X.shape[0] == Y.shape[0], "X and Y must have the same number of samples"

        return self._fit(X, Y, shuffle, clause_drop_p, batch_size)

    def predict(self, X: np.ndarray, batch_size: int = -1, clip_class_sums: bool = False):
        class_sums = self.score(X, batch_size, clip_class_sums)
        preds = (class_sums >= 0).astype(np.uint32)
        return preds, class_sums


class BinaryTM(Classifier):
    def __init__(self, n_clauses: int, T: float, s: float, dim: tuple, **opt_args: Unpack[T_args]):
        super().__init__(n_clauses, T, s, dim, 1, **opt_args)

    def _target_sampling(self, Y: np.ndarray) -> np.ndarray:
        return ((np.copy(Y).astype(np.float32) * 2) - 1) * self.args.T_max  # Convert {0, 1} to {-T_max, T_max}

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        shuffle: bool = True,
        clause_drop_p: float = 0.0,
        batch_size: int = -1,
    ):
        assert Y.ndim == 1, "Y must be 1D array (samples,)"
        assert np.unique(Y).tolist() == [0, 1], "Y must be binary (0 or 1)"
        assert X.shape[0] == Y.shape[0], "X and Y must have the same number of samples."
        encoded_Y = np.where(Y == 1, 1, 0).reshape(-1, 1).astype(np.float32)
        return self._fit(X, encoded_Y, shuffle, clause_drop_p, batch_size)

    def predict(self, X: np.ndarray, batch_size: int = -1, clip_class_sums: bool = False):
        class_sums = self.score(X, batch_size, clip_class_sums)
        preds = (class_sums[:, 0] >= 0).astype(np.uint32)
        return preds, class_sums
