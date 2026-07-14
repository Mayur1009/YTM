import numpy as np
from typing import Unpack
from .base import BaseTM
from .args import T_args


class Classifier(BaseTM):
    def __init__(self, n_clauses: int, T: float, s: float, dim: tuple, n_classes: int, **opt_args: Unpack[T_args]):
        super().__init__(n_clauses, (-T, T), s, dim, n_classes, **opt_args)

    def _encode_Y(self, Y: np.ndarray) -> np.ndarray:
        assert Y.ndim == 2, f"Y must be 2D array (samples, outputs), got {Y.ndim}D"
        assert np.unique(Y).tolist() == [0, 1], "Y must be binary (0 or 1)"
        return ((np.copy(Y).astype(np.float32) * 2) - 1) * self.args.T_max  # Convert {0, 1} to {-T_max, T_max}

    def _label_sampler(self, encoded_Y: np.ndarray, label_sampling: bool) -> np.ndarray:
        # Default behaviour for classifiers
        label_probs = np.full_like(
            encoded_Y, fill_value=self.args.q / max(1, self.args.n_classes - 1), dtype=np.float32
        )
        label_probs[encoded_Y == self.args.T_max] = 1.0
        return label_probs


class MultiClassTM(Classifier):
    def __init__(self, n_clauses: int, T: float, s: float, dim: tuple, n_classes: int, **opt_args: Unpack[T_args]):
        opt_args.setdefault("crit", "softmax_ce")
        super().__init__(n_clauses, T, s, dim, n_classes, **opt_args)

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        shuffle: bool = True,
        clause_drop_p: float = 0.0,
        batch_size: int = -1,
        label_sampling: bool = False,
        lr: float | None = None,
    ):
        assert Y.ndim == 1, "Y must be 1D array (samples,)"
        assert X.shape[0] == Y.shape[0], "X and Y must have the same number of samples."

        one_hot_Y = np.empty((Y.shape[0], self.args.n_classes), dtype=np.int8)
        for i in range(self.args.n_classes):
            one_hot_Y[:, i] = np.where(Y == i, 1, 0)

        return self._fit(X, one_hot_Y, shuffle, clause_drop_p, batch_size, label_sampling, lr=lr)

    def predict(self, X: np.ndarray, batch_size: int = -1, clip_class_sums: bool = False):
        class_sums = self.score(X, batch_size, clip_class_sums)
        preds = np.argmax(class_sums, axis=1)
        return preds, class_sums

    def _label_sampler(self, encoded_Y: np.ndarray, label_sampling: bool) -> np.ndarray:
        label_probs = super()._label_sampler(encoded_Y, label_sampling)
        if label_sampling:
            count_per_class = np.sum(encoded_Y == self.args.T_max, axis=0)
            min_count = np.min(count_per_class)
            for i in range(self.args.n_classes):
                class_indices = np.where(encoded_Y[:, i] == self.args.T_max)[0]
                if len(class_indices) > min_count:
                    selected_indices = self.np_rng.choice(class_indices, size=min_count, replace=False)
                    unselected_indices = np.setdiff1d(class_indices, selected_indices)
                    label_probs[selected_indices, i] = 1.0
                    label_probs[unselected_indices, i] = 0.0
        return label_probs


class MultiOutputTM(Classifier):
    def __init__(self, n_clauses: int, T: float, s: float, dim: tuple, n_classes: int, **opt_args: Unpack[T_args]):
        opt_args.setdefault("crit", "sigmoid_bce")
        super().__init__(n_clauses, T, s, dim, n_classes, **opt_args)

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        shuffle: bool = True,
        clause_drop_p: float = 0.0,
        batch_size: int = -1,
        label_sampling: bool = False,
        lr: float | None = None,
    ):
        assert Y.ndim == 2, f"Y must be 2D array (samples, outputs), got {Y.ndim}D"
        assert X.shape[0] == Y.shape[0], "X and Y must have the same number of samples"
        return self._fit(X, Y, shuffle, clause_drop_p, batch_size, label_sampling, lr=lr)

    def predict(self, X: np.ndarray, batch_size: int = -1, clip_class_sums: bool = False):
        class_sums = self.score(X, batch_size, clip_class_sums)
        preds = (class_sums >= 0).astype(np.uint32)
        return preds, class_sums

    def _label_sampler(self, encoded_Y: np.ndarray, label_sampling: bool) -> np.ndarray:
        if not label_sampling:
            return super()._label_sampler(encoded_Y, label_sampling)

        n_neg = (encoded_Y == self.args.T_min).sum(axis=1, keepdims=True)
        label_probs = np.where(encoded_Y == self.args.T_max, 1.0, self.args.q / np.maximum(1, n_neg)).astype(np.float32)

        for c in range(self.args.n_classes):
            pos_indices = np.where(encoded_Y[:, c] == self.args.T_max)[0]
            neg_indices = np.where(encoded_Y[:, c] == self.args.T_min)[0]
            min_count = min(len(pos_indices), len(neg_indices))

            if len(pos_indices) > min_count:
                selected = self.np_rng.choice(pos_indices, size=min_count, replace=False)
                unselected = np.setdiff1d(pos_indices, selected)
                label_probs[unselected, c] = 0.0

            if len(neg_indices) > min_count:
                selected = self.np_rng.choice(neg_indices, size=min_count, replace=False)
                unselected = np.setdiff1d(neg_indices, selected)
                label_probs[unselected, c] = 0.0

        return label_probs


class BinaryTM(Classifier):
    def __init__(self, n_clauses: int, T: float, s: float, dim: tuple, **opt_args: Unpack[T_args]):
        opt_args.setdefault("crit", "sigmoid_bce")
        super().__init__(n_clauses, T, s, dim, 1, **opt_args)

    def _encode_Y(self, Y: np.ndarray) -> np.ndarray:
        return ((np.copy(Y).astype(np.float32) * 2) - 1) * self.args.T_max  # Convert {0, 1} to {-T_max, T_max}

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        shuffle: bool = True,
        clause_drop_p: float = 0.0,
        batch_size: int = -1,
        label_sampling: bool = False,
        lr: float | None = None,
    ):
        assert Y.ndim == 1, "Y must be 1D array (samples,)"
        assert np.unique(Y).tolist() == [0, 1], "Y must be binary (0 or 1)"
        assert X.shape[0] == Y.shape[0], "X and Y must have the same number of samples."
        return self._fit(X, Y.reshape(-1, 1), shuffle, clause_drop_p, batch_size, label_sampling, lr=lr)

    def predict(self, X: np.ndarray, batch_size: int = -1, clip_class_sums: bool = False):
        class_sums = self.score(X, batch_size, clip_class_sums)
        preds = (class_sums[:, 0] >= 0).astype(np.uint32)
        return preds, class_sums

    def _label_sampler(self, encoded_Y: np.ndarray, label_sampling: bool) -> np.ndarray:
        label_probs = np.ones_like(encoded_Y, dtype=np.float32)
        if label_sampling:
            pos_indices = np.where(encoded_Y[:, 0] == self.args.T_max)[0]
            neg_indices = np.where(encoded_Y[:, 0] == self.args.T_min)[0]
            min_count = min(len(pos_indices), len(neg_indices))

            if len(pos_indices) > min_count:
                selected = self.np_rng.choice(pos_indices, size=min_count, replace=False)
                unselected = np.setdiff1d(pos_indices, selected)
                label_probs[unselected, 0] = 0.0

            if len(neg_indices) > min_count:
                selected = self.np_rng.choice(neg_indices, size=min_count, replace=False)
                unselected = np.setdiff1d(neg_indices, selected)
                label_probs[unselected, 0] = 0.0

        return label_probs
