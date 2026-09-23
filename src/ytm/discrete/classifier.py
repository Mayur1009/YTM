import abc
from typing import Unpack

import numpy as np

from .base import BaseTM
from .config import T_Config


class Classifier(BaseTM):
    @property
    def decision_threshold(self) -> float:
        return getattr(self, "_decision_threshold", 0.0)

    @decision_threshold.setter
    def decision_threshold(self, value: float) -> None:
        self._decision_threshold = value

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        shuffle: bool = True,
        clause_drop_p: float = 0.0,
        batch_size: int = -1,
        label_sampling: bool = False,
    ):
        assert X.shape[0] == Y.shape[0], "X and Y must have the same number of samples."
        return self._fit(X, self._prepare_Y(Y), shuffle, clause_drop_p, batch_size, label_sampling)

    @abc.abstractmethod
    def _prepare_Y(self, Y: np.ndarray) -> np.ndarray:
        """Check the Y shape this classifier accepts and return it as (samples, n_classes)."""
        ...


class MultiClassTM(Classifier):
    def _prepare_Y(self, Y: np.ndarray) -> np.ndarray:
        n_classes = self.config.n_classes
        assert Y.ndim in (1, 2), f"Y must be 1D (samples,) or 2D (samples, {n_classes}) one hot, got {Y.ndim}D"

        if Y.ndim == 2:
            assert Y.shape[1] == n_classes, f"One-hot Y must have n_classes ({n_classes}) columns, got {Y.shape[1]}"
            assert np.isin(Y, (0, 1)).all(), "One-hot Y must be binary containing only {0, 1}."
            assert np.all(Y.sum(axis=1) == 1), "One-hot Y must have only one label per sample."
            return Y

        one_hot_Y = np.empty((Y.shape[0], n_classes), dtype=np.int8)
        for i in range(n_classes):
            one_hot_Y[:, i] = np.where(Y == i, 1, 0)
        return one_hot_Y

    def predict(self, X: np.ndarray, batch_size: int = -1, clip_class_sums: bool = False, force_repack: bool = False):
        class_sums = self.score(X, batch_size, force_repack=force_repack, clip_class_sums=clip_class_sums)
        return np.argmax(class_sums, axis=1), class_sums

    def _label_sampler(self, encoded_Y: np.ndarray, label_sampling: bool) -> np.ndarray:
        cfg = self.config
        label_probs = super()._label_sampler(encoded_Y, label_sampling)
        if not label_sampling:
            return label_probs

        min_count = np.min(np.sum(encoded_Y == cfg._T_max, axis=0))
        for i in range(cfg.n_classes):
            class_indices = np.where(encoded_Y[:, i] == cfg._T_max)[0]
            if len(class_indices) > min_count:
                selected = self._rng.choice(class_indices, size=min_count, replace=False)
                label_probs[np.setdiff1d(class_indices, selected), i] = 0.0
                label_probs[selected, i] = 1.0
        return label_probs


class MultiOutputTM(Classifier):
    def _prepare_Y(self, Y: np.ndarray) -> np.ndarray:
        assert Y.ndim == 2, f"Y must be 2D array (samples, outputs), got {Y.ndim}D"
        return Y

    def predict(self, X: np.ndarray, batch_size: int = -1, clip_class_sums: bool = False, force_repack: bool = False):
        class_sums = self.score(X, batch_size, force_repack=force_repack, clip_class_sums=clip_class_sums)
        return (class_sums > self.decision_threshold).astype(np.uint32), class_sums

    def _label_sampler(self, encoded_Y: np.ndarray, label_sampling: bool) -> np.ndarray:
        cfg = self.config
        if not label_sampling:
            return super()._label_sampler(encoded_Y, label_sampling)

        n_neg = (encoded_Y == cfg._T_min).sum(axis=1, keepdims=True)
        label_probs = np.where(encoded_Y == cfg._T_max, 1.0, cfg.q / np.maximum(1, n_neg)).astype(np.float32)

        for c in range(cfg.n_classes):
            _balance_column(label_probs, encoded_Y, c, cfg._T_max, cfg._T_min, self._rng)
        return label_probs


class BinaryTM(Classifier):
    def __init__(
        self,
        n_clauses: int,
        T: float | tuple[float, float],
        s: float,
        dim: int | tuple[int, ...],
        **opt: Unpack[T_Config],
    ):
        opt.setdefault("coalesced", False)
        opt.setdefault("allow_polarity_change", False)
        super().__init__(n_clauses, T, s, dim, 1, **opt)

    def _prepare_Y(self, Y: np.ndarray) -> np.ndarray:
        assert Y.ndim in (1, 2), f"Y must be 1D (samples,) or 2D (samples, 1), got {Y.ndim}D"
        assert Y.ndim == 1 or Y.shape[1] == 1, f"Y must have a single column, got {Y.shape[1]}"
        assert np.isin(Y, (0, 1)).all(), "Y must be binary containing only {0, 1}."
        return Y.reshape(-1, 1)

    def predict(self, X: np.ndarray, batch_size: int = -1, clip_class_sums: bool = False, force_repack: bool = False):
        class_sums = self.score(X, batch_size, force_repack=force_repack, clip_class_sums=clip_class_sums)
        return (class_sums[:, 0] > self.decision_threshold).astype(np.uint32), class_sums

    def _label_sampler(self, encoded_Y: np.ndarray, label_sampling: bool) -> np.ndarray:
        cfg = self.config
        label_probs = np.ones_like(encoded_Y, dtype=np.float32)
        if label_sampling:
            _balance_column(label_probs, encoded_Y, 0, cfg._T_max, cfg._T_min, self._rng)
        return label_probs


def _balance_column(label_probs, encoded_Y, col, t_max, t_min, rng) -> None:
    pos = np.where(encoded_Y[:, col] == t_max)[0]
    neg = np.where(encoded_Y[:, col] == t_min)[0]
    min_count = min(len(pos), len(neg))

    for side in (pos, neg):
        if len(side) > min_count:
            selected = rng.choice(side, size=min_count, replace=False)
            label_probs[np.setdiff1d(side, selected), col] = 0.0
