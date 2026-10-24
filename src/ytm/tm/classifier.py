from typing import Unpack

import numpy as np

from .base import BaseTM
from .config import T_Config


class Classifier(BaseTM):
    @property
    def decision_threshold(self) -> float:
        """Class sum above which ``predict`` outputs 1, for multi-label and binary models. Default 0.0."""
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
        label_sampling: bool = False,
    ):
        """Train for one epoch.

        Parameters
        ----------
        X : array-like of int, shape (N, *dim) or (N, prod(dim))
            Inputs, each feature within ``[feat_mins, feat_maxs]``.

        Y : array-like of int
            Targets: class labels of shape (N,) in ``[0, n_classes)``, or one-hot or multi-hot of
            shape (N, n_classes). For ``BinaryTM``, 0 or 1 of shape (N,).

        shuffle : bool, default=True
            Visit the samples in a random order.

        clause_drop_p : float, default=0.0
            Probability of dropping each clause for this epoch (Drop Clause).
        """
        assert X.shape[0] == Y.shape[0], "X and Y must have the same number of samples."
        return self._fit(X, self.config.prepare_Y(Y), shuffle, clause_drop_p, label_sampling)


class MultiClassTM(Classifier):
    """TM for multi-class classification: each sample has exactly one of ``n_classes`` classes.

    Parameters
    ----------
    n_clauses : int
        Clauses per class. See :class:`~ytm.tm.config.TMConfig`.

    T : float or tuple of (float, float)
        Vote threshold.

    s : float
        Specificity, at least 1.0.

    dim : int or tuple of int
        Input shape: ``D``, ``(H, W)`` or ``(H, W, C)``.

    n_classes : int
        Number of classes.

    **opt
        Other options of :class:`~ytm.tm.config.TMConfig`, and ``device`` and ``block_size`` of
        :class:`~ytm._core.device_config.DeviceConfig`.
    """

    def predict(self, X: np.ndarray, clip_class_sums: bool = False, force_repack: bool = False):
        """Predict the class of each sample: the class with the highest class sum.

        Parameters
        ----------
        X : array-like of int, shape (N, *dim) or (N, prod(dim))
            Inputs, each feature within ``[feat_mins, feat_maxs]``.

        clip_class_sums : bool, default=False
            Clip the returned class sums to ``[T_min, T_max]``.

        force_repack : bool, default=False
            Repack every clause first, not only the changed ones.

        Returns
        -------
        preds : ndarray of shape (N,)
            Predicted class.

        class_sums : ndarray of shape (N, n_classes)
            Weighted vote sum of each class.
        """
        class_sums = self.score(X, force_repack=force_repack, clip_class_sums=clip_class_sums)
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
    """TM for multi-label classification: each sample can have any number of ``n_classes`` labels.

    Parameters
    ----------
    n_clauses : int
        Clauses per class. See :class:`~ytm.tm.config.TMConfig`.

    T : float or tuple of (float, float)
        Vote threshold.

    s : float
        Specificity, at least 1.0.

    dim : int or tuple of int
        Input shape: ``D``, ``(H, W)`` or ``(H, W, C)``.

    n_classes : int
        Number of labels.

    **opt
        Other options of :class:`~ytm.tm.config.TMConfig`, and ``device`` and ``block_size`` of
        :class:`~ytm._core.device_config.DeviceConfig`.
    """

    def predict(self, X: np.ndarray, clip_class_sums: bool = False, force_repack: bool = False):
        """Predict the labels of each sample: 1 where the class sum is above ``decision_threshold``.

        Parameters
        ----------
        X : array-like of int, shape (N, *dim) or (N, prod(dim))
            Inputs, each feature within ``[feat_mins, feat_maxs]``.

        clip_class_sums : bool, default=False
            Clip the returned class sums to ``[T_min, T_max]``.

        force_repack : bool, default=False
            Repack every clause first, not only the changed ones.

        Returns
        -------
        preds : ndarray of shape (N, n_classes), uint32
            Predicted labels, 0 or 1.

        class_sums : ndarray of shape (N, n_classes)
            Weighted vote sum of each label.
        """
        class_sums = self.score(X, force_repack=force_repack, clip_class_sums=clip_class_sums)
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
    """TM for binary classification: one output, each sample is 0 or 1.

    Defaults differ from :class:`~ytm.tm.config.TMConfig`: ``coalesced=False`` and
    ``allow_polarity_change=False``.

    Parameters
    ----------
    n_clauses : int
        Number of clauses.

    T : float or tuple of (float, float)
        Vote threshold.

    s : float
        Specificity, at least 1.0.

    dim : int or tuple of int
        Input shape: ``D``, ``(H, W)`` or ``(H, W, C)``.

    **opt
        Other options of :class:`~ytm.tm.config.TMConfig`, and ``device`` and ``block_size`` of
        :class:`~ytm._core.device_config.DeviceConfig`.
    """

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

    def predict(self, X: np.ndarray, clip_class_sums: bool = False, force_repack: bool = False):
        """Predict each sample: 1 where the class sum is above ``decision_threshold``, else 0.

        Parameters
        ----------
        X : array-like of int, shape (N, *dim) or (N, prod(dim))
            Inputs, each feature within ``[feat_mins, feat_maxs]``.

        clip_class_sums : bool, default=False
            Clip the returned class sums to ``[T_min, T_max]``.

        force_repack : bool, default=False
            Repack every clause first, not only the changed ones.

        Returns
        -------
        preds : ndarray of shape (N,), uint32
            Predicted output, 0 or 1.

        class_sums : ndarray of shape (N, 1)
            Weighted vote sum.
        """
        class_sums = self.score(X, force_repack=force_repack, clip_class_sums=clip_class_sums)
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
