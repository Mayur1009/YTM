from typing import ClassVar, Unpack

import numpy as np

from .backends.act_loss import SigmoidBCE, SoftmaxCE
from .base import BaseTM
from .config import T_Config


class Classifier(BaseTM):
    _DECISION_THRESHOLDS: ClassVar[dict[str, float]] = {"sigmoid": 0.5, "identity": 0.0, "softmax": 0.5}

    @property
    def decision_threshold(self) -> float | None:
        """Activated output above which ``predict`` outputs 1, for multi-label and binary models. Default depends on the activation: 0.5 for sigmoid and softmax, 0.0 for identity."""
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

        lr : float, default=None
            Learning rate for this epoch. None uses the config's ``lr``.

        lambda_ : float, default=None
            Feedback rate for this epoch. None uses the config's ``lambda_``.

        Returns
        -------
        float
            Mean loss per sample over the epoch.
        """
        assert X.shape[0] == Y.shape[0], "X and Y must have the same number of samples."
        return self._fit(X, self.config.prepare_Y(Y), shuffle, clause_drop_p, lr, lambda_)


class MultiClassTM(Classifier):
    """Guided TM for multi-class classification: each sample has exactly one of ``n_classes`` classes.

    The default ``act_loss`` is :class:`~ytm.guided.backends.act_loss.SoftmaxCE`.

    Parameters
    ----------
    n_clauses : int
        Clauses per class. See :class:`~ytm.guided.config.TMConfig`.

    s : float
        Specificity, at least 1.0.

    dim : int or tuple of int
        Input shape: ``D``, ``(H, W)`` or ``(H, W, C)``.

    n_classes : int
        Number of classes.

    **opt
        Other options of :class:`~ytm.guided.config.TMConfig`, and ``device`` and ``block_size`` of
        :class:`~ytm._core.device_config.DeviceConfig`.
    """

    def __init__(self, n_clauses: int, s: float, dim: int | tuple[int, ...], n_classes: int, **opt: Unpack[T_Config]):
        opt.setdefault("act_loss", SoftmaxCE())
        super().__init__(n_clauses, s, dim, n_classes, **opt)

    def predict(self, X: np.ndarray, force_repack: bool = False):
        """Predict the class of each sample: the class with the highest activated output.

        Parameters
        ----------
        X : array-like of int, shape (N, *dim) or (N, prod(dim))
            Inputs, each feature within ``[feat_mins, feat_maxs]``.

        force_repack : bool, default=False
            Repack every clause first, not only the changed ones.

        Returns
        -------
        preds : ndarray of shape (N,)
            Predicted class.

        class_sums : ndarray of shape (N, n_classes)
            Activated output of each class.
        """
        class_sums = self.score(X, force_repack=force_repack)
        return np.argmax(class_sums, axis=1), class_sums


class MultiOutputTM(Classifier):
    """Guided TM for multi-label classification: each sample can have any number of ``n_classes`` labels.

    The default ``act_loss`` is :class:`~ytm.guided.backends.act_loss.SigmoidBCE`.

    Parameters
    ----------
    n_clauses : int
        Clauses per class. See :class:`~ytm.guided.config.TMConfig`.

    s : float
        Specificity, at least 1.0.

    dim : int or tuple of int
        Input shape: ``D``, ``(H, W)`` or ``(H, W, C)``.

    n_classes : int
        Number of labels.

    **opt
        Other options of :class:`~ytm.guided.config.TMConfig`, and ``device`` and ``block_size`` of
        :class:`~ytm._core.device_config.DeviceConfig`.
    """

    def __init__(self, n_clauses: int, s: float, dim: int | tuple[int, ...], n_classes: int, **opt: Unpack[T_Config]):
        opt.setdefault("act_loss", SigmoidBCE())
        super().__init__(n_clauses, s, dim, n_classes, **opt)

    def predict(self, X: np.ndarray, force_repack: bool = False):
        """Predict the labels of each sample: 1 where the activated output is above ``decision_threshold``.

        Parameters
        ----------
        X : array-like of int, shape (N, *dim) or (N, prod(dim))
            Inputs, each feature within ``[feat_mins, feat_maxs]``.

        force_repack : bool, default=False
            Repack every clause first, not only the changed ones.

        Returns
        -------
        preds : ndarray of shape (N, n_classes)
            Predicted labels, 0 or 1 as uint32. The activated outputs themselves when
            ``decision_threshold`` is None.

        class_sums : ndarray of shape (N, n_classes)
            Activated output of each label.
        """
        class_sums = self.score(X, force_repack=force_repack)
        threshold = self.decision_threshold
        preds = class_sums if threshold is None else (class_sums > threshold).astype(np.uint32)
        return preds, class_sums


class BinaryTM(Classifier):
    """Guided TM for binary classification: one output, each sample is 0 or 1.

    Defaults differ from :class:`~ytm.guided.config.TMConfig`: ``act_loss`` is
    :class:`~ytm.guided.backends.act_loss.SigmoidBCE` and ``coalesced=False``.

    Parameters
    ----------
    n_clauses : int
        Number of clauses.

    s : float
        Specificity, at least 1.0.

    dim : int or tuple of int
        Input shape: ``D``, ``(H, W)`` or ``(H, W, C)``.

    **opt
        Other options of :class:`~ytm.guided.config.TMConfig`, and ``device`` and ``block_size`` of
        :class:`~ytm._core.device_config.DeviceConfig`.
    """

    def __init__(self, n_clauses: int, s: float, dim: int | tuple[int, ...], **opt: Unpack[T_Config]):
        opt.setdefault("act_loss", SigmoidBCE())
        opt.setdefault("coalesced", False)
        super().__init__(n_clauses, s, dim, 1, **opt)

    def predict(self, X: np.ndarray, force_repack: bool = False):
        """Predict each sample: 1 where the activated output is above ``decision_threshold``, else 0.

        Parameters
        ----------
        X : array-like of int, shape (N, *dim) or (N, prod(dim))
            Inputs, each feature within ``[feat_mins, feat_maxs]``.

        force_repack : bool, default=False
            Repack every clause first, not only the changed ones.

        Returns
        -------
        preds : ndarray of shape (N,)
            Predicted output, 0 or 1 as uint32. The activated output itself when
            ``decision_threshold`` is None.

        class_sums : ndarray of shape (N, 1)
            Activated output.
        """
        class_sums = self.score(X, force_repack=force_repack)
        threshold = self.decision_threshold
        preds = class_sums[:, 0] if threshold is None else (class_sums[:, 0] > threshold).astype(np.uint32)
        return preds, class_sums
