from typing import Unpack

import numpy as np

from .base import BaseTM
from .config import RegressionConfig, T_Config


class RegressionTM(BaseTM):
    """TM for regression: one real-valued target per sample.

    Targets in ``y_range`` are scaled to class sums in ``[0, T]``, and predictions are scaled back.
    Defaults differ from :class:`~ytm.tm.config.RegressionConfig`: ``negative_clauses=False``.

    Parameters
    ----------
    n_clauses : int
        Number of clauses.

    T : float
        Vote threshold. Class sums are clipped to ``[0, T]``.

    s : float
        Specificity, at least 1.0.

    dim : int or tuple of int
        Input shape: ``D``, ``(H, W)`` or ``(H, W, C)``.

    y_range : tuple of (float, float)
        Range ``(lo, hi)`` of the targets.

    **opt
        Other options of :class:`~ytm.tm.config.RegressionConfig`, and ``device`` and ``block_size``
        of :class:`~ytm._core.device_config.DeviceConfig`.
    """

    config: RegressionConfig

    @staticmethod
    def _config_cls() -> type[RegressionConfig]:
        return RegressionConfig

    def __init__(
        self,
        n_clauses: int,
        T: float,
        s: float,
        dim: int | tuple[int, ...],
        y_range: tuple[float, float],
        **opt: Unpack[T_Config],
    ):
        opt.setdefault("negative_clauses", False)
        super().__init__(n_clauses, (0.0, float(T)), s, dim, 1, y_range=y_range, **opt)

    def _encode_Y(self, Y: np.ndarray) -> np.ndarray:
        lo, hi = self.config.y_range
        return ((Y - lo) / (hi - lo)).astype(np.float32) * self.config._T_max

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        shuffle: bool = True,
        clause_drop_p: float = 0.0,
    ):
        """Train for one epoch.

        Parameters
        ----------
        X : array-like of int, shape (N, *dim) or (N, prod(dim))
            Inputs, each feature within ``[feat_mins, feat_maxs]``.

        Y : array-like of float, shape (N,)
            Targets, each within ``y_range``.

        shuffle : bool, default=True
            Visit the samples in a random order.

        clause_drop_p : float, default=0.0
            Probability of dropping each clause for this epoch (Drop Clause).
        """
        assert X.shape[0] == Y.shape[0], "X and Y must have the same number of samples"
        return self._fit(X, self.config.prepare_Y(Y), shuffle, clause_drop_p)

    def predict(self, X: np.ndarray, clip_class_sums: bool = True, force_repack: bool = False):
        """Predict the target of each sample: the class sum scaled from ``[0, T]`` back to ``y_range``.

        Parameters
        ----------
        X : array-like of int, shape (N, *dim) or (N, prod(dim))
            Inputs, each feature within ``[feat_mins, feat_maxs]``.

        clip_class_sums : bool, default=True
            Clip class sums to ``[0, T]`` before scaling, so predictions stay within ``y_range``.

        force_repack : bool, default=False
            Repack every clause first, not only the changed ones.

        Returns
        -------
        preds : ndarray of shape (N,)
            Predicted targets.

        class_sums : ndarray of shape (N, 1)
            Weighted vote sum.
        """
        lo, hi = self.config.y_range
        class_sums = self.score(X, force_repack=force_repack, clip_class_sums=clip_class_sums)
        preds = (class_sums[:, 0] / self.config._T_max) * (hi - lo) + lo
        return preds, class_sums
