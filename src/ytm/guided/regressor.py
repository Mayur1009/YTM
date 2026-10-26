from typing import Unpack

import numpy as np

from .backends.act_loss import MSE
from .base import BaseTM
from .config import T_Config


class RegressionTM(BaseTM):
    """Guided TM for regression: one real-valued target per sample.

    Defaults differ from :class:`~ytm.guided.config.TMConfig`: ``act_loss`` is
    :class:`~ytm.guided.backends.act_loss.MSE`.

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
        opt.setdefault("act_loss", MSE())
        super().__init__(n_clauses, s, dim, 1, **opt)

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
        X : array-like of int, shape ``(N, *dim)`` or ``(N, prod(dim))``
            Inputs, each feature within ``[feat_mins, feat_maxs]``.

        Y : array-like of float, shape (N,)
            Targets.

        shuffle : bool, default=True
            Visit the samples in a random order.

        clause_drop_p : float, default=0.0
            Probability of dropping each clause for this epoch (Drop Clause).

        lr : float, default=None
            Learning rate for this epoch. None uses the config's ``lr``.

        lambda_ : float, default=None
            Feedback strength for this epoch. None uses the config's ``lambda_``.

        Returns
        -------
        float
            Mean loss per sample over the epoch.
        """
        assert X.shape[0] == Y.shape[0], "X and Y must have the same number of samples"
        assert Y.ndim == 1, "Y must be 1D array (samples,)"
        return self._fit(X, Y.reshape(-1, 1), shuffle, clause_drop_p, lr, lambda_)

    def predict(self, X: np.ndarray, force_repack: bool = False):
        """Predict the target of each sample: the activated output.

        Parameters
        ----------
        X : array-like of int, shape ``(N, *dim)`` or ``(N, prod(dim))``
            Inputs, each feature within ``[feat_mins, feat_maxs]``.

        force_repack : bool, default=False
            Repack every clause first, not only the changed ones.

        Returns
        -------
        preds : ndarray of shape (N,)
            Predicted targets.

        class_sums : ndarray of shape (N, 1)
            Activated output, the same values as ``preds``.
        """
        class_sums = self.score(X, force_repack=force_repack)
        return class_sums[:, 0], class_sums
