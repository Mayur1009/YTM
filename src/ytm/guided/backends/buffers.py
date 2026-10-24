from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..._core.backends.buffers import FitBuffers
from ..config import FbSignal


@dataclass(kw_only=True)
class GuidedFitBuffers(FitBuffers):
    """Fit buffers of the guided TM. Adds to :class:`~ytm._core.backends.buffers.FitBuffers`:

    Attributes
    ----------
    grad : array of shape (n_classes,)
        Loss gradient w.r.t. the votes for the current sample.

    y_hat : array of shape (n_classes,)
        Activated votes for the current sample.

    loss : array of shape (1,)
        Loss for the current sample.

    lr, lambda_ : float
        Learning rate and feedback strength for this epoch: the ``fit`` call's value, or the
        config's.

    votes_neg_ck, y_hat_neg_ck : array of shape (total_clauses, n_classes)
        Votes and activations if each clause's output were flipped. ``fb_signal="delta_l"`` only,
        else None.

    loss_neg_ck : array of shape (total_clauses,)
        Loss if each clause's output were flipped. ``fb_signal="delta_l"`` only, else None.
    """

    grad: Any = field(init=False)  # (n_classes,) loss gradient w.r.t. votes for the current sample
    y_hat: Any = field(init=False)  # (n_classes,) activated votes for the current sample
    loss: Any = field(init=False)  # (1,) loss for the current sample
    lr: float = field(init=False, default=0.0)  # resolved once per epoch: the call's override, or config.lr
    lambda_: float = field(init=False, default=0.0)  # resolved once per epoch: the call's override, or config.lambda_
    votes_neg_ck: Any = field(init=False, default=None)  # (total_clauses, n_classes) counterfactual votes, delta_l only
    y_hat_neg_ck: Any = field(init=False, default=None)  # (total_clauses, n_classes) counterfactual activations, delta_l only
    loss_neg_ck: Any = field(init=False, default=None)  # (total_clauses,) counterfactual loss, delta_l only

    def __post_init__(self) -> None:
        super().__post_init__()
        cfg, xp = self.cfg, self.xp
        self.grad = xp.empty(cfg.n_classes, dtype=np.float32)
        self.y_hat = xp.empty(cfg.n_classes, dtype=np.float32)
        self.loss = xp.empty(1, dtype=np.float32)
        if cfg._fb_signal == FbSignal.DELTA_L:
            self.votes_neg_ck = xp.empty((cfg._total_clauses, cfg.n_classes), dtype=np.float32)
            self.y_hat_neg_ck = xp.empty((cfg._total_clauses, cfg.n_classes), dtype=np.float32)
            self.loss_neg_ck = xp.empty(cfg._total_clauses, dtype=np.float32)
