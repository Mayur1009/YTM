from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..._core.backends.buffers import FitBuffers
from ..config import FbSignal


@dataclass(kw_only=True)
class GuidedFitBuffers(FitBuffers):
    grad: Any = field(init=False)  # (n_classes,) loss gradient w.r.t. votes for the current sample
    y_hat: Any = field(init=False)  # (n_classes,) activated votes for the current sample
    loss: Any = field(init=False)  # (1,) loss for the current sample
    lr: float | None = None  # resolved once per epoch: the call's override, or config.lr
    lambda_: float | None = None  # resolved once per epoch: the call's override, or config.lambda_
    votes_neg_ck: Any = field(init=False, default=None)  # (total_clauses, n_classes) counterfactual votes, delta_l only
    y_hat_neg_ck: Any = field(init=False, default=None)  # (total_clauses, n_classes) counterfactual activations, delta_l only
    loss_neg_ck: Any = field(init=False, default=None)  # (total_clauses,) counterfactual loss, delta_l only

    def __post_init__(self, cfg) -> None:
        super().__post_init__(cfg)
        xp = self.xp
        self.grad = xp.empty(cfg.n_classes, dtype=np.float32)
        self.y_hat = xp.empty(cfg.n_classes, dtype=np.float32)
        self.loss = xp.empty(1, dtype=np.float32)
        self.lr = cfg.lr if self.lr is None else self.lr
        self.lambda_ = cfg.lambda_ if self.lambda_ is None else self.lambda_
        if cfg._fb_signal == FbSignal.DELTA_L:
            self.votes_neg_ck = xp.empty((cfg._total_clauses, cfg.n_classes), dtype=np.float32)
            self.y_hat_neg_ck = xp.empty((cfg._total_clauses, cfg.n_classes), dtype=np.float32)
            self.loss_neg_ck = xp.empty(cfg._total_clauses, dtype=np.float32)
