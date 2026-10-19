from dataclasses import dataclass
from typing import Any

from ..._core.backends.buffers import FitBuffers


@dataclass(kw_only=True)
class GuidedFitBuffers(FitBuffers):
    grad: Any  # (n_classes,) loss gradient w.r.t. votes for the current sample
    y_hat: Any  # (n_classes,) activated votes for the current sample
    loss: Any  # (1,) loss for the current sample
    lr: float  # resolved once per epoch: the call's override, or config.lr
    lambda_: float  # resolved once per epoch: the call's override, or config.lambda_
    votes_neg_ck: Any = None  # (total_clauses, n_classes) counterfactual votes, delta_l only
    y_hat_neg_ck: Any = None  # (total_clauses, n_classes) counterfactual activations, delta_l only
    loss_neg_ck: Any = None  # (total_clauses,) counterfactual loss, delta_l only
