from dataclasses import dataclass
from typing import Any

from ..._core.backends.buffers import FitBuffers


@dataclass(kw_only=True)
class DiscreteFitBuffers(FitBuffers):
    feedback_type: Any  # (n_clauses, n_classes) one of FB_NONE/T1A/T1B/T2, indexed [rel_clause, class]
    prob: Any  # (n_classes,) signed update probability for the current sample
    label_probs: Any  # (n_samples, n_classes) chance a class gives feedback at all
