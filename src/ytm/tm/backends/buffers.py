from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from ..._core.backends.buffers import FitBuffers


@dataclass(kw_only=True)
class TMFitBuffers(FitBuffers):
    """Fit buffers of the TM. Adds to :class:`~ytm._core.backends.buffers.FitBuffers`:

    Attributes
    ----------
    feedback_type : array of shape (n_clauses, n_classes), uint8
        Feedback of each clause per class on the current sample, one of ``FB_NONE``, ``FB_T1A``,
        ``FB_T1B``, ``FB_T2``.

    prob : array of shape (n_classes,)
        Signed update probability of each class for the current sample.

    label_probs : array of shape (chunk, n_classes)
        Chance that each class gives feedback at all, per sample. A per-sample field.
    """

    feedback_type: Any = field(init=False)  # (n_clauses, n_classes) one of FB_NONE/T1A/T1B/T2, indexed [rel_clause, class]
    prob: Any = field(init=False)  # (n_classes,) signed update probability for the current sample
    label_probs: Any = field(init=False, default=None)  # (n_samples, n_classes) chance a class gives feedback at all

    per_sample_fields: ClassVar[tuple[str, ...]] = (*FitBuffers.per_sample_fields, "label_probs")

    def __post_init__(self) -> None:
        super().__post_init__()
        cfg, xp = self.cfg, self.xp
        self.feedback_type = xp.zeros((cfg._n_clauses, cfg.n_classes), dtype=np.uint8)
        self.prob = xp.zeros(cfg.n_classes, dtype=np.float32)
