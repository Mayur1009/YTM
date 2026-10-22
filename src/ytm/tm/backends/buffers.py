from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from ..._core.backends.buffers import FitBuffers


@dataclass(kw_only=True)
class DiscreteFitBuffers(FitBuffers):
    feedback_type: Any = field(init=False)  # (n_clauses, n_classes) one of FB_NONE/T1A/T1B/T2, indexed [rel_clause, class]
    prob: Any = field(init=False)  # (n_classes,) signed update probability for the current sample
    label_probs: Any = field(init=False, default=None)  # (n_samples, n_classes) chance a class gives feedback at all

    per_sample_fields: ClassVar[tuple[str, ...]] = ("X", "Y", "rng_keys", "label_probs")

    def __post_init__(self) -> None:
        super().__post_init__()
        cfg, xp = self.cfg, self.xp
        self.feedback_type = xp.zeros((cfg._n_clauses, cfg.n_classes), dtype=np.uint8)
        self.prob = xp.zeros(cfg.n_classes, dtype=np.float32)
