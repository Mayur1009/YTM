import numpy as np

from ..._core.backends.base import BaseDevice as CoreBaseDevice
from ..config import TMConfig


class BaseDevice(CoreBaseDevice):
    config: TMConfig

    def _init_weights(self):
        cfg = self.config
        shape = (cfg.n_classes, cfg._n_clauses)
        sign = np.ones(shape, dtype=np.float32)

        if cfg.negative_clauses:
            n_neg = cfg._n_clauses // 2
            if cfg.coalesced:
                pol = np.ones(cfg._n_clauses, dtype=np.float32)
                pol[n_neg:] = -1.0
                for i in range(cfg.n_classes):
                    sign[i, :] = self._rng.permutation(pol)
            else:
                sign[:, n_neg:] = -1.0

        self.clause_weights = self.xp.asarray(sign * cfg.weight_init, dtype=np.float32)
