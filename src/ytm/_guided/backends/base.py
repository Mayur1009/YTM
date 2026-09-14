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

        if cfg.weight_init == "random":
            mag = self._rng.uniform(0.0, 1.0, size=shape)
        elif isinstance(cfg.weight_init, str):
            mag = self._rng.uniform(0.0, float(cfg.weight_init[len("random:") :]), size=shape)
        else:
            mag = float(cfg.weight_init)

        self.clause_weights = self.xp.asarray(sign * mag, dtype=np.float32)
