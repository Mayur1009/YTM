"""A concrete `_core` CPU device, so each test file does not carry its own stub.

`_core.CPUDevice` leaves the fit steps abstract for the variants to fill in. Tests that only
exercise packing, evaluation or interpretation need a class that instantiates, not a trainer.
"""

import numpy as np
import pytest

from ytm._core.backends.cpu import CPUDevice
from ytm._core.config import BaseTMConfig
from ytm._core.device_config import DeviceConfig


class CoreDevice(CPUDevice):
    """Everything `_core` implements, with the per variant fit steps inert."""

    def fit_epoch(self, X, Y, clause_drop_p, batch_size): ...
    def fit_sample(self, rng_key, buf, e): ...
    def _fit_decide_fb(self, buf, e, rng_key): ...
    def _fit_apply_fb(self, buf, e, rng_key): ...
    def _fit_update_weights(self, buf): ...
    def _fit_update_bias(self, buf): ...


def make_device(device: str = "cpu:1", **kwargs) -> CoreDevice:
    """A device from config options, with the defaults small enough to enumerate."""
    cfg = {"n_clauses": 8, "s": 10.0, "dim": (4, 4, 1), "n_classes": 2, "feat_maxs": 3, "seed": 1}
    cfg.update(kwargs)
    return CoreDevice(BaseTMConfig(**cfg), DeviceConfig(device=device))


THREADS = 8  # every hot loop is `#pragma omp parallel for`, and the default device is single threaded


def thread_pair(factory=make_device, **kwargs) -> tuple[CoreDevice, CoreDevice]:
    """The same model compiled for one thread and for many.

    A race in an omp loop is invisible in normal use: the algorithm is stochastic, so a corrupted
    result looks like a different random draw. Identical seeds mean identical starting arrays, so
    any difference in the output is the parallelism and nothing else.
    """
    return factory(device="cpu:1", **kwargs), factory(device=f"cpu:{THREADS}", **kwargs)


def sprinkle_includes(dev: CoreDevice, rng: np.random.Generator, p: float = 0.05) -> None:
    """Sparse includes, so most clauses stay satisfiable.

    At the density `ta_init="random"` produces, essentially every clause contradicts itself and a
    test comparing two evaluators would agree trivially on "accepts nothing".
    """
    cfg = dev.config
    dev.ta_states[:] = cfg._include_state - 1
    dev.ta_states[rng.random(dev.ta_states.shape) < p] = cfg._include_state


@pytest.fixture
def device_factory():
    return make_device
