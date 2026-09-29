import abc

import numpy as np

__all__ = ["ConstTAInit", "TAInit", "UniformTAInit"]


class TAInit(abc.ABC):
    @abc.abstractmethod
    def __call__(self, rng: np.random.Generator, shape: tuple[int, int], n_states: int, *args, **kwargs) -> np.ndarray: ...


class ConstTAInit(TAInit):
    def __init__(self, state: int):
        self.state_val = int(state)

    def __call__(self, rng: np.random.Generator, shape: tuple[int, int], n_states: int) -> np.ndarray:
        assert 0 <= self.state_val < n_states, f"state {self.state_val} out of bounds [0, {n_states})"
        return np.full(shape, self.state_val)


class UniformTAInit(TAInit):
    def __init__(self, low: int | None = None, high: int | None = None):
        self.low = low
        self.high = high

    def __call__(self, rng: np.random.Generator, shape: tuple[int, int], n_states: int) -> np.ndarray:
        lo = 0 if self.low is None else self.low
        hi = n_states if self.high is None else self.high
        assert 0 <= lo < hi <= n_states, f"need 0 <= low < high <= {n_states}, got low={lo}, high={hi}"
        return rng.integers(lo, hi, size=shape)
