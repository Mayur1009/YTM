import abc
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .config import BaseTMConfig

__all__ = ["ConstTAInit", "SampleTAInit", "TAInit", "UniformTAInit"]


class TAInit(abc.ABC):
    """Base class for TA initializers, passed as ``ta_init`` to a model.

    Subclasses implement ``__call__``, which gets the model's RNG, its config and the initial
    clause weights of shape ``(n_classes, n_clauses)``, and returns the initial TA states of
    shape ``(total_clauses, n_literals)``.
    """

    @abc.abstractmethod
    def __call__(self, rng: np.random.Generator, cfg: "BaseTMConfig", weights: np.ndarray | None = None) -> np.ndarray: ...


class ConstTAInit(TAInit):
    """Set every TA to the same state.

    Parameters
    ----------
    state : int
        The state, in ``[0, n_states)``.
    """

    def __init__(self, state: int):
        self.state = int(state)

    def __call__(self, rng: np.random.Generator, cfg: "BaseTMConfig", weights: np.ndarray | None = None) -> np.ndarray:
        assert 0 <= self.state < cfg.n_states, f"state {self.state} out of bounds [0, {cfg.n_states})"
        return np.full((cfg._total_clauses, cfg._n_literals), self.state)


class UniformTAInit(TAInit):
    """Draw every TA state uniformly from ``[low, high)``.

    Parameters
    ----------
    low : int, default=None
        Lowest state. None means 0.

    high : int, default=None
        One past the highest state. None means ``n_states``.
    """

    def __init__(self, low: int | None = None, high: int | None = None):
        self.low = low
        self.high = high

    def __call__(self, rng: np.random.Generator, cfg: "BaseTMConfig", weights: np.ndarray | None = None) -> np.ndarray:
        lo = 0 if self.low is None else self.low
        hi = cfg.n_states if self.high is None else self.high
        assert 0 <= lo < hi <= cfg.n_states, f"need 0 <= low < high <= {cfg.n_states}, got low={lo}, high={hi}"
        return rng.integers(lo, hi, size=(cfg._total_clauses, cfg._n_literals))


class SampleTAInit(TAInit):
    """Start each clause as a copy of a training sample.

    Each clause picks a sample (and, with convolution, a random patch of it) and includes
    exactly the literals that are true for it. All other literals start just below include.

    Parameters
    ----------
    X : array-like of shape (N, ...)
        Samples to copy from, in the same format as ``fit`` takes.

    Y : array-like, default=None
        Labels for ``X``, in the same format as ``fit`` takes. Without ``Y``, samples are picked
        at random. With ``Y``, a clause with positive weight for a class copies a sample of that
        class, and a clause with negative weight copies a sample not of that class.

    margin : int, default=0
        Spread of the initial states. Included literals start uniformly in
        ``[include_state, include_state + margin]``, excluded ones in
        ``[include_state - 1 - margin, include_state - 1]``.

    p : float, default=0.0
        Probability, in ``[0, 1)``, of leaving out each copied literal.

    include_position : bool, default=False
        Also include the position literals of the copied patch.
    """

    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray | None = None,
        margin: int = 0,
        p: float = 0.0,
        include_position: bool = False,
    ):
        self.X = np.asarray(X)
        self.Y = None if Y is None else np.asarray(Y)
        if self.Y is not None:
            assert len(self.Y) == len(self.X), f"X has {len(self.X)} samples but Y has {len(self.Y)}"
        assert margin >= 0, f"margin must be >= 0, got {margin}"
        assert 0.0 <= p < 1.0, f"p must be in [0, 1), got {p}"
        self.margin = int(margin)
        self.p = float(p)
        self.include_position = include_position

    def _pick_class(self, rng: np.random.Generator, cfg) -> np.ndarray:
        """The class each clause is seeded for. Fixed by the bank without coalescing, random with it."""
        if cfg.coalesced:
            return rng.integers(0, cfg.n_classes, size=cfg._total_clauses)
        return np.arange(cfg._total_clauses) // cfg._n_clauses

    def _pick_sample(self, rng: np.random.Generator, Y: np.ndarray | None, cls: np.ndarray, pos: np.ndarray) -> np.ndarray:
        N = len(self.X)
        if Y is None:
            return rng.integers(0, N, size=len(cls))
        idx = np.empty(len(cls), dtype=np.intp)
        for c in range(Y.shape[1]):
            for p in (True, False):
                mask = (cls == c) & (pos == p)
                pool = np.flatnonzero(Y[:, c] == p)
                if len(pool) == 0:
                    pool = np.arange(N)
                idx[mask] = pool[rng.integers(0, len(pool), size=mask.sum())]
        return idx

    def _pick_patch(self, rng: np.random.Generator, cfg, X: np.ndarray, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(idx)
        if cfg._patch_is_image:
            zeros = np.zeros(n, dtype=np.intp)
            return X[idx].reshape(n, -1), zeros, zeros
        py = rng.integers(0, cfg._n_patches_y, size=n)
        px = rng.integers(0, cfg._n_patches_x, size=n)
        ys = py[:, None] * cfg._stride[0] + np.arange(cfg._patch_dim[0])
        xs = px[:, None] * cfg._stride[1] + np.arange(cfg._patch_dim[1])
        patches = X[idx[:, None, None], ys[:, :, None], xs[:, None, :]]
        return patches.reshape(n, -1), py, px

    def _make_clauses(self, rng: np.random.Generator, cfg, patches: np.ndarray, py: np.ndarray, px: np.ndarray) -> np.ndarray:
        n, half = len(patches), cfg._n_literals // 2
        clauses = np.zeros((n, cfg._n_literals), dtype=bool)

        v = patches.astype(np.intp)
        rows = np.broadcast_to(np.arange(n)[:, None], v.shape)
        base = cfg._n_position_feats + cfg._literal_offsets[: v.shape[1]].astype(np.intp)
        lo = v > 0
        clauses[rows[lo], (base + v - 1)[lo]] = True
        if cfg.negated_literals:
            hi = v < cfg._therm_bits
            clauses[rows[hi], (base + v + half)[hi]] = True

        if self.include_position and cfg._position_literals:
            r, xo = np.arange(n), cfg._n_patches_y - 1
            clauses[r[py > 0], (py - 1)[py > 0]] = True
            clauses[r[px > 0], (xo + px - 1)[px > 0]] = True
            if cfg.negated_literals:
                my, mx = py < cfg._n_patches_y - 1, px < cfg._n_patches_x - 1
                clauses[r[my], (py + half)[my]] = True
                clauses[r[mx], (xo + px + half)[mx]] = True

        if self.p > 0:
            clauses &= rng.random(clauses.shape) >= self.p
        return clauses

    def _to_states(self, rng: np.random.Generator, cfg, clauses: np.ndarray) -> np.ndarray:
        inc, m = cfg._include_state, self.margin
        lo = np.where(clauses, inc, max(0, inc - 1 - m))
        hi = np.where(clauses, min(cfg.n_states, inc + m + 1), inc)
        return rng.integers(lo, hi)

    def __call__(self, rng: np.random.Generator, cfg: "BaseTMConfig", weights: np.ndarray | None = None) -> np.ndarray:
        X = cfg.prepare_X(self.X)
        Y = None if self.Y is None else cfg.prepare_Y(self.Y).astype(bool)

        cls = self._pick_class(rng, cfg)
        if weights is None:
            pos = np.ones(len(cls), dtype=bool)
        else:
            pos = weights[cls, np.arange(len(cls)) % cfg._n_clauses] > 0

        idx = self._pick_sample(rng, Y, cls, pos)
        patches, py, px = self._pick_patch(rng, cfg, X, idx)
        clauses = self._make_clauses(rng, cfg, patches, py, px)
        return self._to_states(rng, cfg, clauses)
