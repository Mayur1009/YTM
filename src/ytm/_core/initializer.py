import abc

import numpy as np

from .utils import prepare_X

__all__ = ["ConstTAInit", "SampleTAInit", "TAInit", "UniformTAInit"]


class TAInit(abc.ABC):
    @abc.abstractmethod
    def __call__(self, rng: np.random.Generator, shape: tuple[int, int], n_states: int, **kwargs) -> np.ndarray: ...


class ConstTAInit(TAInit):
    def __init__(self, state: int):
        self.state_val = int(state)

    def __call__(self, rng: np.random.Generator, shape: tuple[int, int], n_states: int, **kwargs) -> np.ndarray:
        assert 0 <= self.state_val < n_states, f"state {self.state_val} out of bounds [0, {n_states})"
        return np.full(shape, self.state_val)


class UniformTAInit(TAInit):
    def __init__(self, low: int | None = None, high: int | None = None):
        self.low = low
        self.high = high

    def __call__(self, rng: np.random.Generator, shape: tuple[int, int], n_states: int, **kwargs) -> np.ndarray:
        lo = 0 if self.low is None else self.low
        hi = n_states if self.high is None else self.high
        assert 0 <= lo < hi <= n_states, f"need 0 <= low < high <= {n_states}, got low={lo}, high={hi}"
        return rng.integers(lo, hi, size=shape)


class SampleTAInit(TAInit):
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

    def _hot_Y(self, n_classes: int) -> np.ndarray:
        # should not be None if reaching here.
        assert self.Y is not None
        Y = self.Y
        if Y.ndim == 1:
            return (Y > 0)[:, None] if n_classes == 1 else Y[:, None] == np.arange(n_classes)
        Y = Y.reshape(len(Y), -1)
        assert Y.shape[1] == n_classes, f"Y must have n_classes ({n_classes}) columns, got {Y.shape[1]}"
        return Y > 0

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

        if self.include_position and cfg.position_literals and cfg._n_patches > 1:
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

    def _to_states(self, rng: np.random.Generator, cfg, n_states: int, clauses: np.ndarray) -> np.ndarray:
        inc, m = cfg._include_state, self.margin
        lo = np.where(clauses, inc, max(0, inc - 1 - m))
        hi = np.where(clauses, min(n_states, inc + m + 1), inc)
        return rng.integers(lo, hi)

    def __call__(
        self,
        rng: np.random.Generator,
        shape: tuple[int, int],
        n_states: int,
        cfg=None,
        weights=None,
        **kwargs,
    ):
        assert cfg is not None, "SampleTAInit needs the model config"
        assert shape == (cfg._total_clauses, cfg._n_literals), f"shape {shape} does not match the config"
        X = prepare_X(cfg, self.X)
        Y = None if self.Y is None else self._hot_Y(cfg.n_classes)

        cls = self._pick_class(rng, cfg)
        if weights is None:
            pos = np.ones(len(cls), dtype=bool)
        else:
            pos = weights[cls, np.arange(len(cls)) % cfg._n_clauses] > 0

        idx = self._pick_sample(rng, Y, cls, pos)
        patches, py, px = self._pick_patch(rng, cfg, X, idx)
        clauses = self._make_clauses(rng, cfg, patches, py, px)
        return self._to_states(rng, cfg, n_states, clauses)
