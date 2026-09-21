import abc

import numpy as np

from .backends.base import BaseDevice
from .config import BaseTMConfig
from .device_config import DeviceConfig
from .utils import ClauseInfo


class BaseTM(abc.ABC):
    cpu_device_cls: type[BaseDevice]

    def __init__(self, dev: BaseDevice):
        self.dev = dev
        self._rng = np.random.default_rng(self.config.seed)

    @property
    def config(self) -> BaseTMConfig:
        return self.dev.config

    @property
    def device_config(self) -> DeviceConfig:
        return self.dev.device_config

    @abc.abstractmethod
    def to(self, device: str, **device_kwargs) -> None:
        """Move the model to another device, in place."""
        ...

    def __getstate__(self) -> dict:
        return {"config": self.config, "params": self.dev.get_state_dict(), "rng": self._rng}

    def __setstate__(self, state: dict) -> None:
        """Unpickling always lands on `cpu:1`; use `.to()` afterwards to move it."""
        self.dev = self.cpu_device_cls(state["config"], DeviceConfig())
        self.dev.load_state_dict(state["params"])
        self._rng = state["rng"]

    @abc.abstractmethod
    def _fit(self, X: np.ndarray, Y: np.ndarray, *args, **kwargs): ...

    def _prepare_X(self, X: np.ndarray) -> np.ndarray:
        cfg = self.config

        # number of features match
        assert np.prod(X.shape[1:]) == np.prod(cfg._dim), f"Expected input features to match dim {cfg._dim}, but got {X.shape[1:]}"

        # Check if values are in provided bounds
        if cfg._patch_is_image:
            lo, hi, axes = cfg._feat_mins.reshape(cfg._dim), cfg._feat_maxs.reshape(cfg._dim), 0
        else:
            lo, hi, axes = cfg._feat_mins[: cfg._dim[2]], cfg._feat_maxs[: cfg._dim[2]], (0, 1, 2)
        Xv = np.asarray(X).reshape(X.shape[0], *cfg._dim)
        assert np.all(Xv.min(axis=axes) >= lo), f"X has values below feat_mins, min is {int(Xv.min())}"
        assert np.all(Xv.max(axis=axes) <= hi), f"X has values above feat_maxs, max is {int(Xv.max())}"

        # Shift X, so that the model always sees X in [0, therm_bits]
        return np.asarray((Xv - lo) if np.any(lo) else Xv, dtype=cfg._fbound_dtype, order="C")

    def score(self, X: np.ndarray, batch_size: int = -1, force_repack: bool = False) -> np.ndarray:
        return self.dev.calc_class_sums(self._prepare_X(X), batch_size, force_repack)

    def transform(self, X: np.ndarray, batch_size: int = -1, force_repack: bool = False) -> np.ndarray:
        return self.dev.transform(self._prepare_X(X), batch_size, force_repack)

    def transform_patchwise(self, X: np.ndarray, batch_size: int = -1, force_repack: bool = False) -> np.ndarray:
        return self.dev.transform_patchwise(self._prepare_X(X), batch_size, force_repack)

    def wic(self, class_id: int, polarity: int, pw_th: float = 0.0, force_repack: bool = False) -> np.ndarray:
        if self.config._n_patches > 1 and not self.config.track_patch_weights:
            raise ValueError("track_patch_weights=True is required for wic() on a convolutional model.")
        return self.dev.wic(class_id, polarity, pw_th, force_repack)

    def wac(self, X: np.ndarray, target_classes: np.ndarray, polarity: int, batch_size: int = -1, force_repack: bool = False) -> np.ndarray:
        return self.dev.wac(self._prepare_X(X), target_classes, polarity, batch_size, force_repack)

    def set_threads(self, n: int) -> None:
        self.dev.set_threads(max(1, n))

    # == getters ==
    def get_weights(self) -> np.ndarray:
        return self.dev.get_weights()

    def get_ta_states(self) -> np.ndarray:
        return self.dev.get_ta_states()

    def get_patch_weights(self) -> np.ndarray:
        return self.dev.get_patch_weights()

    def get_literals(self) -> np.ndarray:
        return np.asarray(self.get_ta_states() >= self.config._include_state, dtype=np.uint8)

    def get_clauses(self, force_repack: bool = True, full: bool = True) -> ClauseInfo:
        cfg = self.config
        self.dev.pack_clauses(force_repack, full)
        packed = self.dev.get_packed_clauses()
        shape = (cfg._n_clause_banks, cfg._n_clauses)

        position_bounds = None
        if cfg._n_patches > 1:
            position_bounds = packed.clause_position_bounds.reshape(*shape, 4)

        # Scatter the compacted bounds back to one entry per feature, unconstrained ones spanning
        # the full range, then undo the internal zero basing so the intervals are in user units.
        n_feat = cfg._n_raw_patch_feats
        feature_bounds = np.empty((cfg._total_clauses, n_feat, 2), dtype=np.int32)
        feature_bounds[:, :, 0] = 0
        feature_bounds[:, :, 1] = cfg._therm_bits

        fids = packed.clause_feat_ids.reshape(cfg._total_clauses, n_feat)
        compact = packed.clause_feat_bounds.reshape(cfg._total_clauses, n_feat, 2)
        used = np.arange(n_feat)[None, :] < packed.clause_n_feats[:, None]
        ci, si = np.nonzero(used)
        feature_bounds[ci, fids[ci, si]] = compact[ci, si]

        feature_bounds = feature_bounds.reshape(*shape, n_feat, 2) + cfg._feat_mins.reshape(-1, 1)

        return ClauseInfo(
            feature_bounds=feature_bounds.reshape(*shape, cfg._n_raw_patch_feats * 2),
            position_bounds=position_bounds,
            has_contra=packed.has_contra.reshape(*shape),
            clause_len=packed.clause_len.reshape(*shape),
        )
