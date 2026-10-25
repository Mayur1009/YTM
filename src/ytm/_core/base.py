import abc
from dataclasses import fields

import numpy as np

from .backends.base import BaseDevice
from .config import BaseTMConfig
from .device_config import DeviceConfig
from .info import ClauseInfo


class BaseTM(abc.ABC):
    def __init__(self, config: BaseTMConfig, device_config: DeviceConfig):
        self.dev = self._build_device(config, device_config)

    @staticmethod
    @abc.abstractmethod
    def _config_cls() -> type[BaseTMConfig]: ...

    @property
    def _rng(self) -> np.random.Generator:
        return self.dev._rng

    @property
    def config(self) -> BaseTMConfig:
        """Model settings, see :class:`~ytm._core.config.BaseTMConfig`. Treat it as read-only: the kernels and derived values are built from it at construction."""
        return self.dev.config

    @property
    def device_config(self) -> DeviceConfig:
        """Device settings, see :class:`~ytm._core.device_config.DeviceConfig`. Use :meth:`to` to change the device."""
        return self.dev.device_config

    @staticmethod
    @abc.abstractmethod
    def _device_cls(kind: str) -> type[BaseDevice]: ...

    def _build_device(self, config: BaseTMConfig, device_config: DeviceConfig, state: dict | None = None) -> BaseDevice:
        return self._device_cls(device_config.kind)(config, device_config, state)

    def to(self, device: str, **device_kwargs) -> None:
        """Move the model to another device, in place.

        Parameters
        ----------
        device : str
            ``"cpu[:n_threads]"`` or ``"cuda[:gpu_id]"``.

        **device_kwargs
            Other :class:`~ytm._core.device_config.DeviceConfig` options, e.g. ``block_size``.
            Options not given go back to their defaults.
        """
        self.dev = self._build_device(self.config, DeviceConfig(device=device, **device_kwargs), self.dev.get_state_dict())

    def __getstate__(self) -> dict:
        """Config is saved as its constructor arguments, minus `ta_init`: its only use is at construction."""
        cfg = self.config
        return {
            "config": {f.name: getattr(cfg, f.name) for f in fields(cfg) if f.init and f.name != "ta_init"},
            "params": self.dev.get_state_dict(),
        }

    def __setstate__(self, state: dict) -> None:
        """Unpickling always lands on `cpu:1`; use `.to()` afterwards to move it."""
        cfg = self._config_cls()(**state["config"])
        self.dev = self._build_device(cfg, DeviceConfig(), state["params"])

    @abc.abstractmethod
    def _fit(self, X: np.ndarray, Y: np.ndarray, *args, **kwargs): ...

    def score(self, X: np.ndarray, force_repack: bool = False) -> np.ndarray:
        """Calculate class sums or model outputs for the given X.

        Parameters
        ----------
        X : array-like of int, shape (N, *dim) or (N, prod(dim))
            Inputs, each feature within ``[feat_mins, feat_maxs]``.

        force_repack : bool, default=False
            Force a clause repack.

        Returns
        -------
        ndarray of shape (N, n_classes), float32
            Output of each class.
        """
        return self.dev.infer(self.config.prepare_X(X), force_repack)

    @abc.abstractmethod
    def to_prob(self, class_sums: np.ndarray) -> np.ndarray:
        """Convert the class sums / raw model outputs to probabilities."""

    def transform(self, X: np.ndarray, force_repack: bool = False) -> np.ndarray:
        """Clause outputs per sample: 1 where the clause fires (on any patch), else 0.

        Parameters
        ----------
        X : array-like of int, shape (N, *dim) or (N, prod(dim))
            Inputs, each feature within ``[feat_mins, feat_maxs]``.

        force_repack : bool, default=False
            Force a clause repack.

        Returns
        -------
        ndarray of shape (N, n_clause_banks, n_clauses), int8
            Clause outputs. ``n_clause_banks`` is 1 when coalesced, else ``n_classes``.
        """
        return self.dev.transform(self.config.prepare_X(X), force_repack)

    def transform_patchwise(self, X: np.ndarray, force_repack: bool = False) -> np.ndarray:
        """Patchwise clause outputs per sample.

        Parameters
        ----------
        X : array-like of int, shape (N, *dim) or (N, prod(dim))
            Inputs, each feature within ``[feat_mins, feat_maxs]``.

        force_repack : bool, default=False
            Force a clause repack.

        Returns
        -------
        ndarray of shape (N, n_clause_banks, n_clauses, n_patches_y, n_patches_x), int8
            Clause outputs. ``n_clause_banks`` is 1 when coalesced, else ``n_classes``.
        """
        return self.dev.transform_patchwise(self.config.prepare_X(X), force_repack)

    def wic(self, class_id: int, polarity: int, pw_th: float = 0.0, force_repack: bool = False) -> np.ndarray:
        """Calculate the WIC score, obtained by taking weighted sum of a set of clauses. This does not
        depend on X, does not care about clause activations, and thus provides the **Global
        Interpretation** of the model.

        ``class_id`` and ``polarity`` define which set of clauses to use.

        .. note::
            This functions does not normalize the WIC, it simply returns the raw values. Use
            :func:`~ytm.utils.norm_asymmetric` on the scores to normalize them.

        Parameters
        ----------
        class_id : int
            Class whose clause weights are used.

        polarity : int
            ``+1`` for the clauses with positive weight for ``class_id``, ``-1`` for negative.

        pw_th : float, default=0.0
            Convolutional models only: skip patches whose patch weight, normalized per clause to
            ``[0, 1]``, is at or below this value.

        force_repack : bool, default=False
            Force a clause repack.

        Returns
        -------
        ndarray of shape (H, W, C), float32
            WIC score per input feature.
        """
        if self.config._n_patches > 1 and not self.config.track_patch_weights:
            raise ValueError("track_patch_weights=True is required for wic() on a convolutional model.")
        return self.dev.wic(class_id, polarity, pw_th, force_repack)

    def wac(self, X: np.ndarray, target_classes: np.ndarray, polarity: int, force_repack: bool = False) -> np.ndarray:
        """Calculate the WAC score, obtained by taking the weighted sum of activated clauses for each
        sample in ``X``. This provides the **Local Interpretation** of the model for a given input.

        .. note::
            This functions does not normalize the WAC, it simply returns the raw values. Use
            :func:`~ytm.utils.norm_asymmetric` on the scores to normalize them.

        Parameters
        ----------
        X : array-like of int, shape (N, *dim) or (N, prod(dim))
            Inputs, each feature within ``[feat_mins, feat_maxs]``.

        target_classes : array-like of int, shape (N,)
            Class whose clause weights are used, per sample.

        polarity : int
            ``+1`` for the clauses with positive weight for the target class, ``-1`` for negative.

        force_repack : bool, default=False
            Force a clause repack.

        Returns
        -------
        ndarray of shape (N, H, W, C), float32
            WAC score per sample and input feature.
        """
        return self.dev.wac(self.config.prepare_X(X), target_classes, polarity, force_repack)

    def set_threads(self, n: int) -> None:
        """Set the number of threads to use with OpenMP. Only applicable for CPU device."""
        self.dev.set_threads(max(1, n))

    # == getters ==
    def get_weights(self) -> np.ndarray:
        """Get weights.

        Returns
        -------
        ndarray of shape (n_classes, n_clauses), float32
        """
        return self.dev.get_weights()

    def get_ta_states(self) -> np.ndarray:
        """Get TA states.

        Returns
        -------
        ndarray of shape (n_clause_banks, n_clauses, n_literals)
        """
        return self.dev.get_ta_states()

    def get_patch_weights(self) -> np.ndarray:
        """Get patch weights.

        Returns
        -------
        ndarray of shape (n_clause_banks, n_clauses, n_patches_y, n_patches_x)
        """
        return self.dev.get_patch_weights()

    def get_literals(self) -> np.ndarray:
        """Get the clauses showing include and excludes only. 1 if the literal is included, 0 if excluded.

        Returns
        -------
        ndarray of shape (n_clause_banks, n_clauses, n_literals), uint8
        """
        return np.asarray(self.get_ta_states() >= self.config._include_state, dtype=np.uint8)

    def get_clauses(self, force_repack: bool = True) -> ClauseInfo:
        """Get the clauses as feature bounds, position bounds and weights.

        The result is cached in ``clauseinfo``.

        Parameters
        ----------
        force_repack : bool, default=True
            Build the clauses again from the current TA states. False returns the cached result
            when there is one.

        Returns
        -------
        ClauseInfo
            See :class:`~ytm._core.info.ClauseInfo`.
        """
        if hasattr(self, "clauseinfo") and not force_repack:
            return self.clauseinfo

        cfg = self.config

        # Pack clauses with `full=true`, so that we dont early skip any clauses.
        self.dev.pack_clauses(force_repack=True, full=True)
        packed = self.dev.get_packed_clauses()
        shape = (cfg._n_clause_banks, cfg._n_clauses)

        position_bounds = None
        if cfg._n_patches > 1:
            position_bounds = packed.clause_position_bounds.reshape(*shape, 4)

        # Convert from list based sparse feature bounds to dense per clause array.
        n_feat = cfg._n_raw_patch_feats
        feature_bounds = np.empty((cfg._total_clauses, n_feat, 2), dtype=np.int32)
        feature_bounds[:, :, 0] = 0  # Set default
        feature_bounds[:, :, 1] = cfg._therm_bits  # Set default

        fids = packed.clause_feat_ids.reshape(cfg._total_clauses, n_feat)  # feature ids.
        compact = packed.clause_feat_bounds.reshape(cfg._total_clauses, n_feat, 2)  # feature bounds for the fids.
        used = np.arange(n_feat)[None, :] < packed.clause_n_feats[:, None]  # mask to select packed.clause_n_feats entries
        ci, si = np.nonzero(used)
        feature_bounds[ci, fids[ci, si]] = compact[ci, si]

        # Shift back the feature bounds back, since the model is trained on shifted X.
        feature_bounds = feature_bounds.reshape(*shape, n_feat, 2) + cfg._feat_mins.reshape(-1, 1)

        # Cache and Return ClauseInfo
        self.clauseinfo = ClauseInfo(
            feature_bounds=feature_bounds.reshape(*shape, cfg._n_raw_patch_feats * 2),
            position_bounds=position_bounds,
            has_contra=packed.has_contra.reshape(*shape),
            clause_len=packed.clause_len.reshape(*shape),
            clause_weights=self.get_weights(),
            config=self.config,
        )
        return self.clauseinfo
