from dataclasses import asdict
from typing import Literal, NamedTuple, Unpack

import numpy as np

from .args import T_args, TMArgs


class ClauseInfo(NamedTuple):
    """Packed clause representation returned by :meth:`BaseTM.get_clauses`.

    Attributes
    ----------
    feature_bounds : ndarray of shape (n_clause_banks, n_clauses, n_raw_patch_feats * 2)
        Closed ``[lower, upper]`` feature inclusion bounds per clause.
    position_bounds : ndarray of shape (n_clause_banks, n_clauses, 4) or None
        Closed ``[min_y, max_y, min_x, max_x]`` position bounds per clause.
        ``None`` when position literals are disabled and input has a single patch.
    clause_density : ndarray of shape (n_clause_banks, n_clauses), dtype int
        Number of included literals per clause. ``-1`` marks an invalid
        clause (contains a contradiction).
    """

    feature_bounds: np.ndarray
    position_bounds: np.ndarray | None
    clause_density: np.ndarray


class BaseTM:
    """Base class for all Tsetlin Machine models.

    Not meant for direct instantiation. Subclasses add task-specific
    ``fit`` and ``predict`` methods. All constructor parameters are shared
    across subclasses and need not be repeated in child class docstrings.

    Parameters
    ----------
    n_clauses : int
        Number of clauses per class.
    T : int or float or tuple of (int or float, int or float)
        Threshold controlling the learning signal magnitude. A scalar ``T``
        sets the symmetric range ``[-T, T]``. A tuple ``(T_min, T_max)``
        sets asymmetric bounds directly.
    s : float
        Specificity parameter. Must be >= 1.0.
        Higher values produce more specific (denser) clauses.
    dim : tuple of (int, int, int)
        Input shape as ``(H, W, C)``. For 1-D input use
        ``(n_features, 1, 1)``.
    n_classes : int
        Number of output classes.
    feat_mins : int or ndarray of shape (n_patch_feats,), default=0
        Minimum value per feature. Scalar broadcasts to all features.
    feat_maxs : int or ndarray of shape (n_patch_feats,), default=1
        Maximum value per feature. Scalar broadcasts to all features.
    patch_dim : tuple of (int, int) or None, default=None
        Patch height and width for convolution TM variants. ``None``
        treats the full input as a single patch (same as no convolution).
    stride : tuple of (int, int), default=(1, 1)
        Stride ``(row, col)`` for patch extraction.
    q : float, default=1.0
        Controls how often non-target classes receive negative feedback.
    weighted : bool, default=True
        Enable integer clause weights. When ``False`` all weights are
        clamped to ``{-1, +1}``.
    max_weight : float, default=float32 max
        Upper bound on absolute clause weight magnitude.
    coalesced : bool, default=True
        Share clauses across class. The model has only one clause bank,
        which is used by all the classes.
    negated_literals : bool, default=True
        Include negated feature literals in the clause.
    position_literals : bool, default=True
        Include patch position literals.
    negative_clauses : bool, default=True
        Allow negative polarity clauses.
    allow_polarity_change : bool, default=True
        Allow clause polarity to flip during training.
    max_includes : int, default=-1
        Literal budget for a clause. ``-1`` means
        unlimited.
    n_states : int, default=256
        Number of TA states. Each TA ranges over ``[0, n_states-1]``.
    include_state : int, default=-1
        TA state threshold above which a literal is considered included.
        ``-1`` defaults to ``n_states // 2``.
    skip_t1a_fb : bool, default=False
        Disable Type Ia feedback.
    skip_t1b_fb : bool, default=False
        Disable Type Ib feedback.
    skip_t2_fb : bool, default=False
        Disable Type II feedback.
    track_patch_weights : bool, default=True
        Accumulate per-patch counts, used for global interpretability.
    boost_tp_fb : bool, default=True
        Make TIa feedback literal increments non-stocastic (prob=1),
        instead of prob=1/s.
    seed : int or None, default=None
        Random seed. ``None`` or negative values draw a random seed.
        ``0`` is not allowed.
    device : {"cpu", "cuda"}, default="cpu"
        Compute device. ``"cuda"`` requires ``cupy`` to be installed.
    n_threads : int, default=1
        Number of CPU threads (CPU device only). Clamped to >= 1.
        Needs ``openmp``.
    compile_flags : list of str or None, default=None
        Extra compiler flags.
    grid_size : int or None, default=None
        CUDA grid size override.
    block_size : int, default=256
        CUDA block size.
    """

    def __init__(
        self,
        n_clauses: int,
        T: float | tuple[float, float],
        s: float,
        dim: tuple[int, int, int],
        n_classes: int,
        **opt_args: Unpack[T_args],
    ):
        self.args = TMArgs(n_clauses, T, s, dim, n_classes, **opt_args)
        self.np_rng = np.random.default_rng(self.args.seed)
        self.rng_state = self.np_rng.integers(1, 1 << 63, dtype=np.uint64)

        if self.args.device == "cpu":
            from .backends.cpu.cpu_backend import CPUDevice

            self.dev = CPUDevice(self.args)
        elif self.args.device == "cuda":
            from .backends.cuda.cuda_backend import CUDADevice

            self.dev = CUDADevice(self.args)
        else:
            raise ValueError(f"Unsupported device: {self.args.device}")

    def _fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        shuffle: bool = True,
        clause_drop_p: float = 0.0,
        batch_size: int = -1,
        label_sampling: bool = False,
    ) -> None:
        assert np.prod(X.shape[1:]) == np.prod(self.args.dim), (
            f"Expected input features to match dim {self.args.dim}, but got {X.shape[1:]}"
        )
        X = np.ascontiguousarray(X)

        N = X.shape[0]
        iota = np.arange(N)
        if shuffle:
            self.np_rng.shuffle(iota)
        X = X[iota]
        Y = Y[iota]

        encoded_Y = self._encode_Y(Y)
        label_probs = self._label_sampler(encoded_Y, label_sampling)

        self.dev.fit_epoch(X, encoded_Y, clause_drop_p, batch_size, label_probs, self.rng_state)
        self.rng_state = self.np_rng.integers(1, 1 << 63, dtype=np.uint64)

    def score(self, X: np.ndarray, batch_size: int = -1, clip_class_sums: bool = False):
        """Return raw class-sum votes for each sample.

        Parameters
        ----------
        X : ndarray of shape (N, ...)
            Input samples.
        batch_size : int, default=-1
            Number of samples processed per batch. ``-1`` processes all
            samples at once.
        clip_class_sums : bool, default=False
            Clip output to ``[T_min, T_max]`` before returning.

        Returns
        -------
        ndarray of shape (N, n_classes)
            Vote sums per class.
        """
        class_sums = self.dev.infer(np.ascontiguousarray(X), batch_size)
        if clip_class_sums:
            class_sums = np.clip(class_sums, self.args.T_min, self.args.T_max)
        return class_sums

    def transform(self, X: np.ndarray, batch_size: int = -1) -> np.ndarray:
        clause_outputs = self.dev.transform(np.ascontiguousarray(X), batch_size)
        return clause_outputs

    def transform_patchwise(self, X: np.ndarray, batch_size: int = -1) -> np.ndarray:
        """Return per-patch clause outputs for each sample.

        Parameters
        ----------
        X : ndarray of shape (N, ...)
            Input samples.
        batch_size : int, default=-1
            Number of samples processed per batch. ``-1`` processes all
            samples at once.

        Returns
        -------
        ndarray of shape (N, n_patches, n_clause_banks, n_clauses)
            Clause activation (0 or 1) per patch per sample.
        """
        patch_outputs = self.dev.transform_patchwise(np.ascontiguousarray(X), batch_size)
        return patch_outputs

    def _encode_Y(self, Y: np.ndarray) -> np.ndarray:
        encoded_Y = np.copy(Y).astype(np.float32) * self.args.T_max
        encoded_Y[encoded_Y == 0] = self.args.T_min
        return encoded_Y

    def _label_sampler(self, encoded_Y: np.ndarray, label_sampling: bool) -> np.ndarray:
        return np.ones_like(encoded_Y, dtype=np.float32)

    def freeze_clauses(self, class_id: int, clause_ids: list[int] | np.ndarray):
        """Freeze selected clauses.

        Parameters
        ----------
        class_id : int
            Class bank index to target. Ignored when ``coalesced=True``.
        clause_ids : list of int or ndarray of int
            Indices of clauses to freeze, in ``[0, n_clauses)``.
        """
        if self.args.coalesced:
            if class_id != 0:
                print(f"Warning: coalesced is true, ignoring class_id {class_id} and freezing clauses for all classes")
            self.dev.freeze_clauses(0, clause_ids)
        else:
            assert class_id < self.args.n_classes, f"Invalid class_id {class_id} for n_classes {self.args.n_classes}"
            self.dev.freeze_clauses(class_id, clause_ids)

    def unfreeze_clauses(self):
        """Unfreeze all clauses, re-enabling feedback for the full clause set."""
        self.dev.unfreeze_clauses()

    def get_weights(self) -> np.ndarray:
        """Return clause weights.

        Returns
        -------
        ndarray of shape (n_clause_banks, n_clauses)
            Integer clause weight per class bank.
        """
        return self.dev.get_weights()

    def get_ta_states(self) -> np.ndarray:
        """Return TA state values.

        Returns
        -------
        ndarray of shape (n_clause_banks, n_clauses, n_literals)
            TA state in ``[0, n_states - 1]`` per literal per clause.
        """
        return self.dev.get_ta_states()

    def get_literals(self) -> np.ndarray:
        """Return clause literals.

        Returns
        -------
        ndarray of shape (n_clause_banks, n_clauses, n_literals), dtype uint8
            ``1`` if literal is included, ``0`` otherwise.
        """
        return np.asarray(self.get_ta_states() >= self.args.include_state, dtype=np.uint8)

    def get_patch_weights(self) -> np.ndarray:
        """Return patch weights.

        Requires ``track_patch_weights=True`` (default). Required for global
        interpretability of convolution TM.

        Returns
        -------
        ndarray of shape (n_clauses, n_patches)
            Patch weights per clause.
        """
        return self.dev.get_patch_weights()

    def get_clauses(self, force_repack=False) -> ClauseInfo:
        """Return clauses, converted from thermometer format to [min, max] bounds.

        Parameters
        ----------
        force_repack : bool, default=False
            Force recompute the packed representation.

        Returns
        -------
        ClauseInfo
            Named tuple with ``feature_bounds``, ``position_bounds``, and
            ``clause_density``. See :class:`ClauseInfo` for field shapes.
        """
        self.dev.pack_clauses(force_repack)
        buf = self.dev.packed_clauses.get()

        clause_feat_bounds = buf.clause_feat_bounds.reshape((self.dev.n_clause_banks, self.args.n_clauses, self.dev.n_raw_patch_feats * 2))

        position_bounds = None
        if self.args.position_literals or self.dev.n_patches > 1:
            position_bounds = buf.clause_position_bounds.reshape((self.dev.n_clause_banks, self.args.n_clauses, 4))

        clause_density = buf.clause_density.reshape((self.dev.n_clause_banks, self.args.n_clauses))

        return ClauseInfo(
            feature_bounds=clause_feat_bounds,
            position_bounds=position_bounds,
            clause_density=clause_density,
        )

    def to(self, device: Literal["cpu", "cuda"]):
        """Move the model to a different compute device.

        Parameters
        ----------
        device : {"cpu", "cuda"}
            Target device. ``"cuda"`` requires ``cupy``.
        """
        orig_dev_state = self.dev.get_state_dict()
        if device == "cpu":
            from .backends.cpu.cpu_backend import CPUDevice

            self.dev = CPUDevice(self.args)
        elif device == "cuda":
            from .backends.cuda.cuda_backend import CUDADevice

            self.dev = CUDADevice(self.args)
        else:
            raise ValueError(f"Unsupported device: {device}")

        self.dev.load_state_dict(orig_dev_state)
        self.args.device = device

    def set_threads(self, n: int) -> None:
        """Set the number of CPU threads.

        Parameters
        ----------
        n : int
            Number of threads. Clamped to >= 1. No effect on CUDA device.
        """
        self.args.n_threads = max(1, n)
        self.dev.set_threads(self.args.n_threads)

    def get_state_dict(self) -> dict:
        """Return a serializable snapshot of the model.

        The returned dict contains ``args``, ``params``, ``rng_state``, and
        ``np_rng_state``. Pass it to :meth:`load_state_dict` or
        :meth:`from_state_dict` to restore the model.

        Returns
        -------
        dict
            Complete model suitable for ``pickle``.
        """
        return {
            "args": asdict(self.args),
            "params": self.dev.get_state_dict(),
            "rng_state": self.rng_state,
            "np_rng_state": self.np_rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore model state from a dict produced by :meth:`get_state_dict`.

        Resets the model in-place. Always restores to CPU regardless of the
        device stored in ``state``. Call :meth:`to` afterwards if needed.

        Parameters
        ----------
        state : dict
            State dict as returned by :meth:`get_state_dict`.
        """
        state["args"]["device"] = "cpu"
        BaseTM.__init__(self, **state["args"])
        self.dev.load_state_dict(state["params"])

        self.rng_state = state.get("rng_state", self.rng_state)
        self.np_rng.bit_generator.state = state.get("np_rng_state", self.np_rng.bit_generator.state)

    @classmethod
    def from_state_dict(cls, state: dict) -> "BaseTM":
        """Construct a model instance from a state dict without calling ``__init__``.

        Parameters
        ----------
        state : dict
            State dict as returned by :meth:`get_state_dict`.

        Returns
        -------
        BaseTM
            Restored model instance of the calling subclass type.
        """
        instance = cls.__new__(cls)
        instance.load_state_dict(state)
        return instance

    def __getstate__(self):
        return self.get_state_dict()

    def __setstate__(self, state):
        self.load_state_dict(state)
