from ctypes import POINTER
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from ..config import BaseTMConfig


@dataclass(kw_only=True)
class FitBuffers:
    """Arrays the fit kernels read and write, kept on the device for a whole epoch.

    Work arrays (clause outputs, votes, feedback lists) are made once in ``__post_init__``.
    ``start_epoch`` takes the epoch's inputs: per-sample fields (``per_sample_fields``) stay on
    the host, the rest are moved to the device. ``load_chunk(start, end)`` moves one slice of
    the per-sample fields to the device, so an epoch can run in chunks that fit in memory.
    ``end_epoch`` drops the epoch's inputs.

    With ``make_ptrs``, every numpy array set as an attribute also gets a ctypes pointer
    ``p_<name>`` for the cpu kernels. It is None when the array is None.

    Parameters
    ----------
    xp : module
        Array module, ``numpy`` or ``cupy``.

    cfg : BaseTMConfig
        Model config, for sizes and dtypes.

    make_ptrs : bool, default=True
        Make the ``p_<name>`` pointers. Only the cpu backend needs them.

    Attributes
    ----------
    epoch_fields : dict
        The keyword arguments of the last ``start_epoch``, on the host.

    per_sample_fields : tuple of str
        Fields with one row per sample, needed for loading chunk by chunk.

    N : int
        Total number of samples in the epoch.

    chunk_start, chunk_end : int
        Sample range ``[chunk_start, chunk_end)`` currently loaded.

    X : array of shape ``(chunk, *dim)``
        Inputs of the loaded chunk, cast for the kernels.

    Y : array of shape (chunk, n_classes)
        Targets of the loaded chunk.

    rng_keys : array of shape (chunk,)
        RNG key of each sample in the loaded chunk.

    clause_drop_mask : array of shape (total_clauses,)
        Clauses dropped for the whole epoch (Drop Clause).

    lit_drop_mask : array of shape (n_literals,)
        Literals to skip.

    class_active : array of shape (n_classes,), uint8
        1 when the class gives feedback on the current sample.

    clause_output : array of shape (total_clauses,)
        1 when the clause fired on the current sample.

    selected_pids : array of shape (total_clauses,)
        Selected patch ids for the active clauses. Shape (1,) with one patch.

    votes : array of shape (n_classes,)
        Weighted vote sum for the current sample.

    t1b_n, t1a_t2_n : array of shape (1,), uint32
        Number of valid entries in the T1b list and in the T1a/T2 list.

    t1b_cid, t1a_t2_cid : array of shape (total_clauses,), uint32
        Clauses with T1b feedback, and clauses with T1a or T2 feedback, on the current sample.

    t1b_cnt : array of shape (total_clauses,)
        T1b count over classes, parallel to ``t1b_cid``.

    t1a_cnt, t2_cnt : array of shape (total_clauses,)
        T1a and T2 counts over classes, parallel to ``t1a_t2_cid``.
    """

    xp: Any
    cfg: BaseTMConfig
    make_ptrs: bool = True

    epoch_fields: dict[str, Any] = field(init=False, default_factory=dict)
    per_sample_fields: ClassVar[tuple[str, ...]] = ("X", "Y", "rng_keys")

    N: int = field(init=False, default=0)
    chunk_start: int = field(init=False, default=0)
    chunk_end: int = field(init=False, default=0)

    X: Any = field(init=False, default=None)  # (n_samples, *dim) the epoch's inputs, cast for the kernels
    Y: Any = field(init=False, default=None)  # (n_samples, n_classes) targets
    clause_drop_mask: Any = field(init=False, default=None)  # (total_clauses,) one mask for the whole epoch, per Drop Clause
    rng_keys: Any = field(init=False, default=None)  # (n_samples,) rng key of each sample
    lit_drop_mask: Any = field(init=False)  # (n_literals,) mask to skip a feature
    class_active: Any = field(init=False)  # (n_classes,) uint8, 1 when the class gets feedback on the current sample
    clause_output: Any = field(init=False)  # (total_clauses,) 1 when the clause fired on this sample
    selected_pids: Any = field(init=False)  # (total_clauses,) patch each clause matched on, only valid where it fired
    votes: Any = field(init=False)  # (n_classes,) weighted vote sum for the current sample
    t1b_n: Any = field(init=False)  # (1,) uint32, valid entries of the T1b list
    t1b_cid: Any = field(init=False)  # (total_clauses,) uint32, clauses with T1b on the current sample
    t1b_cnt: Any = field(init=False)  # (total_clauses,) T1b count over classes, parallel to t1b_cid
    t1a_t2_n: Any = field(init=False)  # (1,) uint32, valid entries of the T1a/T2 list
    t1a_t2_cid: Any = field(init=False)  # (total_clauses,) uint32, clauses with T1a or T2 on the current sample
    t1a_cnt: Any = field(init=False)  # (total_clauses,) T1a count over classes
    t2_cnt: Any = field(init=False)  # (total_clauses,) T2 count over classes

    def __post_init__(self) -> None:
        cfg, xp = self.cfg, self.xp
        n = cfg._total_clauses
        self.lit_drop_mask = xp.zeros(cfg._n_literals, dtype=np.uint8)
        self.class_active = xp.ones(cfg.n_classes, dtype=np.uint8)
        self.clause_output = xp.empty(n, dtype=np.int8)
        self.selected_pids = xp.empty(n if cfg._n_patches > 1 else 1, dtype=cfg._npatches_dtype)
        self.votes = xp.empty(cfg.n_classes, dtype=np.float32)
        self.t1b_n = xp.zeros(1, dtype=np.uint32)
        self.t1b_cid = xp.empty(n, dtype=np.uint32)
        self.t1b_cnt = xp.empty(n, dtype=cfg._nclasses_dtype)
        self.t1a_t2_n = xp.zeros(1, dtype=np.uint32)
        self.t1a_t2_cid = xp.empty(n, dtype=np.uint32)
        self.t1a_cnt = xp.empty(n, dtype=cfg._nclasses_dtype)
        self.t2_cnt = xp.empty(n, dtype=cfg._nclasses_dtype)

    def start_epoch(self, **fields: Any) -> None:
        """Take the epoch's inputs as keyword arguments, ``X`` included.

        Per-sample fields are kept on the host until ``load_chunk``. The others are set now, numpy
        arrays moved to the device.
        """
        assert "X" in fields, "start_epoch needs X"
        for name, value in fields.items():
            if name not in self.__dataclass_fields__:
                raise AttributeError(f"{type(self).__name__} has no field {name!r}")
            if name not in self.per_sample_fields:
                if isinstance(value, np.ndarray):
                    value = self.xp.asarray(value, order="C")
                setattr(self, name, value)

        self.epoch_fields = fields
        self.chunk_start = self.chunk_end = 0
        self.N = fields["X"].shape[0]

    def end_epoch(self) -> None:
        """Drop the epoch's inputs and reset the chunk range."""
        for name in self.epoch_fields:
            setattr(self, name, None)

        self.epoch_fields = {}
        self.chunk_start = self.chunk_end = self.N = 0

    def load_chunk(self, start: int, end: int) -> None:
        """Move samples ``[start, end)`` of every per-sample field to the device."""
        for name in self.per_sample_fields:
            setattr(self, name, self.xp.asarray(self.epoch_fields[name][start:end], order="C"))
        self.chunk_start, self.chunk_end = start, end

    def per_sample_bytes(self) -> int:
        return sum(self.epoch_fields[name][0].nbytes for name in self.per_sample_fields)

    if TYPE_CHECKING:
        from ctypes import _Pointer

        def __getattr__(self, name: str) -> _Pointer[Any] | None: ...

    def __setattr__(self, name: str, value: Any) -> None:
        """Set the attribute, and with ``make_ptrs`` also ``p_<name>``: a ctypes pointer for numpy arrays, None for None."""
        super().__setattr__(name, value)
        if not self.make_ptrs:
            return
        if isinstance(value, np.ndarray):
            super().__setattr__(f"p_{name}", value.ctypes.data_as(POINTER(np.ctypeslib.as_ctypes_type(value.dtype))))
        elif value is None:
            super().__setattr__(f"p_{name}", None)


@dataclass
class PackedClauses:
    """Clauses in compact form for evaluation.

    Attributes
    ----------
    clause_feat_ids : array of shape (total_clauses, n_raw_patch_feats)
        Features each clause constrains. Only the first ``clause_n_feats`` entries are used.

    clause_feat_bounds : array of shape (total_clauses, n_raw_patch_feats, 2)
        Closed range ``[lower, upper]`` allowed for each of those features, parallel to
        ``clause_feat_ids``.

    clause_n_feats : array of shape (total_clauses,)
        Number of used entries in the two arrays above.

    clause_position_bounds : array of shape (total_clauses, 4)
        Closed patch range ``[min_y, max_y, min_x, max_x]`` the clause can match. Shape (1, 1)
        with one patch.

    has_contra : array of shape (total_clauses,)
        1 when the clause includes a contradiction and can never fire.

    clause_len : array of shape (total_clauses,)
        Number of included literals. 0 means the clause is empty.

    is_clause_synced : array of shape (total_clauses,)
        0 when the clause changed since it was last packed and needs repacking.
    """

    clause_feat_ids: Any  # (total_clauses, n_raw_patch_feats) the features this clause constrains
    clause_feat_bounds: Any  # (total_clauses, n_raw_patch_feats, 2) closed [lower, upper], parallel to the ids
    clause_n_feats: Any  # (total_clauses,) how many entries of the two above are in use
    clause_position_bounds: Any  # (total_clauses, 4) closed [min_y, max_y, min_x, max_x], (1, 1) with one patch
    has_contra: Any  # (total_clauses,) 1 when the clause is unsatisfiable
    clause_len: Any  # (total_clauses,) included literals, 0 means the clause is vacuous
    is_clause_synced: Any  # (total_clauses,) 0 when the clause needs repacking
