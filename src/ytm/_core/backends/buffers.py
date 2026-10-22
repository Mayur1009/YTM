from ctypes import POINTER
from dataclasses import InitVar, dataclass, field
from typing import Any

import numpy as np


@dataclass(kw_only=True)
class FitBuffers:
    make_ptrs: bool = True
    xp: Any
    cfg: InitVar[Any]
    X: Any  # (n_samples, *dim) the epoch's inputs, cast for the kernels
    Y: Any  # (n_samples, n_classes) targets
    clause_drop_mask: Any  # (total_clauses,) one mask for the whole epoch, per Drop Clause
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

    def __post_init__(self, cfg) -> None:
        xp = self.xp
        n = cfg._total_clauses
        self.X = xp.ascontiguousarray(self.X, dtype=cfg._fbound_dtype)
        self.Y = xp.ascontiguousarray(self.Y, dtype=np.float32)
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

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        if not self.make_ptrs:
            return
        if isinstance(value, np.ndarray):
            super().__setattr__(f"p_{name}", value.ctypes.data_as(POINTER(np.ctypeslib.as_ctypes_type(value.dtype))))
        elif value is None:
            super().__setattr__(f"p_{name}", None)


@dataclass
class PackedClauses:
    clause_feat_ids: Any  # (total_clauses, n_raw_patch_feats) the features this clause constrains
    clause_feat_bounds: Any  # (total_clauses, n_raw_patch_feats, 2) closed [lower, upper], parallel to the ids
    clause_n_feats: Any  # (total_clauses,) how many entries of the two above are in use
    clause_position_bounds: Any  # (total_clauses, 4) closed [min_y, max_y, min_x, max_x], (1, 1) with one patch
    has_contra: Any  # (total_clauses,) 1 when the clause is unsatisfiable
    clause_len: Any  # (total_clauses,) included literals, 0 means the clause is vacuous
    is_clause_synced: Any  # (total_clauses,) 0 when the clause needs repacking
