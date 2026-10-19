from dataclasses import dataclass
from typing import Any


@dataclass(kw_only=True)
class FitBuffers:
    X: Any  # (n_samples, *dim) the epoch's inputs, cast for the kernels
    Y: Any  # (n_samples, n_classes) targets
    clause_drop_mask: Any  # (total_clauses,) one mask for the whole epoch, per Drop Clause
    clause_output: Any  # (total_clauses,) 1 when the clause fired on this sample
    selected_pids: Any  # (total_clauses,) patch each clause matched on, only valid where it fired
    votes: Any  # (n_classes,) weighted vote sum for the current sample
    t1b_n: Any  # (1,) uint32, valid entries of the T1b list
    t1b_cid: Any  # (total_clauses,) uint32, clauses with T1b on the current sample
    t1b_cnt: Any  # (total_clauses,) T1b count over classes, parallel to t1b_cid
    t1a_t2_n: Any  # (1,) uint32, valid entries of the T1a/T2 list
    t1a_t2_cid: Any  # (total_clauses,) uint32, clauses with T1a or T2 on the current sample
    t1a_cnt: Any  # (total_clauses,) T1a count over classes
    t2_cnt: Any  # (total_clauses,) T2 count over classes


@dataclass
class PackedClauses:
    clause_feat_ids: Any  # (total_clauses, n_raw_patch_feats) the features this clause constrains
    clause_feat_bounds: Any  # (total_clauses, n_raw_patch_feats, 2) closed [lower, upper], parallel to the ids
    clause_n_feats: Any  # (total_clauses,) how many entries of the two above are in use
    clause_position_bounds: Any  # (total_clauses, 4) closed [min_y, max_y, min_x, max_x], (1, 1) with one patch
    has_contra: Any  # (total_clauses,) 1 when the clause is unsatisfiable
    clause_len: Any  # (total_clauses,) included literals, 0 means the clause is vacuous
    is_clause_synced: Any  # (total_clauses,) 0 when the clause needs repacking
