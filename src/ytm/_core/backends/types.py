from dataclasses import dataclass
from typing import Any


@dataclass
class PackedClauses:
    """Clauses compiled from TA states into per feature intervals, rebuilt when out of sync."""

    clause_feat_bounds: Any  # (total_clauses, n_raw_patch_feats, 2) closed [lower, upper]
    clause_position_bounds: Any  # (total_clauses, 4) closed [min_y, max_y, min_x, max_x]
    bounded_feat_ids: Any  # (total_clauses, n_raw_patch_feats) features that constrain anything
    n_bounded_feats: Any  # (total_clauses,) how many of the above are in use
    clause_density: Any  # (total_clauses,) included literals, -1 marks a contradiction
    is_clause_synced: Any  # (total_clauses,) 0 when the clause needs repacking
