from typing import Any
from dataclasses import dataclass
import abc
import numpy as np
from tqdm import tqdm
from ..args import TMArgs

@dataclass
class PackedClauses:
    clause_position_bounds: Any
    clause_feat_bounds: Any
    bounded_feat_ids: Any
    n_bounded_feats: Any
    clause_density: Any
    is_clause_synced: Any

    def get(self)-> "PackedClauses":
        return PackedClauses(
            clause_position_bounds=np.asarray(self.clause_position_bounds),
            clause_feat_bounds=np.asarray(self.clause_feat_bounds),
            bounded_feat_ids=np.asarray(self.bounded_feat_ids),
            n_bounded_feats=np.asarray(self.n_bounded_feats),
            clause_density=np.asarray(self.clause_density),
            is_clause_synced=np.asarray(self.is_clause_synced),
        )


def tqdm_bar(iter, **kwargs):
    args = dict(
        leave=False,
        dynamic_ncols=True,
        bar_format="{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
    )
    for k, v in kwargs.items():
        args[k] = v
    return tqdm(iter, **args)


class BaseDevice(abc.ABC):
    xp: types.ModuleType
    def __init__(self, args: TMArgs):
        self.args = args

        self.n_clause_banks = 1 if self.args.coalesced else self.args.n_classes
        self.total_clauses = self.n_clause_banks * self.args.n_clauses

        # The number of raw features in a patch
        self.n_raw_patch_feats = self.args.patch_dim[0] * self.args.patch_dim[1] * self.args.dim[2]

        self.n_patches_y = ((self.args.dim[0] - self.args.patch_dim[0]) // self.args.stride[0]) + 1
        self.n_patches_x = ((self.args.dim[1] - self.args.patch_dim[1]) // self.args.stride[1]) + 1
        self.n_patches = self.n_patches_y * self.n_patches_x

        # Number of position features. Uses thermometer encoding so need 1 less than the possible positions.
        self.n_position_feats = (self.n_patches_y - 1) + (self.n_patches_x - 1)

        # Thermometer bits per feature: max - min
        self.therm_bits = self.args.feat_maxs - self.args.feat_mins

        # The number of literals needed to represent all features using thermometer encoding
        self.n_patch_feats = int(np.sum(self.therm_bits))

        # Literal offsets: prefix sum for indexing into literals by feature
        self.literal_offsets = np.zeros(self.n_raw_patch_feats + 1, dtype=np.int32)
        self.literal_offsets[1:] = np.cumsum(self.therm_bits)

        # Total number of literals for all features + position
        self.n_literals = self.n_patch_feats + self.n_position_feats

        if self.args.negated_literals:
            self.n_literals *= 2

        if self.args.max_includes <= 0 or self.args.max_includes > self.n_literals:
            self.args.max_includes = self.n_literals

        # Precompute lit_to_fid lookup table: O(1) lookup instead of O(N_RAW_PATCH_FEATS) scan
        self.lit_to_fid = np.zeros(self.n_patch_feats, dtype=np.int32)
        for fid in range(self.n_raw_patch_feats):
            for lit in range(self.literal_offsets[fid], self.literal_offsets[fid + 1]):
                self.lit_to_fid[lit] = fid

        self.np_rng = np.random.default_rng(self.args.seed)

        self.dev_init()

    @abc.abstractmethod
    def dev_init(self):
        pass

    @abc.abstractmethod
    def pack_clauses(self):
        pass


    def _init_clauses(self):
        if self.args.ta_init == "middle":
            self.ta_states = self.xp.full(
                (self.total_clauses, self.n_literals),
                self.args.include_state - 1,
                dtype=np.uint32,
            )
        elif self.args.ta_init == "random":
            self.ta_states = self.xp.asarray(
                self.np_rng.integers(0, self.args.n_states, size=(self.total_clauses, self.n_literals)),
                dtype=np.uint32,
            )
        else:
            self.ta_states = self.xp.full(
                (self.total_clauses, self.n_literals),
                int(self.args.ta_init),
                dtype=np.uint32,
            )

    def _init_weights(self):
        shape = (self.args.n_classes, self.args.n_clauses)

        if self.args.weight_init == "random":
            mag = self.np_rng.uniform(0.0, 1.0, size=shape).astype(np.float32)
        else:
            mag = np.full(shape, float(self.args.weight_init), dtype=np.float32)

        if self.args.negative_clauses:
            n_neg_polarity = self.args.n_clauses // 2
            sign = np.ones(shape, dtype=np.float32)
            if self.args.coalesced:
                for i in range(self.args.n_classes):
                    pol = np.ones(self.args.n_clauses, dtype=np.float32)
                    pol[n_neg_polarity:] = -1.0
                    sign[i, :] = self.np_rng.permutation(pol)
            else:
                sign[:, n_neg_polarity:] = -1.0
            mag = mag * sign

        self.clause_weights = self.xp.asarray(mag, dtype=np.float32)

        if self.args.track_patch_weights:
            self.patch_weights = self.xp.zeros((self.total_clauses, self.n_patches), dtype=np.int32)
        else:
            self.patch_weights = self.xp.zeros((1, 1), dtype=np.int32)
