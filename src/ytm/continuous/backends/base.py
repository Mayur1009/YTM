import abc
import numpy as np


class BaseDevice(abc.ABC):
    def __init__(self, args):
        self.args = args

        self.n_clause_banks = 1 if self.args.coalesced else self.args.n_classes
        self.total_clauses = self.n_clause_banks * self.args.n_clauses
        self.n_patches = (self.args.dim[0] - self.args.patch_dim[0] + 1) * (
            self.args.dim[1] - self.args.patch_dim[1] + 1
        )

        # The number of raw features in a patch
        self.n_features = self.args.patch_dim[0] * self.args.patch_dim[1] * self.args.dim[2]

        # Number of position features. Uses thermometer encoding so need 1 less than the possible positions.
        self.n_position_feat = (
            (self.args.dim[0] - self.args.patch_dim[0]) + (self.args.dim[1] - self.args.patch_dim[1])
            if self.args.position_literals
            else 0
        )

        # Thermometer bits per feature: max - min
        self.therm_bits = self.args.feat_maxs - self.args.feat_mins

        # The number of literals needed to represent all features using thermometer encoding
        self.n_feat_literals = int(np.sum(self.therm_bits))

        # Literal offsets: prefix sum for indexing into literals by feature
        self.literal_offsets = np.zeros(self.n_features + 1, dtype=np.int32)
        self.literal_offsets[1:] = np.cumsum(self.therm_bits)

        # Total number of literals for all features + position
        self.n_literals = self.n_feat_literals + self.n_position_feat

        if self.args.negated_literals:
            self.n_literals *= 2

        self.n_literal_chunks = (self.n_literals + 31) // 32

        if self.args.max_includes <= 0 or self.args.max_includes > self.n_literals:
            self.args.max_includes = self.n_literals

        self.np_rng = np.random.default_rng(self.args.seed)

        self.dev_init()

    @abc.abstractmethod
    def dev_init(self):
        pass
