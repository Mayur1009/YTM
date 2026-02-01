from typing import Unpack
import numpy as np
from .config import TMOpt, TMOptArgs


class TM:
    def __init__(
        self,
        number_of_clauses: int,
        T: int,
        s: float,
        dim: tuple[int, int, int],
        n_classes: int,
        **opt_args: Unpack[TMOptArgs],
    ):
        # Required arguments
        self.number_of_clauses = number_of_clauses
        self.T = T
        self.s = s
        self.dim = dim
        self.number_of_outputs = n_classes
        self.opt = TMOpt(**opt_args)
        self._param_validation()
        self._init_device()
        self._mk_buffers()

    def _param_validation(self):
        if self.opt.patch_dim is None:
            self.patch_dim = (self.dim[0], self.dim[1])
        else:
            self.patch_dim = self.opt.patch_dim

        if self.opt.initial_state == "middle":
            self.initial_state = (self.opt.number_of_ta_states - 1) // 2
        elif self.opt.initial_state == "random":
            self.initial_state = -1
        elif isinstance(self.opt.initial_state, int) and 0 <= self.opt.initial_state < self.opt.number_of_ta_states:
            self.initial_state = self.opt.initial_state
        else:
            raise ValueError(
                "initial_state must be 'middle', 'random', or an integer between 0 and number_of_ta_states - 1."
            )

        if self.opt.include_state == "middle":
            self.include_state = ((self.opt.number_of_ta_states - 1) // 2) + 1
        elif isinstance(self.opt.include_state, int) and 0 <= self.opt.include_state < self.opt.number_of_ta_states:
            self.include_state = self.opt.include_state
        else:
            raise ValueError("include_state must be 'middle' or an integer between 0 and number_of_ta_states - 1.")

        if not self.opt.weighted and self.opt.initial_weight != 1.0:
            raise ValueError("initial_weight can only be set when weighted is True.")

        if self.opt.encode_loc:
            self.number_of_features = (
                (self.patch_dim[0] * self.patch_dim[1] * self.dim[2])
                + (self.dim[0] - self.patch_dim[0])
                + (self.dim[1] - self.patch_dim[1])
            )
        else:
            self.number_of_features = self.patch_dim[0] * self.patch_dim[1] * self.dim[2]

        if self.opt.negated_literals:
            self.number_of_literals = self.number_of_features * 2
        else:
            self.number_of_literals = self.number_of_features

        self.num_literal_chunks = (self.number_of_literals + 31) // 32

        self.number_of_patches = (self.dim[0] - self.patch_dim[0] + 1) * (self.dim[1] - self.patch_dim[1] + 1)

        if self.opt.coalesced:
            self.total_number_of_clauses = self.number_of_clauses
        else:
            self.total_number_of_clauses = self.number_of_clauses * self.number_of_outputs

        self.rng = np.random.default_rng(self.opt.seed)

    def _init_clauses(self):
        if self.initial_state == -1:
            # Random initialization
            ta_states = self.rng.integers(
                low=0,
                high=self.opt.number_of_ta_states,
                size=(self.total_number_of_clauses * self.number_of_literals),
                dtype=np.uint32,
            )
        else:
            # Fixed initialization
            ta_states = np.full(
                (self.total_number_of_clauses * self.number_of_literals),
                self.initial_state,
                dtype=np.uint32,
            )

        self.ta_states_dev = self.dev.allocate(self.total_number_of_clauses * self.number_of_literals, ta_states.nbytes)
        self.dev.to_device(self.ta_states_dev, ta_states)

    def _init_weights(self):
        num_neg_polarity = self.number_of_clauses // 2
        weights = np.zeros((self.number_of_outputs, self.number_of_clauses), dtype=np.float32)
        for i in range(self.number_of_outputs):
            wt = np.ones((self.number_of_clauses,), dtype=np.float32) * self.opt.initial_weight
            if self.opt.init_neg_weights:
                wt[num_neg_polarity:] *= -1.0
            weights[i, :] = self.rng.permutation(wt) if self.opt.coalesced else wt

        self.clause_weights_dev = self.dev.allocate(self.number_of_outputs * self.number_of_clauses, weights.nbytes)
        self.dev.to_device(self.clause_weights_dev, weights.flatten())

    def _mk_buffers(self):
        self._init_clauses()
        self._init_weights()
        self.patch_weights_dev = self.dev.allocate(self.number_of_clauses * self.number_of_patches, self.number_of_clauses * self.number_of_patches * 4)
        self.dev.memset(self.patch_weights_dev, 0, self.number_of_clauses * self.number_of_patches)

    def _init_device(self):
        header = f"""
        #define CLAUSES_PER_CLASS {self.number_of_clauses}ULL
        #define THRESH {self.T}
        #define S {self.s}
        #define DIM0 {self.dim[0]}ULL
        #define DIM1 {self.dim[1]}ULL
        #define DIM2 {self.dim[2]}ULL
        #define CLASSES {self.number_of_outputs}
        #define Q {self.opt.q}
        #define PATCH_DIM0 {self.patch_dim[0]}
        #define PATCH_DIM1 {self.patch_dim[1]}
        #define WEIGHTED {1 if self.opt.weighted else 0}
        #define COALESCED {1 if self.opt.coalesced else 0}
        #define ENCODE_LOC {1 if self.opt.encode_loc else 0}
        #define MAX_INCLUDED_LITERALS {self.number_of_literals if self.opt.max_included_literals is None else self.opt.max_included_literals}ULL
        #define NEGATED_LITERALS {1 if self.opt.negated_literals else 0}
        #define NEGATIVE_POLARITY {1 if self.opt.negative_polarity else 0}
        #define ALLOW_POLARITY_CHANGE {1 if self.opt.allow_polarity_change else 0}
        #define MAX_WEIGHT {self.opt.max_weight}
        #define MAX_TA_STATE {self.opt.number_of_ta_states - 1}
        #define INCLUDE_TA_STATE {self.include_state}
        #define TYPE1A_FB {1 if self.opt.type1a_fb else 0}
        #define TYPE1B_FB {1 if self.opt.type1b_fb else 0}
        #define TYPE2_FB {1 if self.opt.type2_fb else 0}
        #define TOTAL_CLAUSES {self.total_number_of_clauses}ULL
        #define PATCHES {self.number_of_patches}ULL
        #define LITERALS {self.number_of_literals}ULL
        """

        if self.opt.backend.lower().strip() == "cpu":
            from .backends.cpu import CPUBackend
            self.dev = CPUBackend(header=header, seed=self.opt.seed, **self.opt.backend_args)

        elif self.opt.backend.lower().strip() == "cuda":
            from .backends.cuda import CUDABackend
            self.dev = CUDABackend(header=header, seed=self.opt.seed, **self.opt.backend_args)

        else:
            raise NotImplementedError(f"Backend '{self.opt.backend}' is not implemented.")


    def encode(self, X: np.ndarray, batch_size: int = -1) -> np.ndarray[tuple[int, int, int], np.dtype[np.uint32]]:
        X = X.astype(np.int32).reshape((X.shape[0], self.dim[0], self.dim[1], self.dim[2]))
        N = X.shape[0]
        if batch_size == -1:
            batch_size = N

        encoded_X = np.zeros((N, self.number_of_patches, self.num_literal_chunks), dtype=np.uint32)

        for i in range(0, N, batch_size):
            X_batch = X[i : i + batch_size]
            X_dev = self.dev.allocate(np.prod(np.array(X_batch.shape)), X_batch.nbytes)
            self.dev.to_device(X_dev, X_batch)

            encoded_X_batch_dev = self.dev.allocate(X_batch.shape[0] * self.number_of_patches * self.num_literal_chunks, encoded_X[i : i + batch_size].nbytes)
            self.dev.memset(encoded_X_batch_dev, 0, X_batch.shape[0] * self.number_of_patches * self.num_literal_chunks)
            self.dev.encode_batch(X_dev, encoded_X_batch_dev, X_batch.shape[0], self.number_of_patches)

            temp = np.zeros((X_batch.shape[0], self.number_of_patches, self.num_literal_chunks), dtype=np.uint32)
            self.dev.to_host(temp, encoded_X_batch_dev)
            encoded_X[i : i + batch_size] = temp.reshape((X_batch.shape[0], self.number_of_patches, self.num_literal_chunks))

        return encoded_X

