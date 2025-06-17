import pathlib
import pickle
import sys

import numpy as np
import pycuda.autoinit  # noqa: F401
import pycuda.curandom as curandom
from pycuda.compiler import SourceModule
from pycuda.driver import Context as ctx  # pyright: ignore[reportAttributeAccessIssue]
from pycuda.driver import mem_alloc, memcpy_dtoh, memcpy_htod  # pyright: ignore[reportAttributeAccessIssue]
from pycuda.gpuarray import to_gpu
from scipy.sparse import csr_matrix
from tqdm import tqdm

current_dir = pathlib.Path(__file__).parent


def get_kernel(file):
    path = current_dir.joinpath(file)
    with path.open("r") as f:
        ker = f.read()
    return ker


code_header = """
#include <curand_kernel.h>
	
#define INT_SIZE 32

#define LA_CHUNKS (((FEATURES-1)/INT_SIZE + 1))
#define CLAUSE_CHUNKS ((CLAUSES-1)/INT_SIZE + 1)

#if (FEATURES % 32 != 0)
#define FILTER (~(0xffffffff << (FEATURES % INT_SIZE)))
#else
#define FILTER 0xffffffff
#endif

#define PATCH_CHUNKS (((PATCHES-1)/INT_SIZE + 1))

#if (PATCH_CHUNKS % 32 != 0)
#define PATCH_FILTER (~(0xffffffff << (PATCHES % INT_SIZE)))
#else
#define PATCH_FILTER 0xffffffff
#endif
"""

code_update = get_kernel("cuda/code_update.cu")
code_evaluate = get_kernel("cuda/code_evaluate.cu")
code_prepare = get_kernel("cuda/code_prepare.cu")
code_encode = get_kernel("cuda/code_encode.cu")
code_transform = get_kernel("cuda/code_transform.cu")
code_clauses = get_kernel("cuda/code_clauses.cu")



class CommonTsetlinMachine:
    def __init__(
        self,
        number_of_clauses: int,
        T: int,
        s: float,
        q: float = 1.0,
        max_included_literals: int | None = None,
        boost_true_positive_feedback: int = 1,
        number_of_state_bits: int = 8,
        append_negated: bool = True,
        r: float = 1.0,
        sr: float | None = None,
        encode_loc: bool = True,
        max_weight: int | None = None,
		seed: int | None = None,
        grid=(16 * 13, 1, 1),
        block=(128, 1, 1),
    ):
        # Initialize Hyperparams
        self.number_of_clauses = number_of_clauses
        self.number_of_clause_chunks = (number_of_clauses - 1) / 32 + 1
        self.number_of_state_bits = number_of_state_bits
        self.T = T
        self.s = s
        self.q = q
        self.max_included_literals = max_included_literals
        self.boost_true_positive_feedback = boost_true_positive_feedback
        self.append_negated = 1 if append_negated else 0
        self.r = r
        if sr is None:
            self.sr = s
        else:
            self.sr = sr
        self.encode_loc = 1 if encode_loc else 0
        self.max_weight = max_weight
        self.grid = grid
        self.block = block

        self.X_train = np.array([])
        self.X_test = np.array([])
        self.encoded_Y = np.array([])
        self.encoded_X_base = np.array([])
        self.encoded_X_packed_base = np.array([])
        self.ta_state = np.array([])
        self.clause_weights = np.array([])
        self.patch_weights = np.array([])

        if seed is None:
            self.rng_gpu = curandom.XORWOWRandomNumberGenerator()
        else:
            def _custom_seed_getter(count):
                return to_gpu(np.array([seed] * count, dtype=np.int32))

            self.rng_gpu = curandom.XORWOWRandomNumberGenerator(_custom_seed_getter)

        self.negative_clauses = 1  # Default is 1, set to 0 in RegressionTsetlinMachine
        self.initialized = False



    #### FIT AND SCORE ####
    def _fit(self, X, encoded_Y, epochs=1, incremental=True):
        # Initialize fit
        if not self.initialized:
            self._init_fit()
            self._init_kernels()
            self._reset_states_weights()
            self.initialized = True

        # If not incremental, clear ta-state and clause_weghts
        elif not incremental:
            self._reset_states_weights()

        # Copy data to Gpu
        if not np.array_equal(self.X_train, np.concatenate((X.indptr, X.indices))):
            self.X_train = np.concatenate((X.indptr, X.indices))
            self.X_train_indptr_gpu = mem_alloc(X.indptr.nbytes)
            memcpy_htod(self.X_train_indptr_gpu, X.indptr)

            self.X_train_indices_gpu = mem_alloc(X.indices.nbytes)
            memcpy_htod(self.X_train_indices_gpu, X.indices)

        if not np.array_equal(self.encoded_Y, encoded_Y):
            self.encoded_Y = encoded_Y
            self.encoded_Y_gpu = mem_alloc(encoded_Y.nbytes)
            memcpy_htod(self.encoded_Y_gpu, encoded_Y)

        # Initialize GPU memory for temporary data
        encoded_X_gpu = mem_alloc(self.encoded_X_base.nbytes)
        class_sum_gpu = mem_alloc(self.number_of_outputs * 4)
        clause_outputs_gpu = mem_alloc(self.number_of_clauses * 4)
        clause_patches_gpu = mem_alloc(self.number_of_clauses * 4)

        grid_encode = (min(self.grid[0], (self.number_of_patches + self.block[0] - 1) // self.block[0]), 1, 1)
        grid_evaluate = (min(self.grid[0], (self.number_of_clauses + self.block[0] - 1) // self.block[0]), 1, 1)
        grid_update = (min(self.grid[0], (self.number_of_clauses + self.block[0] - 1) // self.block[0]), 1, 1)

        class_sum_base = np.zeros(self.number_of_outputs).astype(np.int32)
        for _ in range(epochs):
            for e in tqdm(range(X.shape[0]), leave=False, desc="Fit"):
                memcpy_htod(class_sum_gpu, class_sum_base)
                memcpy_htod(encoded_X_gpu, self.encoded_X_base)

                self.encode.prepared_call(
                    grid_encode,
                    self.block,
                    self.X_train_indptr_gpu,
                    self.X_train_indices_gpu,
                    encoded_X_gpu,
                    np.int32(e),
                    np.int32(0),
                )
                ctx.synchronize()

                self.evaluate_update.prepared_call(
                    grid_evaluate,
                    self.block,
                    self.rng_gpu.state,
                    self.ta_state_gpu,
                    self.clause_weights_gpu,
                    self.patch_weights_gpu,
                    class_sum_gpu,
                    clause_outputs_gpu,
                    clause_patches_gpu,
                    encoded_X_gpu,
                )
                ctx.synchronize()

                self.update.prepared_call(
                    grid_update,
                    self.block,
                    self.rng_gpu.state,
                    self.ta_state_gpu,
                    self.clause_weights_gpu,
                    class_sum_gpu,
                    clause_outputs_gpu,
                    clause_patches_gpu,
                    encoded_X_gpu,
                    self.encoded_Y_gpu,
                    np.int32(e),
                )
                ctx.synchronize()

        # Free GPU memory
        encoded_X_gpu.free()
        class_sum_gpu.free()
        clause_outputs_gpu.free()
        clause_patches_gpu.free()
        return

    def _score(self, X):
        if not self.initialized:
            print("Error: Model not trained.")
            sys.exit(-1)

        if len(self.encoded_X_packed_base) == 0:
            self._init_encoded_X_packed_base()

        if not np.array_equal(self.X_test, np.concatenate((X.indptr, X.indices))):
            self.X_test = np.concatenate((X.indptr, X.indices))

            self.X_test_indptr_gpu = mem_alloc(X.indptr.nbytes)
            memcpy_htod(self.X_test_indptr_gpu, X.indptr)

            self.X_test_indices_gpu = mem_alloc(X.indices.nbytes)
            memcpy_htod(self.X_test_indices_gpu, X.indices)

        # Initialize GPU memory for temporary data
        encoded_X_packed_gpu = mem_alloc(self.encoded_X_packed_base.nbytes)
        class_sum_gpu = mem_alloc(self.number_of_outputs * 4)
        included_literals_gpu = mem_alloc(self.number_of_clauses * self.number_of_features * 2 * 4)
        included_literals_length_gpu = mem_alloc(self.number_of_clauses * 4)

        grid_prepare = (min(self.grid[0], (self.number_of_clauses + self.block[0] - 1) // self.block[0]), 1, 1)
        grid_encode = (min(self.grid[0], (self.number_of_patches + self.block[0] - 1) // self.block[0]), 1, 1)
        grid_evaluate = (min(self.grid[0], (self.number_of_clauses + self.block[0] - 1) // self.block[0]), 1, 1)
        self.prepare_packed.prepared_call(
            grid_prepare,
            self.block,
            self.rng_gpu.state,
            self.ta_state_gpu,
            included_literals_gpu,
            included_literals_length_gpu,
        )
        ctx.synchronize()

        class_sums = np.zeros((X.shape[0], self.number_of_outputs), dtype=np.int32)
        for e in tqdm(range(X.shape[0]), leave=False, desc="Predict"):
            memcpy_htod(class_sum_gpu, class_sums[e, :])
            memcpy_htod(encoded_X_packed_gpu, self.encoded_X_packed_base)

            self.encode_packed.prepared_call(
                grid_encode,
                self.block,
                self.X_test_indptr_gpu,
                self.X_test_indices_gpu,
                encoded_X_packed_gpu,
                np.int32(e),
                np.int32(0),
            )
            ctx.synchronize()

            self.evaluate_packed.prepared_call(
                grid_evaluate,
                self.block,
                included_literals_gpu,
                included_literals_length_gpu,
                self.clause_weights_gpu,
                class_sum_gpu,
                encoded_X_packed_gpu,
            )
            ctx.synchronize()

            memcpy_dtoh(class_sums[e, :], class_sum_gpu)

        # Free GPU memory
        encoded_X_packed_gpu.free()
        class_sum_gpu.free()
        included_literals_gpu.free()
        included_literals_length_gpu.free()

        return class_sums

    #### GPU INITIALIZATION ####
    def _init_kernels(self):
        parameters = f"""
		#define CLAUSES {self.number_of_clauses}
		#define THRESH {self.T}
		#define S {self.s}
		#define Q {self.q}
		#define DIM0 {self.dim[0]}
		#define DIM1 {self.dim[1]}
		#define DIM2 {self.dim[2]}
		#define PATCH_DIM0 {self.patch_dim[0]}
		#define PATCH_DIM1 {self.patch_dim[1]}
		#define APPEND_NEGATED {self.append_negated}
		#define STATE_BITS {self.number_of_state_bits}
		#define BOOST_TRUE_POSITIVE_FEEDBACK {self.boost_true_positive_feedback}
		#define MAX_INCLUDED_LITERALS {self.max_included_literals}
		#define NEGATIVE_CLAUSES {self.negative_clauses}
		#define RESISTANCE {self.r}
		#define SR {self.sr}
		#define CLASSES {self.number_of_outputs}
		#define FEATURES {self.number_of_features}
		#define PATCHES {self.number_of_patches}
		#define MAX_STATE {(1 << self.number_of_state_bits) - 1}
		#define ENCODE_LOC {self.encode_loc}
		#define MAX_WEIGHT {"INT_MAX" if self.max_weight is None else self.max_weight}
		"""

        # Encode and pack input
        mod_encode = SourceModule(parameters + code_header + code_encode, no_extern_c=True)
        self.encode = mod_encode.get_function("encode")
        self.encode.prepare("PPPii")

        self.encode_packed = mod_encode.get_function("encode_packed")
        self.encode_packed.prepare("PPPii")

        self.produce_autoencoder_examples = mod_encode.get_function("produce_autoencoder_example")
        self.produce_autoencoder_examples.prepare("PPiPPiPPiPPiiii")

        # Prepare
        mod_prepare = SourceModule(parameters + code_header + code_prepare, no_extern_c=True)
        self.prepare = mod_prepare.get_function("prepare")
        self.prepare.prepare("PPP")
        self.prepare_packed = mod_prepare.get_function("prepare_packed")
        self.prepare_packed.prepare("PPPP")

        # Update
        mod_update = SourceModule(parameters + code_header + code_update, no_extern_c=True)
        self.update = mod_update.get_function("update")
        self.update.prepare("PPPPPPPPi")

        self.evaluate_update = mod_update.get_function("evaluate")
        self.evaluate_update.prepare("PPPPPPPP")

        # Evaluate
        mod_evaluate = SourceModule(parameters + code_header + code_evaluate, no_extern_c=True)
        self.evaluate = mod_evaluate.get_function("evaluate")
        self.evaluate.prepare("PPPP")

        self.evaluate_packed = mod_evaluate.get_function("evaluate_packed")
        self.evaluate_packed.prepare("PPPPP")

        # Transform
        mod_transform = SourceModule(parameters + code_header + code_transform, no_extern_c=True)
        self.transform_gpu = mod_transform.get_function("transform")
        self.transform_gpu.prepare("PPPP")

        self.transform_patchwise_gpu = mod_transform.get_function("transform_patchwise")
        self.transform_patchwise_gpu.prepare("PPPP")

        # Misc Clause operations
        mod_clauses = SourceModule(parameters + code_header + code_clauses, no_extern_c=True)
        self.get_literals_gpu = mod_clauses.get_function("get_literals")
        self.get_literals_gpu.prepare("PP")

        self.get_ta_states_gpu = mod_clauses.get_function("get_ta_states")
        self.get_ta_states_gpu.prepare("PP")

        # Allocate GPU memory
        self.ta_state_gpu = mem_alloc(self.number_of_clauses * self.number_of_ta_chunks * self.number_of_state_bits * 4)
        self.clause_weights_gpu = mem_alloc(self.number_of_outputs * self.number_of_clauses * 4)
        self.patch_weights_gpu = mem_alloc(self.number_of_clauses * self.number_of_patches * 4)

    #### STATES, WEIGHTS, AND INPUT INITIALIZATION ####
    def _reset_states_weights(self):
        grid_prepare = (min(self.grid[0], (self.number_of_clauses + self.block[0] - 1) // self.block[0]), 1, 1)
        self.prepare.prepared_call(
            grid_prepare,
            self.block,
            self.rng_gpu.state,
            self.ta_state_gpu,
            self.clause_weights_gpu,
        )
        ctx.synchronize()

    def _init_fit(self):
        if self.encode_loc:
            self.number_of_features = int(
                self.patch_dim[0] * self.patch_dim[1] * self.dim[2]
                + (self.dim[0] - self.patch_dim[0])
                + (self.dim[1] - self.patch_dim[1])
            )
        else:
            self.number_of_features = int(self.patch_dim[0] * self.patch_dim[1] * self.dim[2])

        if self.append_negated:
            self.number_of_features *= 2

        if self.max_included_literals is None:
            self.max_included_literals = self.number_of_features

        self.number_of_patches = int((self.dim[0] - self.patch_dim[0] + 1) * (self.dim[1] - self.patch_dim[1] + 1))
        self.number_of_ta_chunks = int((self.number_of_features - 1) / 32 + 1)

        if len(self.encoded_X_base) == 0:
            encoded_X = np.zeros((self.number_of_patches, self.number_of_ta_chunks), dtype=np.uint32)
            for patch_coordinate_y in range(self.dim[1] - self.patch_dim[1] + 1):
                for patch_coordinate_x in range(self.dim[0] - self.patch_dim[0] + 1):
                    p = patch_coordinate_y * (self.dim[0] - self.patch_dim[0] + 1) + patch_coordinate_x

                    if self.append_negated:
                        for k in range(self.number_of_features // 2, self.number_of_features):
                            chunk = k // 32
                            pos = k % 32
                            encoded_X[p, chunk] |= 1 << pos

                    for y_threshold in range(self.dim[1] - self.patch_dim[1]):
                        patch_pos = y_threshold
                        if patch_coordinate_y > y_threshold:
                            chunk = patch_pos // 32
                            pos = patch_pos % 32
                            encoded_X[p, chunk] |= 1 << pos

                            if self.append_negated:
                                chunk = (patch_pos + self.number_of_features // 2) // 32
                                pos = (patch_pos + self.number_of_features // 2) % 32
                                encoded_X[p, chunk] &= ~np.uint32(1 << pos)

                    for x_threshold in range(self.dim[0] - self.patch_dim[0]):
                        patch_pos = (self.dim[1] - self.patch_dim[1]) + x_threshold
                        if patch_coordinate_x > x_threshold:
                            chunk = patch_pos // 32
                            pos = patch_pos % 32
                            encoded_X[p, chunk] |= 1 << pos

                            if self.append_negated:
                                chunk = (patch_pos + self.number_of_features // 2) // 32
                                pos = (patch_pos + self.number_of_features // 2) % 32
                                encoded_X[p, chunk] &= ~np.uint32(1 << pos)

            self.encoded_X_base = encoded_X.reshape(-1)

    def _init_encoded_X_packed_base(self):
        # Encoded X packed
        encoded_X_packed = np.zeros(((self.number_of_patches - 1) // 32 + 1, self.number_of_features), dtype=np.uint32)
        if self.append_negated:
            for p_chunk in range((self.number_of_patches - 1) // 32 + 1):
                for k in range(self.number_of_features // 2, self.number_of_features):
                    encoded_X_packed[p_chunk, k] = ~np.uint32(0)

        for patch_coordinate_y in range(self.dim[1] - self.patch_dim[1] + 1):
            for patch_coordinate_x in range(self.dim[0] - self.patch_dim[0] + 1):
                p = patch_coordinate_y * (self.dim[0] - self.patch_dim[0] + 1) + patch_coordinate_x
                p_chunk = p // 32
                p_pos = p % 32

                for y_threshold in range(self.dim[1] - self.patch_dim[1]):
                    patch_pos = y_threshold
                    if patch_coordinate_y > y_threshold:
                        encoded_X_packed[p_chunk, patch_pos] |= 1 << p_pos

                        if self.append_negated:
                            encoded_X_packed[p_chunk, patch_pos + self.number_of_features // 2] &= ~np.uint32(
                                1 << p_pos
                            )

                for x_threshold in range(self.dim[0] - self.patch_dim[0]):
                    patch_pos = (self.dim[1] - self.patch_dim[1]) + x_threshold
                    if patch_coordinate_x > x_threshold:
                        encoded_X_packed[p_chunk, patch_pos] |= 1 << p_pos

                        if self.append_negated:
                            encoded_X_packed[p_chunk, patch_pos + self.number_of_features // 2] &= ~np.uint32(
                                1 << p_pos
                            )

        self.encoded_X_packed_base = encoded_X_packed.reshape(-1)

    #### CAUSE and WEIGHT OPERATIONS ####
    def ta_action(self, clause, ta):
        if np.array_equal(self.ta_state, np.array([])):
            self.ta_state = np.empty(
                self.number_of_clauses * self.number_of_ta_chunks * self.number_of_state_bits, dtype=np.uint32
            )
            memcpy_dtoh(self.ta_state, self.ta_state_gpu)
        ta_state = self.ta_state.reshape((self.number_of_clauses, self.number_of_ta_chunks, self.number_of_state_bits))
        return (ta_state[clause, ta // 32, self.number_of_state_bits - 1] & (1 << (ta % 32))) > 0

    def get_literals(self):
        literals_gpu = mem_alloc(self.number_of_clauses * self.number_of_features * 4)
        self.get_literals_gpu(
            self.ta_state_gpu,
            literals_gpu,
            grid=self.grid,
            block=self.block,
        )
        ctx.synchronize()

        literals = np.empty((self.number_of_clauses * self.number_of_features), dtype=np.uint32)
        memcpy_dtoh(literals, literals_gpu)
        return literals.reshape((self.number_of_clauses, self.number_of_features)).astype(np.uint8)

    def get_ta_states(self):
        ta_states_gpu = mem_alloc(self.number_of_clauses * self.number_of_features * 4)
        self.get_ta_states_gpu(
            self.ta_state_gpu,
            ta_states_gpu,
            grid=self.grid,
            block=self.block,
        )
        ctx.synchronize()

        ta_states = np.empty((self.number_of_clauses * self.number_of_features), dtype=np.uint32)
        memcpy_dtoh(ta_states, ta_states_gpu)
        return ta_states.reshape((self.number_of_clauses, self.number_of_features))

    def get_weights(self):
        self.clause_weights = np.empty(self.number_of_outputs * self.number_of_clauses, dtype=np.int32)
        memcpy_dtoh(self.clause_weights, self.clause_weights_gpu)

        return self.clause_weights.reshape((self.number_of_outputs, self.number_of_clauses))

    def get_patch_weights(self):
        self.patch_weights = np.empty(self.number_of_clauses * self.number_of_patches, dtype=np.int32)
        memcpy_dtoh(self.patch_weights, self.patch_weights_gpu)

        return self.patch_weights.reshape(
            (
                self.number_of_clauses,
                self.dim[0] - self.patch_dim[0] + 1,
                self.dim[1] - self.patch_dim[1] + 1,
            )
        )

    # Transform input data for processing at next layer
    def transform(self, X) -> csr_matrix:
        """Returns csr_matix of clause outputs. Array shape: (num_samples, num_clauses)"""
        if not self.initialized:
            print("Error: Model not trained.")
            sys.exit(-1)

        if len(self.encoded_X_packed_base) == 0:
            self._init_encoded_X_packed_base()

        if len(self.encoded_X_packed_base) == 0:
            self._init_encoded_X_packed_base()

        X = csr_matrix(X)
        number_of_examples = X.shape[0]

        # Copy data to GPU
        X_indptr_gpu = mem_alloc(X.indptr.nbytes)
        X_indices_gpu = mem_alloc(X.indices.nbytes)
        memcpy_htod(X_indptr_gpu, X.indptr)
        memcpy_htod(X_indices_gpu, X.indices)

        X_transformed = np.empty((number_of_examples, self.number_of_clauses), dtype=np.uint32)
        X_transformed_gpu = mem_alloc(self.number_of_clauses * 4)

        # Initialize GPU memory for temporary data
        encoded_X_packed_gpu = mem_alloc(self.encoded_X_packed_base.nbytes)
        included_literals_gpu = mem_alloc(self.number_of_clauses * self.number_of_features * 2 * 4)
        included_literals_length_gpu = mem_alloc(self.number_of_clauses * 4)

        grid_prepare = (min(self.grid[0], (self.number_of_clauses + self.block[0] - 1) // self.block[0]), 1, 1)
        self.prepare_packed.prepared_call(
            grid_prepare,
            self.block,
            self.rng_gpu.state,
            self.ta_state_gpu,
            included_literals_gpu,
            included_literals_length_gpu,
        )
        ctx.synchronize()

        for e in range(number_of_examples):
            memcpy_htod(encoded_X_packed_gpu, self.encoded_X_packed_base)
            self.encode_packed.prepared_call(
                self.grid,
                self.block,
                X_indptr_gpu,
                X_indices_gpu,
                encoded_X_packed_gpu,
                np.int32(e),
                np.int32(0),
            )
            ctx.synchronize()

            self.transform_gpu(
                included_literals_gpu,
                included_literals_length_gpu,
                encoded_X_packed_gpu,
                X_transformed_gpu,
                grid=self.grid,
                block=self.block,
            )
            ctx.synchronize()

            memcpy_dtoh(X_transformed[e, :], X_transformed_gpu)

        encoded_X_packed_gpu.free()
        included_literals_gpu.free()
        included_literals_length_gpu.free()

        X_transformed = (X_transformed > 0).astype(np.uint8)
        return csr_matrix(X_transformed)

    def transform_patchwise(self, X) -> csr_matrix:
        """Returns SPARSE CSR MATRIX of patch outputs for each clause. Array shape: (num_samples, num_clauses * num_patches)"""
        if not self.initialized:
            print("Error: Model not trained.")
            sys.exit(-1)

        if len(self.encoded_X_packed_base) == 0:
            self._init_encoded_X_packed_base()

        X = csr_matrix(X)
        number_of_examples = X.shape[0]

        # Copy data to GPU
        X_indptr_gpu = mem_alloc(X.indptr.nbytes)
        X_indices_gpu = mem_alloc(X.indices.nbytes)
        memcpy_htod(X_indptr_gpu, X.indptr)
        memcpy_htod(X_indices_gpu, X.indices)

        # Array to capture output from gpu
        X_transformed = np.empty(
            (number_of_examples, self.number_of_clauses * self.number_of_patches),
            dtype=np.uint32,
        )
        X_transformed_gpu = mem_alloc(self.number_of_clauses * self.number_of_patches * 4)

        # Initialize GPU memory for temporary data
        encoded_X_packed_gpu = mem_alloc(self.encoded_X_packed_base.nbytes)
        included_literals_gpu = mem_alloc(self.number_of_clauses * self.number_of_features * 2 * 4)
        included_literals_length_gpu = mem_alloc(self.number_of_clauses * 4)

        grid_prepare = (min(self.grid[0], (self.number_of_clauses + self.block[0] - 1) // self.block[0]), 1, 1)
        self.prepare_packed.prepared_call(
            grid_prepare,
            self.block,
            self.rng_gpu.state,
            self.ta_state_gpu,
            included_literals_gpu,
            included_literals_length_gpu,
        )
        ctx.synchronize()

        for e in range(number_of_examples):
            memcpy_htod(encoded_X_packed_gpu, self.encoded_X_packed_base)
            self.encode_packed.prepared_call(
                self.grid,
                self.block,
                X_indptr_gpu,
                X_indices_gpu,
                encoded_X_packed_gpu,
                np.int32(e),
                np.int32(0),
            )
            ctx.synchronize()

            self.transform_patchwise_gpu(
                included_literals_gpu,
                included_literals_length_gpu,
                encoded_X_packed_gpu,
                X_transformed_gpu,
                grid=self.grid,
                block=self.block,
            )
            ctx.synchronize()

            memcpy_dtoh(X_transformed[e, :], X_transformed_gpu)

        encoded_X_packed_gpu.free()
        included_literals_gpu.free()
        included_literals_length_gpu.free()

        # NOTE: RETURNS CSR_MATRIX
        return csr_matrix(X_transformed.reshape((number_of_examples, self.number_of_clauses * self.number_of_patches)))

    def encode_X_patches(self, X):
        """
        Convert X into patches and encode them in ta_chunks as TM would in normal training. Also, decode the encoded patches for sanity check.
        The decoded patches when put together should give original X.

        :param X: Input data
                        shape: (num_samples, dim[0] * dim[1] * dim[2])

        :return: Patches encoded in ta_chunks
                        shape: (num_samples, num_patches, num_ta_chunks)
        """
        if not self.initialized:
            print("Error: Model not trained.")
            sys.exit(-1)

        X = csr_matrix(X)
        number_of_examples = X.shape[0]

        X_indptr_gpu = mem_alloc(X.indptr.nbytes)
        X_indices_gpu = mem_alloc(X.indices.nbytes)
        memcpy_htod(X_indptr_gpu, X.indptr)
        memcpy_htod(X_indices_gpu, X.indices)

        encoded_X = np.zeros((number_of_examples, self.number_of_patches * self.number_of_ta_chunks), dtype=np.uint32)

        grid_encode = (min(self.grid[0], (self.number_of_patches + self.block[0] - 1) // self.block[0]), 1, 1)

        encoded_X_gpu = mem_alloc(self.encoded_X_base.nbytes)

        for e in range(number_of_examples):
            memcpy_htod(encoded_X_gpu, self.encoded_X_base)
            self.encode.prepared_call(
                grid_encode,
                self.block,
                X_indptr_gpu,
                X_indices_gpu,
                encoded_X_gpu,
                np.int32(e),
                np.int32(self.dim[0]),
                np.int32(self.dim[1]),
                np.int32(self.dim[2]),
                np.int32(self.patch_dim[0]),
                np.int32(self.patch_dim[1]),
                np.int32(self.append_negated),
                np.int32(0),
            )
            ctx.synchronize()

            memcpy_dtoh(encoded_X[e, :], encoded_X_gpu)

        encoded_X_gpu.free()

        return encoded_X.reshape((number_of_examples, self.number_of_patches, self.number_of_ta_chunks))

    def encode_X_packed_patches(self, X):
        """
        Convert X into patches and encode them in ta_chunks as TM would in normal training. Also, decode the encoded patches for sanity check.
        The decoded patches when put together should give original X.

        :param X: Input data
                        shape: (num_samples, dim[0] * dim[1] * dim[2])

        :return: Literals encoded in patch_chunks
                        shape: (num_samples, num_patch_chunks, num_literals)
        """
        if not self.initialized:
            print("Error: Model not trained.")
            sys.exit(-1)

        if len(self.encoded_X_packed_base) == 0:
            self._init_encoded_X_packed_base()

        X = csr_matrix(X)
        number_of_examples = X.shape[0]

        X_indptr_gpu = mem_alloc(X.indptr.nbytes)
        X_indices_gpu = mem_alloc(X.indices.nbytes)
        memcpy_htod(X_indptr_gpu, X.indptr)
        memcpy_htod(X_indices_gpu, X.indices)

        number_of_patch_chunks = (self.number_of_patches - 1) // 32 + 1
        encoded_X = np.zeros((number_of_examples, number_of_patch_chunks * self.number_of_features), dtype=np.uint32)

        grid_encode = (min(self.grid[0], (self.number_of_patches + self.block[0] - 1) // self.block[0]), 1, 1)

        encoded_X_packed_gpu = mem_alloc(self.encoded_X_packed_base.nbytes)

        for e in range(number_of_examples):
            memcpy_htod(encoded_X_packed_gpu, self.encoded_X_packed_base)
            self.encode.prepared_call(
                grid_encode,
                self.block,
                X_indptr_gpu,
                X_indices_gpu,
                encoded_X_packed_gpu,
                np.int32(e),
                np.int32(self.dim[0]),
                np.int32(self.dim[1]),
                np.int32(self.dim[2]),
                np.int32(self.patch_dim[0]),
                np.int32(self.patch_dim[1]),
                np.int32(self.append_negated),
                np.int32(0),
            )
            ctx.synchronize()

            memcpy_dtoh(encoded_X[e, :], encoded_X_packed_gpu)

        encoded_X_packed_gpu.free()

        return encoded_X.reshape((number_of_examples, number_of_patch_chunks, self.number_of_features))

    def unchunk_encoded_X(self, encoded_X):
        """
        Undo the chunking of encoded_X. The encoded_X is chunked into ta_chunks for processing in the TM. This function "unchunks" the encoded_X.

        :param encoded_X: Encoded patches
                        shape: (num_samples, num_patches, num_ta_chunks)

        :return: Unchunked patches
                        shape: (num_samples, num_patches, num_features)
        """

        num_samples = encoded_X.shape[0]
        unchunked_X = np.zeros((num_samples, self.number_of_patches, self.number_of_features), dtype=np.uint8)

        for e in range(num_samples):
            for p in range(self.number_of_patches):
                for k in range(self.number_of_features):
                    chunk = k // 32
                    pos = k % 32
                    unchunked_X[e, p, k] = (encoded_X[e, p, chunk] & (1 << pos)) > 0

        return unchunked_X.reshape((num_samples, self.number_of_patches, self.number_of_features))

    def unchunk_encoded_X_packed(self, encoded_X_packed):
        """
        Undo the chunking of encoded_X. The encoded_X is chunked into ta_chunks for processing in the TM. This function "unchunks" the encoded_X.

        :param encoded_X_packed: Encoded patches
                        shape: (num_samples, num_patch_chunks, num_features)

        :return: Unchunked patches
                        shape: (num_samples, num_patches, num_features)
        """

        num_samples = encoded_X_packed.shape[0]
        unchunked_X = np.zeros((num_samples, self.number_of_patches, self.number_of_features), dtype=np.uint8)

        for e in range(num_samples):
            for p in range(self.number_of_patches):
                chunk = p // 32
                pos = p % 32
                for k in range(self.number_of_features):
                    unchunked_X[e, p, k] = (encoded_X_packed[e, chunk, k] & (1 << pos)) > 0

        return unchunked_X.reshape((num_samples, self.number_of_patches, self.number_of_features))

    #### SAVE AND LOAD ####
    def save(self, fname=""):
        # Copy data from GPU to CPU
        self.ta_state = np.empty(
            self.number_of_clauses * self.number_of_ta_chunks * self.number_of_state_bits,
            dtype=np.uint32,
        )
        self.clause_weights = np.empty(self.number_of_outputs * self.number_of_clauses, dtype=np.int32)
        self.patch_weights = np.empty(self.number_of_clauses * self.number_of_patches, dtype=np.int32)

        memcpy_dtoh(self.ta_state, self.ta_state_gpu)
        memcpy_dtoh(self.clause_weights, self.clause_weights_gpu)
        memcpy_dtoh(self.patch_weights, self.patch_weights_gpu)

        state_dict = {
            # State arrays
            "ta_state": self.ta_state,
            "clause_weights": self.clause_weights,
            "patch_weights": self.patch_weights,
            "number_of_outputs": self.number_of_outputs,
            "number_of_features": self.number_of_features,
            "min_y": self.min_y,
            "max_y": self.max_y,
            "negative_clauses": self.negative_clauses,  # Set in children classes, should be set in this class.
            # Parameters
            "number_of_clauses": self.number_of_clauses,
            "T": self.T,
            "s": self.s,
            "q": self.q,
            "patch_dim": self.patch_dim,
            "r": self.r,
            "sr": self.sr,
            "dim": self.dim,
            "max_included_literals": self.max_included_literals,
            "boost_true_positive_feedback": self.boost_true_positive_feedback,
            "number_of_state_bits": self.number_of_state_bits,
            "append_negated": self.append_negated,
            "encode_loc": self.encode_loc,
            "max_weight": self.max_weight,
        }

        # Save to file
        if len(fname) > 0:
            print(f"Saving model to {fname}.")
            with open(fname, "wb") as f:
                pickle.dump(state_dict, f)

        return state_dict

    def load(self, state_dict={}, fname=""):
        if len(fname) == 0 and len(state_dict) == 0:
            print("Error: No file or state_dict provided. Pass either a file name or a state_dict.")
            return

        # Load from file
        if len(fname) > 0:
            print(f"Loading model from {fname}.")
            with open(fname, "rb") as f:
                state_dict = pickle.load(f)

        # Load arrays state_dict
        self.ta_state = state_dict["ta_state"]
        self.clause_weights = state_dict["clause_weights"]
        self.patch_weights = state_dict["patch_weights"]
        self.number_of_outputs = state_dict["number_of_outputs"]
        self.dim = state_dict["dim"]
        self.patch_dim = state_dict["patch_dim"]
        self.min_y = state_dict["min_y"]
        self.max_y = state_dict["max_y"]
        self.negative_clauses = state_dict["negative_clauses"]

        self._init_fit()
        self._init_kernels()

        memcpy_htod(self.ta_state_gpu, self.ta_state)
        memcpy_htod(self.clause_weights_gpu, self.clause_weights)
        memcpy_htod(self.patch_weights_gpu, self.patch_weights)

        self.initialized = True
