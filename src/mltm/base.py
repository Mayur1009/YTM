import pathlib
import pickle
import sys

import numpy as np
import pycuda.autoinit  # noqa: F401
import pycuda.curandom as curandom
from pycuda.compiler import SourceModule
from pycuda.driver import Context as ctx  # pyright: ignore[reportAttributeAccessIssue]
from pycuda.driver import mem_alloc, memcpy_dtoh, memcpy_htod, device_attribute  # pyright: ignore[reportAttributeAccessIssue]
from pycuda.gpuarray import to_gpu
from tqdm import tqdm

current_dir = pathlib.Path(__file__).parent


def get_kernel(file):
    path = current_dir.joinpath(file)
    with path.open("r") as f:
        ker = f.read()
    return ker


def get_device_properties():
    """Query GPU device properties for optimization"""
    device = pycuda.autoinit.device
    attrs = device.get_attributes()
    properties = {
        "max_threads_per_block": attrs[device_attribute.MAX_THREADS_PER_BLOCK],
        "max_block_dim_x": attrs[device_attribute.MAX_BLOCK_DIM_X],
        "max_grid_dim_x": attrs[device_attribute.MAX_GRID_DIM_X],
        "warp_size": attrs[device_attribute.WARP_SIZE],
        "multiprocessor_count": attrs[device_attribute.MULTIPROCESSOR_COUNT],
        "max_shared_memory_per_block": attrs[device_attribute.MAX_SHARED_MEMORY_PER_BLOCK],
    }
    return properties


def kernel_config(data_size, props, preferred_block_size=128):
    """Get optimal grid and block configuration for 1D kernel"""

    # Ensure hardware compliance
    block_size = min(preferred_block_size, props["max_threads_per_block"])
    block_size = ((block_size + 31) // 32) * 32

    # Calculate grid size
    grid_size = (data_size + block_size - 1) // block_size

    # Limit grid size to reasonable bounds
    max_blocks = min(65535, props["multiprocessor_count"] * 8)
    grid_size = min(grid_size, max_blocks)

    return (grid_size, 1, 1), (block_size, 1, 1)


new_kernel = get_kernel("cuda/kernel.cu")


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
                # Make sure that each thread gets a different seed
                return to_gpu(np.array([(seed + i) * (i + 1) for i in range(1, count + 1)], dtype=np.int32))

            self.rng_gpu = curandom.XORWOWRandomNumberGenerator(_custom_seed_getter)

        self.negative_clauses = 1  # Default is 1, set to 0 in RegressionTsetlinMachine
        self.initialized = False

        self.device_props = get_device_properties()

    def _validate_fit_args(self, X, encoded_Y):
        pass

    #### FIT AND SCORE ####
    def _fit(self, X, encoded_Y, epochs=1, incremental=True):
        N = X.shape[0]
        self._validate_fit_args(X, encoded_Y)

        if not self.initialized:
            self._calc_variables()
            self._gpu_init()
            self._init_default_vals()
            self.initialized = True

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
        encoded_Y_gpu = mem_alloc(self.number_of_outputs * 4)
        class_sum_gpu = mem_alloc(self.number_of_outputs * 4)
        selected_patch_ids_gpu = mem_alloc(self.number_of_clauses * 4)
        num_includes_gpu = mem_alloc(self.number_of_clauses * 4)

        # temp_ta_states = np.empty(self.number_of_clauses * self.number_of_features, dtype=np.uint32)
        # temp_clause_weights = np.empty(self.number_of_clauses * self.number_of_outputs, dtype=np.float32)
        # memcpy_dtoh(temp_ta_states, self.ta_state_gpu)
        # memcpy_dtoh(temp_clause_weights, self.clause_weights_gpu)
        # temp_ta_states = temp_ta_states.reshape((self.number_of_clauses, self.number_of_features))
        # temp_clause_weights = temp_clause_weights.reshape((self.number_of_clauses, self.number_of_outputs))
        # print(f'{temp_ta_states=}')
        # print(f'{temp_clause_weights=}')
        # breakpoint()

        # clause_eval_block = (16, 16, 1) if self.number_of_clauses > 256 else (8, 8, 1)
        # clause_eval_grid = (
        #         min(32, (self.number_of_clauses + clause_eval_block[0] - 1) // clause_eval_block[0]),
        #         min(32, (self.number_of_outputs + clause_eval_block[1] - 1) // clause_eval_block[1]),
        #         1,
        # )

        class_sum_base = np.zeros(self.number_of_outputs).astype(np.float32)
        selected_patch_ids_base = -1 * np.ones(self.number_of_clauses, dtype=np.int32)
        class_sum_sample = np.zeros(self.number_of_outputs, dtype=np.float32)
        num_correct_preds = 0

        pbar = tqdm(range(X.shape[0]), total=N, desc="Fit", leave=False, dynamic_ncols=True)
        for e in pbar:
            memcpy_htod(class_sum_gpu, class_sum_base)
            memcpy_htod(selected_patch_ids_gpu, selected_patch_ids_base)
            memcpy_htod(encoded_X_gpu, self.encoded_X_base)
            memcpy_htod(encoded_Y_gpu, self.encoded_Y[e, :])

            self.kernel_encode.prepared_call(
                *self.encode_config,
                self.X_train_indptr_gpu,
                self.X_train_indices_gpu,
                encoded_X_gpu,
                np.int32(e),
            )
            ctx.synchronize()

            # temp_encoded_X = np.zeros_like(self.encoded_X_base, dtype=np.uint32)
            # memcpy_dtoh(temp_encoded_X, encoded_X_gpu)
            # temp_encoded_X = temp_encoded_X.reshape(self.number_of_patches, self.number_of_features)

            self.kernel_clause_eval.prepared_call(
                *self.clause_eval_config,
                self.rng_gpu.state,
                self.ta_state_gpu,
                self.clause_weights_gpu,
                encoded_X_gpu,
                selected_patch_ids_gpu,
            )
            ctx.synchronize()

            memcpy_htod(class_sum_gpu, class_sum_base)
            self.kernel_calc_class_sums.prepared_call(
                *self.calc_class_sums_config,
                selected_patch_ids_gpu,
                self.clause_weights_gpu,
                class_sum_gpu,
            )
            ctx.synchronize()

            # temp_class_sum2 = np.zeros(self.number_of_outputs, dtype=np.float32)
            # memcpy_dtoh(temp_class_sum2, class_sum_gpu)
            # pred = np.argmax(temp_class_sum2)
            # true_label = np.argmax(self.encoded_Y[e, :])
            # if pred == true_label:
            #     num_correct_samples += 1
            # print(f"{temp_class_sum2=}")
            #
            # temp_selected_patch_ids = np.zeros(self.number_of_clauses, dtype=np.int32)
            # memcpy_dtoh(temp_selected_patch_ids, selected_patch_ids_gpu)
            # print(f"{temp_selected_patch_ids=}")

            self.kernel_calc_num_includes.prepared_call(
                *self.calc_num_includes_config,
                self.ta_state_gpu,
                num_includes_gpu,
            )
            ctx.synchronize()

            # temp_num_includes = np.zeros(self.number_of_clauses, dtype=np.int32)
            # memcpy_dtoh(temp_num_includes, num_includes_gpu)
            # print(f"{temp_num_includes=}")

            self.kernel_clause_update.prepared_call(
                *self.clause_update_config,
                self.rng_gpu.state,
                self.ta_state_gpu,
                self.clause_weights_gpu,
                class_sum_gpu,
                selected_patch_ids_gpu,
                num_includes_gpu,
                encoded_X_gpu,
                encoded_Y_gpu,
            )
            ctx.synchronize()

            memcpy_dtoh(class_sum_sample, class_sum_gpu)
            num_correct_preds += 1 if np.argmax(class_sum_sample) == np.argmax(encoded_Y[e, :]) else 0
            pbar.set_postfix_str(
                f"Correct: {num_correct_preds}/{e + 1} (Acc: {(num_correct_preds / (e + 1)) * 100:.4f}%)"
            )
            # prob = np.abs(np.clip(class_sum_sample, -self.T, self.T) + encoded_Y[e]) / (2 * self.T)
            #
            # if (e + 1) % 1000 == 0:
            #     pbar.set_postfix_str(f"Avg prob: {np.mean(prob):.4f}")

            # if e % 20000 == 0:
            #     breakpoint()

            # temp_clause_weights = np.zeros(self.number_of_clauses * self.number_of_outputs, dtype=np.float32)
            # memcpy_dtoh(temp_clause_weights, self.clause_weights_gpu)
            # temp_clause_weights = temp_clause_weights.reshape(self.number_of_clauses, self.number_of_outputs)
            # print(f"{temp_clause_weights=}")
            #
            # temp_ta_states = np.zeros(self.number_of_clauses * self.number_of_features, dtype=np.uint32)
            # memcpy_dtoh(temp_ta_states, self.ta_state_gpu)
            # temp_ta_states = temp_ta_states.reshape(self.number_of_clauses, self.number_of_features)
            # print(f"{temp_ta_states=}")
            #
            #
            # breakpoint()

        # Free GPU memory
        # encoded_X_gpu.free()
        # encoded_Y_gpu.free()
        # class_sum_gpu.free()
        # selected_patch_ids_gpu.free()
        # num_includes_gpu.free()
        return

    def _score(self, X):
        if not self.initialized:
            print("Error: Model not trained.")
            sys.exit(-1)

        if not hasattr(self, "encoded_X_base"):
            self._init_encoded_X_base()

        if not np.array_equal(self.X_test, np.concatenate((X.indptr, X.indices))):
            self.X_test = np.concatenate((X.indptr, X.indices))

            self.X_test_indptr_gpu = mem_alloc(X.indptr.nbytes)
            memcpy_htod(self.X_test_indptr_gpu, X.indptr)

            self.X_test_indices_gpu = mem_alloc(X.indices.nbytes)
            memcpy_htod(self.X_test_indices_gpu, X.indices)

        # Initialize GPU memory for temporary data
        encoded_X_gpu = mem_alloc(self.encoded_X_base.nbytes)
        class_sum_gpu = mem_alloc(self.number_of_outputs * 4)
        selected_patch_ids_gpu = mem_alloc(self.number_of_clauses * 4)

        class_sums = np.zeros((X.shape[0], self.number_of_outputs), dtype=np.float32)
        class_sum_base = np.zeros(self.number_of_outputs).astype(np.float32)
        selected_patch_ids_base = -1 * np.ones(self.number_of_clauses, dtype=np.int32)

        for e in tqdm(range(X.shape[0]), leave=False, desc="Fit"):
            memcpy_htod(class_sum_gpu, class_sum_base)
            memcpy_htod(selected_patch_ids_gpu, selected_patch_ids_base)
            memcpy_htod(encoded_X_gpu, self.encoded_X_base)

            self.kernel_encode.prepared_call(
                self.grid,
                self.block,
                self.X_test_indptr_gpu,
                self.X_test_indices_gpu,
                encoded_X_gpu,
                np.int32(e),
            )
            ctx.synchronize()

            self.kernel_clause_eval.prepared_call(
                self.grid,
                self.block,
                self.rng_gpu.state,
                self.ta_state_gpu,
                self.clause_weights_gpu,
                encoded_X_gpu,
                selected_patch_ids_gpu,
            )
            ctx.synchronize()

            self.kernel_calc_class_sums.prepared_call(
                self.grid,
                self.block,
                selected_patch_ids_gpu,
                self.clause_weights_gpu,
                class_sum_gpu,
            )
            ctx.synchronize()

            memcpy_dtoh(class_sums[e, :], class_sum_gpu)

        # Free GPU memory
        # encoded_X_gpu.free()
        # encoded_Y_gpu.free()
        # class_sum_gpu.free()
        # selected_patch_ids_gpu.free()

        return class_sums

    #### GPU INITIALIZATION ####
    def _gpu_init(self):
        self.gpu_macro_string = f"""
        #define CLAUSES {self.number_of_clauses}
        #define THRESH {self.T}
        #define S {self.s}
        #define Q {self.q}
        #define DIM0 {self.dim[0]}
        #define DIM1 {self.dim[1]}
        #define DIM2 {self.dim[2]}
        #define PATCH_DIM0 {self.patch_dim[0]}
        #define PATCH_DIM1 {self.patch_dim[1]}
        #define PATCHES {self.number_of_patches}
        #define LITERALS {self.number_of_literals}
        #define MAX_INCLUDED_LITERALS {self.max_included_literals}
        #define APPEND_NEGATED {self.append_negated}
        #define NEGATIVE_CLAUSES {self.negative_clauses}
        #define CLASSES {self.number_of_outputs}
        #define STATE_BITS {self.number_of_state_bits}
        #define MAX_TA_STATE {(1 << self.number_of_state_bits) - 1}
        #define ENCODE_LOC {self.encode_loc}
        """
        mod_new_kernel = SourceModule(self.gpu_macro_string + new_kernel, no_extern_c=True)

        self.kernel_init = mod_new_kernel.get_function("initialize")
        self.kernel_init.prepare("PPP")

        self.kernel_encode = mod_new_kernel.get_function("encode")
        self.kernel_encode.prepare("PPPi")

        self.kernel_clause_eval = mod_new_kernel.get_function("clause_eval")
        self.kernel_clause_eval.prepare("PPPPP")

        self.kernel_calc_class_sums = mod_new_kernel.get_function("calc_class_sums")
        self.kernel_calc_class_sums.prepare("PPP")

        self.kernel_calc_num_includes = mod_new_kernel.get_function("calc_num_includes")
        self.kernel_calc_num_includes.prepare("PP")

        self.kernel_clause_update = mod_new_kernel.get_function("clause_update")
        self.kernel_clause_update.prepare("PPPPPPPP")

        self.encode_config = kernel_config(self.number_of_patches, self.device_props)
        self.clause_eval_config = kernel_config(self.number_of_clauses * self.number_of_patches, self.device_props)
        self.calc_class_sums_config = kernel_config(self.number_of_clauses, self.device_props)
        self.calc_num_includes_config = kernel_config(self.number_of_clauses, self.device_props)
        self.clause_update_config = kernel_config(self.number_of_clauses, self.device_props)

        # Allocate GPU memory
        self.ta_state_gpu = mem_alloc(self.number_of_clauses * self.number_of_literals * 4)
        self.clause_weights_gpu = mem_alloc(self.number_of_clauses * self.number_of_outputs * 4)
        self.patch_weights_gpu = mem_alloc(self.number_of_clauses * self.number_of_patches * 4)

    #### STATES, WEIGHTS, AND INPUT INITIALIZATION ####
    def _init_default_vals(self):
        self.kernel_init.prepared_call(
            self.grid,
            self.block,
            self.rng_gpu.state,
            self.ta_state_gpu,
            self.clause_weights_gpu,
        )
        ctx.synchronize()

    def _calc_variables(self):
        if self.encode_loc:
            self.number_of_literals = int(
                self.patch_dim[0] * self.patch_dim[1] * self.dim[2]
                + (self.dim[0] - self.patch_dim[0])
                + (self.dim[1] - self.patch_dim[1])
            )
        else:
            self.number_of_literals = int(self.patch_dim[0] * self.patch_dim[1] * self.dim[2])

        if self.append_negated:
            self.number_of_literals *= 2

        if self.max_included_literals is None:
            self.max_included_literals = self.number_of_literals

        self.number_of_patches = int((self.dim[0] - self.patch_dim[0] + 1) * (self.dim[1] - self.patch_dim[1] + 1))

        if len(self.encoded_X_base) == 0:
            self._init_encoded_X_base()

    def _init_encoded_X_base(self):
        encoded_X = np.zeros((self.number_of_patches, self.number_of_literals), dtype=np.uint32)

        for patch_coordinate_y in range(self.dim[1] - self.patch_dim[1] + 1):
            for patch_coordinate_x in range(self.dim[0] - self.patch_dim[0] + 1):
                p = patch_coordinate_y * (self.dim[0] - self.patch_dim[0] + 1) + patch_coordinate_x

                if self.append_negated:
                    for k in range(self.number_of_literals // 2, self.number_of_literals):
                        encoded_X[p, k] = 1

                for y_threshold in range(self.dim[1] - self.patch_dim[1]):
                    patch_pos = y_threshold
                    if patch_coordinate_y > y_threshold:
                        encoded_X[p, patch_pos] = 1

                        if self.append_negated:
                            encoded_X[p, patch_pos + self.number_of_literals // 2] = 0

                for x_threshold in range(self.dim[0] - self.patch_dim[0]):
                    patch_pos = (self.dim[1] - self.patch_dim[1]) + x_threshold
                    if patch_coordinate_x > x_threshold:
                        encoded_X[p, patch_pos] = 1

                        if self.append_negated:
                            encoded_X[p, patch_pos + self.number_of_literals // 2] = 0

        self.encoded_X_base = encoded_X.reshape(-1)

    # def _init_encoded_X_packed_base(self):
    #     # Encoded X packed
    #     encoded_X_packed = np.zeros(((self.number_of_patches - 1) // 32 + 1, self.number_of_features), dtype=np.uint32)
    #     if self.append_negated:
    #         for p_chunk in range((self.number_of_patches - 1) // 32 + 1):
    #             for k in range(self.number_of_features // 2, self.number_of_features):
    #                 encoded_X_packed[p_chunk, k] = ~np.uint32(0)
    #
    #     for patch_coordinate_y in range(self.dim[1] - self.patch_dim[1] + 1):
    #         for patch_coordinate_x in range(self.dim[0] - self.patch_dim[0] + 1):
    #             p = patch_coordinate_y * (self.dim[0] - self.patch_dim[0] + 1) + patch_coordinate_x
    #             p_chunk = p // 32
    #             p_pos = p % 32
    #
    #             for y_threshold in range(self.dim[1] - self.patch_dim[1]):
    #                 patch_pos = y_threshold
    #                 if patch_coordinate_y > y_threshold:
    #                     encoded_X_packed[p_chunk, patch_pos] |= 1 << p_pos
    #
    #                     if self.append_negated:
    #                         encoded_X_packed[p_chunk, patch_pos + self.number_of_features // 2] &= ~np.uint32(
    #                             1 << p_pos
    #                         )
    #
    #             for x_threshold in range(self.dim[0] - self.patch_dim[0]):
    #                 patch_pos = (self.dim[1] - self.patch_dim[1]) + x_threshold
    #                 if patch_coordinate_x > x_threshold:
    #                     encoded_X_packed[p_chunk, patch_pos] |= 1 << p_pos
    #
    #                     if self.append_negated:
    #                         encoded_X_packed[p_chunk, patch_pos + self.number_of_features // 2] &= ~np.uint32(
    #                             1 << p_pos
    #                         )
    #
    #     self.encoded_X_packed_base = encoded_X_packed.reshape(-1)

    #### CAUSE and WEIGHT OPERATIONS ####
    def ta_action(self, clause, ta):
        if np.array_equal(self.ta_state, np.array([])):
            self.ta_state = np.empty(self.number_of_clauses * self.number_of_literals, dtype=np.uint32)
            memcpy_dtoh(self.ta_state, self.ta_state_gpu)
        ta_state = self.ta_state.reshape((self.number_of_clauses, self.number_of_literals))
        return ta_state[clause, ta]

    # def get_literals(self):
    #     literals_gpu = mem_alloc(self.number_of_clauses * self.number_of_features * 4)
    #     self.get_literals_gpu(
    #         self.ta_state_gpu,
    #         literals_gpu,
    #         grid=self.grid,
    #         block=self.block,
    #     )
    #     ctx.synchronize()
    #
    #     literals = np.empty((self.number_of_clauses * self.number_of_features), dtype=np.uint32)
    #     memcpy_dtoh(literals, literals_gpu)
    #     return literals.reshape((self.number_of_clauses, self.number_of_features)).astype(np.uint8)

    # def get_ta_states(self):
    #     ta_states_gpu = mem_alloc(self.number_of_clauses * self.number_of_features * 4)
    #     self.get_ta_states_gpu(
    #         self.ta_state_gpu,
    #         ta_states_gpu,
    #         grid=self.grid,
    #         block=self.block,
    #     )
    #     ctx.synchronize()
    #
    #     ta_states = np.empty((self.number_of_clauses * self.number_of_features), dtype=np.uint32)
    #     memcpy_dtoh(ta_states, ta_states_gpu)
    #     return ta_states.reshape((self.number_of_clauses, self.number_of_features))

    def get_weights(self):
        self.clause_weights = np.empty(self.number_of_clauses * self.number_of_outputs, dtype=np.int32)
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
    # def transform(self, X) -> csr_matrix:
    #     """Returns csr_matix of clause outputs. Array shape: (num_samples, num_clauses)"""
    #     if not self.initialized:
    #         print("Error: Model not trained.")
    #         sys.exit(-1)
    #
    #     if len(self.encoded_X_packed_base) == 0:
    #         self._init_encoded_X_packed_base()
    #
    #     if len(self.encoded_X_packed_base) == 0:
    #         self._init_encoded_X_packed_base()
    #
    #     X = csr_matrix(X)
    #     number_of_examples = X.shape[0]
    #
    #     # Copy data to GPU
    #     X_indptr_gpu = mem_alloc(X.indptr.nbytes)
    #     X_indices_gpu = mem_alloc(X.indices.nbytes)
    #     memcpy_htod(X_indptr_gpu, X.indptr)
    #     memcpy_htod(X_indices_gpu, X.indices)
    #
    #     X_transformed = np.empty((number_of_examples, self.number_of_clauses), dtype=np.uint32)
    #     X_transformed_gpu = mem_alloc(self.number_of_clauses * 4)
    #
    #     # Initialize GPU memory for temporary data
    #     encoded_X_packed_gpu = mem_alloc(self.encoded_X_packed_base.nbytes)
    #     included_literals_gpu = mem_alloc(self.number_of_clauses * self.number_of_features * 2 * 4)
    #     included_literals_length_gpu = mem_alloc(self.number_of_clauses * 4)
    #
    #     grid_prepare = (min(self.grid[0], (self.number_of_clauses + self.block[0] - 1) // self.block[0]), 1, 1)
    #     self.prepare_packed.prepared_call(
    #         grid_prepare,
    #         self.block,
    #         self.rng_gpu.state,
    #         self.ta_state_gpu,
    #         included_literals_gpu,
    #         included_literals_length_gpu,
    #     )
    #     ctx.synchronize()
    #
    #     for e in range(number_of_examples):
    #         memcpy_htod(encoded_X_packed_gpu, self.encoded_X_packed_base)
    #         self.encode_packed.prepared_call(
    #             self.grid,
    #             self.block,
    #             X_indptr_gpu,
    #             X_indices_gpu,
    #             encoded_X_packed_gpu,
    #             np.int32(e),
    #             np.int32(0),
    #         )
    #         ctx.synchronize()
    #
    #         self.transform_gpu(
    #             included_literals_gpu,
    #             included_literals_length_gpu,
    #             encoded_X_packed_gpu,
    #             X_transformed_gpu,
    #             grid=self.grid,
    #             block=self.block,
    #         )
    #         ctx.synchronize()
    #
    #         memcpy_dtoh(X_transformed[e, :], X_transformed_gpu)
    #
    #     encoded_X_packed_gpu.free()
    #     included_literals_gpu.free()
    #     included_literals_length_gpu.free()
    #
    #     X_transformed = (X_transformed > 0).astype(np.uint8)
    #     return csr_matrix(X_transformed)
    #
    # def transform_patchwise(self, X) -> csr_matrix:
    #     """Returns SPARSE CSR MATRIX of patch outputs for each clause. Array shape: (num_samples, num_clauses * num_patches)"""
    #     if not self.initialized:
    #         print("Error: Model not trained.")
    #         sys.exit(-1)
    #
    #     if len(self.encoded_X_packed_base) == 0:
    #         self._init_encoded_X_packed_base()
    #
    #     X = csr_matrix(X)
    #     number_of_examples = X.shape[0]
    #
    #     # Copy data to GPU
    #     X_indptr_gpu = mem_alloc(X.indptr.nbytes)
    #     X_indices_gpu = mem_alloc(X.indices.nbytes)
    #     memcpy_htod(X_indptr_gpu, X.indptr)
    #     memcpy_htod(X_indices_gpu, X.indices)
    #
    #     # Array to capture output from gpu
    #     X_transformed = np.empty(
    #         (number_of_examples, self.number_of_clauses * self.number_of_patches),
    #         dtype=np.uint32,
    #     )
    #     X_transformed_gpu = mem_alloc(self.number_of_clauses * self.number_of_patches * 4)
    #
    #     # Initialize GPU memory for temporary data
    #     encoded_X_packed_gpu = mem_alloc(self.encoded_X_packed_base.nbytes)
    #     included_literals_gpu = mem_alloc(self.number_of_clauses * self.number_of_features * 2 * 4)
    #     included_literals_length_gpu = mem_alloc(self.number_of_clauses * 4)
    #
    #     grid_prepare = (min(self.grid[0], (self.number_of_clauses + self.block[0] - 1) // self.block[0]), 1, 1)
    #     self.prepare_packed.prepared_call(
    #         grid_prepare,
    #         self.block,
    #         self.rng_gpu.state,
    #         self.ta_state_gpu,
    #         included_literals_gpu,
    #         included_literals_length_gpu,
    #     )
    #     ctx.synchronize()
    #
    #     for e in range(number_of_examples):
    #         memcpy_htod(encoded_X_packed_gpu, self.encoded_X_packed_base)
    #         self.encode_packed.prepared_call(
    #             self.grid,
    #             self.block,
    #             X_indptr_gpu,
    #             X_indices_gpu,
    #             encoded_X_packed_gpu,
    #             np.int32(e),
    #             np.int32(0),
    #         )
    #         ctx.synchronize()
    #
    #         self.transform_patchwise_gpu(
    #             included_literals_gpu,
    #             included_literals_length_gpu,
    #             encoded_X_packed_gpu,
    #             X_transformed_gpu,
    #             grid=self.grid,
    #             block=self.block,
    #         )
    #         ctx.synchronize()
    #
    #         memcpy_dtoh(X_transformed[e, :], X_transformed_gpu)
    #
    #     encoded_X_packed_gpu.free()
    #     included_literals_gpu.free()
    #     included_literals_length_gpu.free()
    #
    #     return csr_matrix(X_transformed.reshape((number_of_examples, self.number_of_clauses * self.number_of_patches)))

    #### SAVE AND LOAD ####
    def save(self, fname=""):
        # Copy data from GPU to CPU
        self.ta_state = np.empty(
            self.number_of_clauses * self.number_of_literals,
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
            "number_of_features": self.number_of_literals,
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

        self._calc_variables()
        self._gpu_init()

        memcpy_htod(self.ta_state_gpu, self.ta_state)
        memcpy_htod(self.clause_weights_gpu, self.clause_weights)
        memcpy_htod(self.patch_weights_gpu, self.patch_weights)

        self.initialized = True
