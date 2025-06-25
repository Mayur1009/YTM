import pathlib

import numpy as np
import pycuda.autoinit  # noqa: F401
import pycuda.curandom as curandom
from pycuda.compiler import SourceModule
from pycuda.driver import Context as ctx  # pyright: ignore[reportAttributeAccessIssue]
from pycuda.driver import mem_alloc, memcpy_dtoh, memcpy_htod, device_attribute, memset_d32  # pyright: ignore[reportAttributeAccessIssue]
from pycuda.gpuarray import to_gpu
from scipy.sparse import csr_array
from tqdm import tqdm
from line_profiler import profile

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
    # block_size = ((block_size + 31) // 32) * 32

    # Calculate grid size
    grid_size = (data_size + block_size - 1) // block_size

    # Limit grid size to reasonable bounds
    max_blocks = min(65535, props["multiprocessor_count"] * 4)
    grid_size = min(grid_size, max_blocks)

    return (grid_size, 1, 1), (block_size, 1, 1)


kernel_str = get_kernel("cuda/kernel.cu")


class BaseTM:
    def __init__(
        self,
        number_of_clauses: int,
        T: int,
        s: float,
        dim: tuple[int, int, int],
        n_classes: int,
        q: float = 1.0,
        patch_dim: tuple[int, int] | None = None,
        number_of_ta_states: int = 256,
        max_included_literals: int | None = None,
        append_negated: bool = True,
        init_neg_weights: bool = True,
        negative_polarity: bool = True,
        encode_loc: bool = True,
        seed: int | None = None,
        block_size: int = 128,
    ):
        # Initialize Hyperparams
        self.number_of_clauses = number_of_clauses
        self.number_of_clause_chunks = (number_of_clauses - 1) / 32 + 1
        self.number_of_ta_states = number_of_ta_states
        self.T = T
        self.s = s
        self.dim = dim
        self.number_of_outputs = n_classes
        self.q = q
        self.max_included_literals = max_included_literals
        self.append_negated = 1 if append_negated else 0
        self.encode_loc = 1 if encode_loc else 0

        if patch_dim is None:
            self.patch_dim = (dim[0], dim[1])
        else:
            self.patch_dim = (patch_dim[0], patch_dim[1])

        if seed is None:
            self.rng_gpu = curandom.XORWOWRandomNumberGenerator()
        else:

            def _custom_seed_getter(count):
                # Make sure that each thread gets a different seed
                return to_gpu(np.array([(seed + i) for i in range(1, count + 1)], dtype=np.int32))

            self.rng_gpu = curandom.XORWOWRandomNumberGenerator(_custom_seed_getter)

        self.init_neg_weights = 1 if init_neg_weights else 0
        self.negative_clauses = 1 if negative_polarity else 0
        self.coalesced = True  # TODO: Implement coalesced and non-coalesced versions
        self.initialized = False

        self.block_size = block_size
        self.device_props = get_device_properties()

    #### FIT AND SCORE ####
    @profile
    def _fit_batch(self, X: csr_array, encoded_Y):
        N = X.shape[0]

        if not self.initialized:
            self._calc_variables()
            self._gpu_init()
            self._init_default_vals()
            self.initialized = True

        X_indptr_gpu = mem_alloc(X.indptr.nbytes)
        X_indices_gpu = mem_alloc(X.indices.nbytes)
        encoded_Y_gpu = mem_alloc(encoded_Y.nbytes)
        memcpy_htod(X_indptr_gpu, X.indptr)
        memcpy_htod(X_indices_gpu, X.indices)
        memcpy_htod(encoded_Y_gpu, encoded_Y)

        # Initialize GPU memory for temporary data
        packed_clauses_gpu = mem_alloc(self.number_of_clauses * self.number_of_literal_chunks * 4)
        class_sum_gpu = mem_alloc(self.number_of_outputs * 4)
        selected_patch_ids_gpu = mem_alloc(self.number_of_clauses * 4)
        num_includes_gpu = mem_alloc(self.number_of_clauses * 4)

        # Encode all the samples in the batch
        encoded_X_gpu = mem_alloc(N * self.number_of_patches * self.number_of_literal_chunks * 4)
        memcpy_htod(encoded_X_gpu, np.stack([self.encoded_X_base] * N, axis=0).reshape(-1))
        self.kernel_encode_batch.prepared_call(
            *kernel_config(N * self.number_of_patches, self.device_props, self.block_size),
            X_indptr_gpu,
            X_indices_gpu,
            encoded_X_gpu,
            np.int32(N),
        )
        ctx.synchronize()

        pbar = tqdm(range(N), desc="Fitting Batch", leave=False, dynamic_ncols=True)
        for e in pbar:
            self.kernel_pack_clauses.prepared_call(
                *self.pack_clauses_config,
                self.ta_state_gpu,
                packed_clauses_gpu,
                num_includes_gpu,
            )
            ctx.synchronize()

            memset_d32(selected_patch_ids_gpu, 0xFFFFFFFF, self.number_of_clauses)  # Initialize with -1
            self.kernel_clause_eval.prepared_call(
                *self.clause_eval_config,
                self.rng_gpu.state,
                packed_clauses_gpu,
                self.clause_weights_gpu,
                encoded_X_gpu,
                selected_patch_ids_gpu,
                np.int32(e),
            )
            ctx.synchronize()

            memset_d32(class_sum_gpu, 0, self.number_of_outputs)
            self.kernel_calc_class_sums.prepared_call(
                *self.calc_class_sums_config,
                selected_patch_ids_gpu,
                self.clause_weights_gpu,
                class_sum_gpu,
            )
            ctx.synchronize()

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
                np.int32(e),
            )
            ctx.synchronize()
        return

    @profile
    def _score_batch(self, X) -> np.ndarray[tuple[int, int], np.dtype[np.float32]]:
        if not self.initialized:
            print("Error: Model not trained.")
            raise RuntimeError("Model not trained. Call fit() before score().")

        N = X.shape[0]
        if not hasattr(self, "encoded_X_base"):
            self._init_encoded_X_base()

        X_indptr_gpu = mem_alloc(X.indptr.nbytes)
        X_indices_gpu = mem_alloc(X.indices.nbytes)
        memcpy_htod(X_indptr_gpu, X.indptr)
        memcpy_htod(X_indices_gpu, X.indices)

        # Initialize GPU memory for temporary data
        packed_clauses_gpu = mem_alloc(self.number_of_clauses * self.number_of_literal_chunks * 4)
        includes_gpu = mem_alloc(self.number_of_clauses * 4)
        class_sum_gpu = mem_alloc(self.number_of_outputs * 4)
        selected_patch_ids_gpu = mem_alloc(self.number_of_clauses * 4)

        encoded_X_gpu = mem_alloc(N * self.number_of_patches * self.number_of_literal_chunks * 4)
        memcpy_htod(encoded_X_gpu, np.stack([self.encoded_X_base] * N, axis=0).reshape(-1))
        self.kernel_encode_batch.prepared_call(
            *kernel_config(N * self.number_of_patches, self.device_props, self.block_size),
            X_indptr_gpu,
            X_indices_gpu,
            encoded_X_gpu,
            np.int32(N),
        )
        ctx.synchronize()

        class_sums = np.zeros((X.shape[0], self.number_of_outputs), dtype=np.float32)

        self.kernel_pack_clauses.prepared_call(
            *self.pack_clauses_config,
            self.ta_state_gpu,
            packed_clauses_gpu,
            includes_gpu,
        )
        ctx.synchronize()

        for e in tqdm(range(X.shape[0]), leave=False, desc="Score Batch"):
            memset_d32(selected_patch_ids_gpu, 0xFFFFFFFF, self.number_of_clauses)  # Initialize with -1
            self.kernel_clause_eval_infer.prepared_call(
                *self.clause_eval_config,
                self.rng_gpu.state,
                packed_clauses_gpu,
                self.clause_weights_gpu,
                encoded_X_gpu,
                selected_patch_ids_gpu,
                np.int32(e),
            )
            ctx.synchronize()

            memset_d32(class_sum_gpu, 0, self.number_of_outputs)
            self.kernel_calc_class_sums.prepared_call(
                *self.calc_class_sums_config,
                selected_patch_ids_gpu,
                self.clause_weights_gpu,
                class_sum_gpu,
            )
            ctx.synchronize()

            memcpy_dtoh(class_sums[e, :], class_sum_gpu)

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
        #define INIT_NEG_WEIGHTS {self.init_neg_weights}
        #define NEGATIVE_CLAUSES {self.negative_clauses}
        #define CLASSES {self.number_of_outputs}
        #define MAX_TA_STATE {self.number_of_ta_states}
        #define ENCODE_LOC {self.encode_loc}
        """
        mod_new_kernel = SourceModule(self.gpu_macro_string + kernel_str, no_extern_c=True)

        self.kernel_init = mod_new_kernel.get_function("initialize")
        self.kernel_init.prepare("PPP")

        self.kernel_encode_batch = mod_new_kernel.get_function("encode_batch")
        self.kernel_encode_batch.prepare("PPPi")

        self.kernel_pack_clauses = mod_new_kernel.get_function("pack_clauses")
        self.kernel_pack_clauses.prepare("PPP")

        self.kernel_clause_eval = mod_new_kernel.get_function("clause_eval")
        self.kernel_clause_eval.prepare("PPPPPi")

        self.kernel_clause_eval_infer = mod_new_kernel.get_function("clause_eval_infer")
        self.kernel_clause_eval_infer.prepare("PPPPPi")

        self.kernel_calc_class_sums = mod_new_kernel.get_function("calc_class_sums")
        self.kernel_calc_class_sums.prepare("PPP")

        self.kernel_clause_update = mod_new_kernel.get_function("clause_update")
        self.kernel_clause_update.prepare("PPPPPPPPi")

        self.initialize_config = kernel_config(
            self.number_of_clauses,
            self.device_props,
            self.block_size,
        )
        self.encode_config = kernel_config(
            self.number_of_patches,
            self.device_props,
            self.block_size,
        )
        self.pack_clauses_config = kernel_config(
            self.number_of_clauses,
            self.device_props,
            self.block_size,
        )
        self.clause_eval_config = kernel_config(
            self.number_of_clauses,
            self.device_props,
            self.block_size,
        )
        self.calc_class_sums_config = kernel_config(
            self.number_of_clauses,
            self.device_props,
            self.block_size,
        )
        self.clause_update_config = kernel_config(
            self.number_of_clauses,
            self.device_props,
            self.block_size,
        )

        # Allocate GPU memory
        self.ta_state_gpu = mem_alloc(self.number_of_clauses * self.number_of_literals * 4)
        self.clause_weights_gpu = mem_alloc(self.number_of_clauses * self.number_of_outputs * 4)
        self.patch_weights_gpu = mem_alloc(self.number_of_clauses * self.number_of_patches * 4)

    #### STATES, WEIGHTS, AND INPUT INITIALIZATION ####
    def _init_default_vals(self):
        self.kernel_init.prepared_call(
            *self.initialize_config,
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

        self.number_of_literal_chunks = ((self.number_of_literals - 1) // 32) + 1

        if self.max_included_literals is None:
            self.max_included_literals = self.number_of_literals

        self.number_of_patches = int((self.dim[0] - self.patch_dim[0] + 1) * (self.dim[1] - self.patch_dim[1] + 1))

        if not hasattr(self, "encoded_X_base"):
            self._init_encoded_X_base()

    def _init_encoded_X_base(self):
        encoded_X = np.zeros((self.number_of_patches, self.number_of_literal_chunks), dtype=np.uint32)
        for patch_coordinate_y in range(self.dim[1] - self.patch_dim[1] + 1):
            for patch_coordinate_x in range(self.dim[0] - self.patch_dim[0] + 1):
                p = patch_coordinate_y * (self.dim[0] - self.patch_dim[0] + 1) + patch_coordinate_x

                if self.append_negated:
                    for k in range(self.number_of_literals // 2, self.number_of_literals):
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
                            chunk = (patch_pos + self.number_of_literals // 2) // 32
                            pos = (patch_pos + self.number_of_literals // 2) % 32
                            encoded_X[p, chunk] &= ~np.uint32(1 << pos)

                for x_threshold in range(self.dim[0] - self.patch_dim[0]):
                    patch_pos = (self.dim[1] - self.patch_dim[1]) + x_threshold
                    if patch_coordinate_x > x_threshold:
                        chunk = patch_pos // 32
                        pos = patch_pos % 32
                        encoded_X[p, chunk] |= 1 << pos

                        if self.append_negated:
                            chunk = (patch_pos + self.number_of_literals // 2) // 32
                            pos = (patch_pos + self.number_of_literals // 2) % 32
                            encoded_X[p, chunk] &= ~np.uint32(1 << pos)

        self.encoded_X_base = encoded_X.reshape(-1)

    #### CAUSE and WEIGHT OPERATIONS ####
    def get_ta_state(self):
        self.ta_state = np.empty(self.number_of_clauses * self.number_of_literals, dtype=np.uint32)
        memcpy_dtoh(self.ta_state, self.ta_state_gpu)
        return self.ta_state.reshape((self.number_of_clauses, self.number_of_literals))

    def get_weights(self):
        self.clause_weights = np.empty(self.number_of_clauses * self.number_of_outputs, dtype=np.float32)
        memcpy_dtoh(self.clause_weights, self.clause_weights_gpu)

        return self.clause_weights.reshape((self.number_of_clauses, self.number_of_outputs))

    def __getstate__(self) -> object:
        pass

    def __setstate__(self, state: object) -> None:
        pass

    #### SAVE AND LOAD ####
    def save(self):
        # Copy data from GPU to CPU
        self.ta_state = np.empty(
            self.number_of_clauses * self.number_of_literals,
            dtype=np.uint32,
        )
        self.clause_weights = np.empty(self.number_of_clauses * self.number_of_outputs, dtype=np.float32)

        memcpy_dtoh(self.ta_state, self.ta_state_gpu)
        memcpy_dtoh(self.clause_weights, self.clause_weights_gpu)

        state_dict = {
            # State arrays
            "ta_state": self.ta_state,
            "clause_weights": self.clause_weights,
            "number_of_outputs": self.number_of_outputs,
            "number_of_literals": self.number_of_literals,
            "min_y": self.min_y,
            "max_y": self.max_y,
            "negative_clauses": self.negative_clauses,
            # Parameters
            "number_of_clauses": self.number_of_clauses,
            "T": self.T,
            "s": self.s,
            "q": self.q,
            "patch_dim": self.patch_dim,
            "dim": self.dim,
            "max_included_literals": self.max_included_literals,
            "number_of_ta_states": self.number_of_ta_states,
            "append_negated": self.append_negated,
            "encode_loc": self.encode_loc,
        }

        return state_dict

    def load(self, state_dict):
        # Load arrays state_dict
        self.ta_state = state_dict["ta_state"]
        self.clause_weights = state_dict["clause_weights"]
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

        self.initialized = True
