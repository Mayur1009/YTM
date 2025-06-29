import pathlib

import numpy as np
import pycuda.autoinit  # noqa: F401
import pycuda.curandom as curandom
from pycuda.compiler import SourceModule
from pycuda.driver import Context as ctx  # pyright: ignore[reportAttributeAccessIssue]
from pycuda.driver import mem_alloc, memcpy_dtoh, memcpy_htod, memset_d32  # pyright: ignore[reportAttributeAccessIssue]
from pycuda.gpuarray import to_gpu
from scipy.sparse import csr_array
from tqdm import tqdm
from line_profiler import profile

from .cuda_utils import kernel_config, get_kernel, device_props


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

        self.seed = seed
        if seed is None:
            self.rng_gpu = curandom.XORWOWRandomNumberGenerator()
        else:

            def _custom_seed_getter(count):
                # Make sure that each thread gets a different seed
                return to_gpu(np.array([(seed + i) for i in range(1, count + 1)], dtype=np.int32))

            self.rng_gpu = curandom.XORWOWRandomNumberGenerator(_custom_seed_getter)

        self.init_neg_weights = 1 if init_neg_weights else 0
        self.negative_clauses = 1 if negative_polarity else 0
        self.initialized = False

        self.block_size = block_size

        self._calc_variables()
        self._gpu_init()
        self._init_default_vals(self.block_size)
        self.initialized = True

    @profile
    def encode(
        self,
        X: np.ndarray[tuple[int, int], np.dtype[np.uint32]],
        block_size: int | None = None,
    ) -> np.ndarray[tuple[int, int, int], np.dtype[np.uint32]]:
        assert X.ndim == 2, "X must be a 2D array (samples, dim0 * dim1 * dim2)."
        N = X.shape[0]

        max_uint32 = np.iinfo(np.uint32).max
        max_safe_N = max_uint32 // self.number_of_patches
        if N > max_safe_N:
            raise OverflowError(
                f"X has too many samples ({N}). Maximum of {max_safe_N} samples can be processed with current number_of_patches. Call this method multiple times with smaller batches of X."
            )

        if block_size is None:
            block_size = self.block_size

        if not hasattr(self, "encoded_X_base"):
            self._init_encoded_X_base()

        csrX = csr_array(X.astype(np.uint32))
        X_indptr_gpu = mem_alloc(csrX.indptr.nbytes)
        X_indices_gpu = mem_alloc(csrX.indices.nbytes)
        memcpy_htod(X_indptr_gpu, csrX.indptr)
        memcpy_htod(X_indices_gpu, csrX.indices)

        encoded_X_gpu = mem_alloc(N * self.number_of_patches * self.number_of_literal_chunks * 4)
        memcpy_htod(encoded_X_gpu, np.stack([self.encoded_X_base] * N, axis=0).reshape(-1))
        self.kernel_encode_batch.prepared_call(
            *kernel_config(N * self.number_of_patches, device_props, block_size),
            X_indptr_gpu,
            X_indices_gpu,
            encoded_X_gpu,
            np.int32(N),
        )
        ctx.synchronize()
        encoded_X = np.empty((N * self.number_of_patches * self.number_of_literal_chunks), dtype=np.uint32)
        memcpy_dtoh(encoded_X, encoded_X_gpu)

        X_indptr_gpu.free()
        X_indices_gpu.free()
        encoded_X_gpu.free()

        return encoded_X.reshape((N, self.number_of_patches, self.number_of_literal_chunks))

    #### FIT AND SCORE ####
    @profile
    def _fit_batch(self, encoded_X, encoded_Y, block_size: int | None = None):
        N = encoded_X.shape[0]
        encoded_X_gpu = mem_alloc(encoded_X.nbytes)
        memcpy_htod(encoded_X_gpu, encoded_X)
        encoded_Y_gpu = mem_alloc(encoded_Y.nbytes)
        memcpy_htod(encoded_Y_gpu, encoded_Y)

        # Initialize GPU memory for temporary data
        packed_clauses_gpu = mem_alloc(self.number_of_clauses * self.number_of_literal_chunks * 4)
        class_sum_gpu = mem_alloc(self.number_of_outputs * 4)
        selected_patch_ids_gpu = mem_alloc(self.number_of_clauses * 4)
        num_includes_gpu = mem_alloc(self.number_of_clauses * 4)

        if block_size is None:
            block_size = self.block_size

        config_n_clauses = kernel_config(self.number_of_clauses, device_props, block_size)

        pbar = tqdm(range(N), desc="Fitting Batch", leave=False, dynamic_ncols=True)
        for e in pbar:
            self.kernel_pack_clauses.prepared_call(
                *config_n_clauses,
                self.ta_state_gpu,
                packed_clauses_gpu,
                num_includes_gpu,
            )
            ctx.synchronize()

            memset_d32(selected_patch_ids_gpu, 0xFFFFFFFF, self.number_of_clauses)  # Initialize with -1
            self.kernel_clause_eval.prepared_call(
                *config_n_clauses,
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
                *config_n_clauses,
                selected_patch_ids_gpu,
                self.clause_weights_gpu,
                class_sum_gpu,
            )
            ctx.synchronize()

            self.kernel_clause_update.prepared_call(
                *config_n_clauses,
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

        encoded_X_gpu.free()
        encoded_Y_gpu.free()
        packed_clauses_gpu.free()
        class_sum_gpu.free()
        selected_patch_ids_gpu.free()
        num_includes_gpu.free()
        return

    @profile
    def _score_batch(self, encoded_X, block_size: int | None = None) -> np.ndarray[tuple[int, int], np.dtype[np.float32]]:
        N = encoded_X.shape[0]
        max_uint32 = np.iinfo(np.uint32).max
        max_safe_N = max_uint32 // self.number_of_clauses
        if N > max_safe_N:
            raise OverflowError(
                f"X has too many samples ({N}). Maximum of {max_safe_N} samples can be processed with current number_of_clauses. Call this method multiple times with smaller batches of X."
            )

        encoded_X_gpu = mem_alloc(N * self.number_of_patches * self.number_of_literal_chunks * 4)
        memcpy_htod(encoded_X_gpu, encoded_X)

        # Initialize GPU memory for temporary data
        packed_clauses_gpu = mem_alloc(self.number_of_clauses * self.number_of_literal_chunks * 4)
        includes_gpu = mem_alloc(self.number_of_clauses * 4)
        class_sums_gpu = mem_alloc(N * self.number_of_outputs * 4)

        if block_size is None:
            block_size = self.block_size

        self.kernel_pack_clauses.prepared_call(
            *kernel_config(self.number_of_clauses, device_props, block_size),
            self.ta_state_gpu,
            packed_clauses_gpu,
            includes_gpu,
        )
        ctx.synchronize()

        memset_d32(class_sums_gpu, 0, N * self.number_of_outputs)
        self.kernel_calc_class_sums_infer_batch.prepared_call(
            *kernel_config(N * self.number_of_clauses, device_props, block_size),
            packed_clauses_gpu,
            self.clause_weights_gpu,
            includes_gpu,
            encoded_X_gpu,
            np.int32(N),
            class_sums_gpu,
        )
        ctx.synchronize()

        class_sums = np.zeros((N, self.number_of_outputs), dtype=np.float32)
        memcpy_dtoh(class_sums, class_sums_gpu)

        encoded_X_gpu.free()
        packed_clauses_gpu.free()
        includes_gpu.free()
        class_sums_gpu.free()

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
        current_dir = pathlib.Path(__file__).parent
        kernel_str = get_kernel("cuda/kernel.cu", current_dir)
        mod_new_kernel = SourceModule(self.gpu_macro_string + kernel_str, no_extern_c=True)

        self.kernel_init = mod_new_kernel.get_function("initialize")
        self.kernel_init.prepare("PPP")

        self.kernel_encode_batch = mod_new_kernel.get_function("encode_batch")
        self.kernel_encode_batch.prepare("PPPi")

        self.kernel_pack_clauses = mod_new_kernel.get_function("pack_clauses")
        self.kernel_pack_clauses.prepare("PPP")

        self.kernel_clause_eval = mod_new_kernel.get_function("clause_eval")
        self.kernel_clause_eval.prepare("PPPPPi")

        self.kernel_calc_class_sums = mod_new_kernel.get_function("calc_class_sums")
        self.kernel_calc_class_sums.prepare("PPP")

        self.kernel_calc_class_sums_infer_batch = mod_new_kernel.get_function("calc_class_sums_infer_batch")
        self.kernel_calc_class_sums_infer_batch.prepare("PPPPiP")

        self.kernel_clause_update = mod_new_kernel.get_function("clause_update")
        self.kernel_clause_update.prepare("PPPPPPPPi")

        # self.initialize_config = kernel_config(
        #     self.number_of_clauses,
        #     device_props,
        #     self.block_size,
        # )
        # self.pack_clauses_config = kernel_config(
        #     self.number_of_clauses,
        #     device_props,
        #     self.block_size,
        # )
        # self.clause_eval_config = kernel_config(
        #     self.number_of_clauses,
        #     device_props,
        #     self.block_size,
        # )
        # self.calc_class_sums_config = kernel_config(
        #     self.number_of_clauses,
        #     device_props,
        #     self.block_size,
        # )
        # self.clause_update_config = kernel_config(
        #     self.number_of_clauses,
        #     device_props,
        #     self.block_size,
        # )

        # Allocate GPU memory
        self.ta_state_gpu = mem_alloc(self.number_of_clauses * self.number_of_literals * 4)
        self.clause_weights_gpu = mem_alloc(self.number_of_clauses * self.number_of_outputs * 4)

    #### STATES, WEIGHTS, AND INPUT INITIALIZATION ####
    def _init_default_vals(self, block_size: int = 128):
        self.kernel_init.prepared_call(
            *kernel_config(self.number_of_clauses, device_props, block_size),
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

    ## SERIALIZATION ##
    def __getstate__(self):
        args = {
            "number_of_clauses": self.number_of_clauses,
            "T": self.T,
            "s": self.s,
            "dim": self.dim,
            "n_classes": self.number_of_outputs,
            "q": self.q,
            "patch_dim": getattr(self, "patch_dim", None),
            "number_of_ta_states": self.number_of_ta_states,
            "max_included_literals": self.max_included_literals,
            "append_negated": bool(self.append_negated),
            "init_neg_weights": bool(self.init_neg_weights),
            "negative_polarity": bool(self.negative_clauses),
            "encode_loc": bool(self.encode_loc),
            "seed": self.seed,
            "block_size": self.block_size,
        }
        if not self.initialized:
            return {"initialized": False, "args": args, "state": None}

        state_dict = self.get_state_dict()
        return {"initialized": True, "args": args, "state": state_dict}

    def __setstate__(self, state):
        args = state["args"]
        self.__init__(
            number_of_clauses=args["number_of_clauses"],
            T=args["T"],
            s=args["s"],
            dim=args["dim"],
            n_classes=args["n_classes"],
            q=args["q"],
            patch_dim=args["patch_dim"],
            number_of_ta_states=args["number_of_ta_states"],
            max_included_literals=args["max_included_literals"],
            append_negated=args["append_negated"],
            init_neg_weights=args["init_neg_weights"],
            negative_polarity=args["negative_polarity"],
            encode_loc=args["encode_loc"],
            seed=args["seed"],
            block_size=args["block_size"],
        )
        initialized = state["initialized"]
        if initialized:
            self.load_state_dict(state)

    #### SAVE AND LOAD ####
    def get_state_dict(self):
        # Copy data from GPU to CPU
        self.ta_state = np.empty(
            self.number_of_clauses * self.number_of_literals,
            dtype=np.uint32,
        )
        self.clause_weights = np.empty(self.number_of_clauses * self.number_of_outputs, dtype=np.float32)

        memcpy_dtoh(self.ta_state, self.ta_state_gpu)
        memcpy_dtoh(self.clause_weights, self.clause_weights_gpu)
        state_dict = {
            "ta_state": self.ta_state,
            "clause_weights": self.clause_weights,
            "min_y": self.min_y,
            "max_y": self.max_y,
        }

        return state_dict

    def load_state_dict(self, state):
        state_dict = state["state"]
        self.ta_state = state_dict["ta_state"]
        self.clause_weights = state_dict["clause_weights"]
        self.min_y = state_dict["min_y"]
        self.max_y = state_dict["max_y"]

        # self._calc_variables()
        # self._gpu_init()

        memcpy_htod(self.ta_state_gpu, self.ta_state)
        memcpy_htod(self.clause_weights_gpu, self.clause_weights)

        # self.initialized = True
