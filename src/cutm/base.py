import pathlib

import numpy as np
import pycuda.autoinit  # noqa: F401
import pycuda.curandom as curandom
from pycuda.compiler import SourceModule
from pycuda.driver import Context as ctx  # pyright: ignore[reportAttributeAccessIssue]
from pycuda.driver import mem_alloc, memcpy_dtoh, memcpy_htod, memset_d32  # pyright: ignore[reportAttributeAccessIssue]
from pycuda.gpuarray import to_gpu
from tqdm import tqdm

from .cuda_utils import kernel_config, get_kernel, device_props


class BaseTM:
    """
    Base class for TM

    Attributes
    ----------
    number_of_clauses : [int]
        Number of clauses per class.
    T : [int]
        Target value
    s : [float]
        Specificity value
    dim : tuple[int, int, int]
        Dimensions of the input data (dim0, dim1, dim2).
    n_classes : int
        Number of classes.
    q : float, optional
        Q value, default is 1.0.
    patch_dim : tuple[int, int] | None, optional
        Dimensions of the patches (patch_dim0, patch_dim1). If None, defaults to (dim0, dim1).
    number_of_ta_states : int, optional
        Number of TA states, default is 256.
    max_included_literals : int | None, optional
        Maximum number of included literals per clause. If None, defaults to the number of literals.
    append_negated : bool, optional
        Whether to append negated literals, default is True.
    init_neg_weights : bool, optional
        Whether to initialize negative weights, default is True.
    negative_polarity : bool, optional
        Whether to use negative polarity for clauses, default is True.
    encode_loc : bool, optional
        Whether to encode location information in the literals, default is True. Only relevant if `patch_dim` is not None.
    coalesced : bool, optional
        Wheather to use coalesced clause banks, default is True.
    seed : int | None, optional
        Random seed
    block_size : int, optional
        Block size for CUDA kernels, default is 128.
    """
    def __init__(
        self,
        number_of_clauses_per_class: int,
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
        coalesced: bool = True,
        seed: int | None = None,
        block_size: int = 128,
    ):
        # Initialize Hyperparams
        self.number_of_clauses_per_class = number_of_clauses_per_class
        self.number_of_ta_states = number_of_ta_states
        self.T = T
        self.s = s
        self.dim = dim
        self.number_of_outputs = n_classes
        self.q = q
        self.max_included_literals = max_included_literals
        self.append_negated = 1 if append_negated else 0
        self.encode_loc = 1 if encode_loc else 0
        self.coalesced = coalesced
        self.number_of_clause_banks = 1 if coalesced else self.number_of_outputs
        self.number_of_clauses = self.number_of_clause_banks * self.number_of_clauses_per_class

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

    def encode(
        self,
        X: np.ndarray[tuple[int, int], np.dtype[np.uint32]],
        block_size: int | None = None,
    ) -> np.ndarray[tuple[int, int, int], np.dtype[np.uint32]]:
        assert X.ndim == 2, "X must be a 2D array (samples, dim0 * dim1 * dim2)."
        assert X.dtype == np.uint32, "X must be of type np.uint32."
        N = X.shape[0]
        if block_size is None:
            block_size = self.block_size

        max_uint32 = np.iinfo(np.uint32).max
        max_safe_N = max_uint32 // self.number_of_patches

        encoded_X = np.empty((N, self.number_of_patches, self.number_of_literal_chunks), dtype=np.uint32)
        for i in range(0, N, max_safe_N):
            X_safe = X[i : i + max_safe_N]
            X_gpu = mem_alloc(X_safe.nbytes)
            memcpy_htod(X_gpu, X_safe)

            encoded_X_gpu = mem_alloc(X_safe.shape[0] * self.number_of_patches * self.number_of_literal_chunks * 4)
            memset_d32(encoded_X_gpu, 0, X_safe.shape[0] * self.number_of_patches * self.number_of_literal_chunks)
            self.kernel_encode_batch.prepared_call(
                *kernel_config(X_safe.shape[0] * self.number_of_patches, device_props, block_size),
                X_gpu,
                encoded_X_gpu,
                np.int32(X_safe.shape[0]),
            )
            ctx.synchronize()

            encoded_X_safe = np.empty(
                (X_safe.shape[0] * self.number_of_patches * self.number_of_literal_chunks), dtype=np.uint32
            )
            memcpy_dtoh(encoded_X_safe, encoded_X_gpu)

            encoded_X[i : i + max_safe_N] = encoded_X_safe.reshape(
                (X_safe.shape[0], self.number_of_patches, self.number_of_literal_chunks)
            )

        return encoded_X

    #### FIT AND SCORE ####
    def _fit(self, encoded_X, encoded_Y, block_size: int | None = None, **kwargs):
        N = encoded_X.shape[0]
        encoded_X_gpu = mem_alloc(encoded_X.nbytes)
        encoded_Y_gpu = mem_alloc(encoded_Y.nbytes)
        memcpy_htod(encoded_X_gpu, encoded_X)
        memcpy_htod(encoded_Y_gpu, encoded_Y)

        # Calculate imbalance modifiers
        balance = kwargs.get("balance", False)
        if balance:
            true_cnt = (encoded_Y > 0).sum(axis=0)
            false_cnt = (encoded_Y <= 0).sum(axis=0)
            true_mod = np.asarray(true_cnt / true_cnt.mean(), dtype=np.float64)
            false_mod = np.asarray(false_cnt / (max(1, self.number_of_outputs - 1) * true_cnt), dtype=np.float64)
        else:
            true_mod = np.ones(self.number_of_outputs, dtype=np.float64)
            false_mod = np.ones(self.number_of_outputs, dtype=np.float64)
        true_mod_gpu = mem_alloc(true_mod.nbytes)
        false_mod_gpu = mem_alloc(false_mod.nbytes)
        memcpy_htod(true_mod_gpu, true_mod)
        memcpy_htod(false_mod_gpu, false_mod)


        # Initialize GPU memory for temporary data
        packed_clauses_gpu = mem_alloc(self.number_of_clauses * self.number_of_literal_chunks * 4)
        class_sum_gpu = mem_alloc(self.number_of_outputs * 4)
        clause_outputs_gpu = mem_alloc(self.number_of_clauses * self.number_of_patches * 4)
        selected_patch_ids_gpu = mem_alloc(self.number_of_clauses * 4)
        num_includes_gpu = mem_alloc(self.number_of_clauses * 4)

        if block_size is None:
            block_size = self.block_size

        config_n_clauses = kernel_config(self.number_of_clauses, device_props, block_size)
        config_patchwise = kernel_config(self.number_of_clauses * self.number_of_patches, device_props, block_size)

        pbar = tqdm(range(N), desc="Fitting Batch", leave=False, dynamic_ncols=True)
        for e in pbar:
            self.kernel_pack_clauses.prepared_call(
                *config_n_clauses,
                self.ta_state_gpu,
                packed_clauses_gpu,
                num_includes_gpu,
            )
            ctx.synchronize()

            self.kernel_fast_eval.prepared_call(
                *config_patchwise,
                packed_clauses_gpu,
                num_includes_gpu,
                encoded_X_gpu,
                clause_outputs_gpu,
                np.int32(e),
            )
            ctx.synchronize()

            memset_d32(class_sum_gpu, 0, self.number_of_outputs)
            self.kernel_select_active.prepared_call(
                *config_n_clauses,
                self.rng_gpu.state,
                self.clause_weights_gpu,
                clause_outputs_gpu,
                self.patch_weights_gpu,
                selected_patch_ids_gpu,
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
                true_mod_gpu,
                false_mod_gpu,
                encoded_X_gpu,
                encoded_Y_gpu,
                np.int32(e),
            )
            ctx.synchronize()

        return

    def _pack_clauses_gpu(self):
        packed_clauses_gpu = mem_alloc(self.number_of_clauses * self.number_of_literal_chunks * 4)
        includes_gpu = mem_alloc(self.number_of_clauses * 4)

        self.kernel_pack_clauses.prepared_call(
            *kernel_config(self.number_of_clauses, device_props, self.block_size),
            self.ta_state_gpu,
            packed_clauses_gpu,
            includes_gpu,
        )
        ctx.synchronize()

        return packed_clauses_gpu, includes_gpu

    def _score_batch(
        self, encoded_X, block_size: int | None = None
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float32]]:
        N = encoded_X.shape[0]

        if block_size is None:
            block_size = self.block_size

        max_uint32 = np.iinfo(np.uint32).max
        max_safe_N = max_uint32 // self.number_of_clauses

        packed_clauses_gpu, includes_gpu = self._pack_clauses_gpu()
        class_sums = np.zeros((N, self.number_of_outputs), dtype=np.float32)
        for i in range(0, N, max_safe_N):
            X_safe = encoded_X[i : i + max_safe_N]
            X_gpu = mem_alloc(X_safe.nbytes)
            memcpy_htod(X_gpu, X_safe)

            class_sums_gpu = mem_alloc(X_safe.shape[0] * self.number_of_outputs * 4)
            memset_d32(class_sums_gpu, 0, X_safe.shape[0] * self.number_of_outputs)
            self.kernel_calc_class_sums_infer_batch.prepared_call(
                *kernel_config(X_safe.shape[0] * self.number_of_clauses, device_props, block_size),
                packed_clauses_gpu,
                self.clause_weights_gpu,
                includes_gpu,
                X_gpu,
                np.int32(X_safe.shape[0]),
                class_sums_gpu,
            )
            ctx.synchronize()

            class_sums_safe = np.empty((X_safe.shape[0], self.number_of_outputs), dtype=np.float32)
            memcpy_dtoh(class_sums_safe, class_sums_gpu)

            class_sums[i : i + max_safe_N] = class_sums_safe

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
        #define COALESCED {1 if self.coalesced else 0}
        #define CLAUSE_BANKS {self.number_of_clause_banks}
        """
        current_dir = pathlib.Path(__file__).parent
        kernel_str = get_kernel("cuda/kernel.cu", current_dir)
        mod_new_kernel = SourceModule(
            self.gpu_macro_string + kernel_str,
            options=["-O3", "--use_fast_math"],
            no_extern_c=True,
        )

        self.kernel_init = mod_new_kernel.get_function("initialize")
        self.kernel_init.prepare("PPP")

        self.kernel_encode_batch = mod_new_kernel.get_function("encode_batch")
        self.kernel_encode_batch.prepare("PPi")

        self.kernel_pack_clauses = mod_new_kernel.get_function("pack_clauses")
        self.kernel_pack_clauses.prepare("PPP")

        self.kernel_fast_eval = mod_new_kernel.get_function("fast_eval")
        self.kernel_fast_eval.prepare("PPPPi")

        self.kernel_select_active = mod_new_kernel.get_function("select_active")
        self.kernel_select_active.prepare("PPPPPP")

        self.kernel_calc_class_sums_infer_batch = mod_new_kernel.get_function("calc_class_sums_infer_batch")
        self.kernel_calc_class_sums_infer_batch.prepare("PPPPiP")

        self.kernel_clause_update = mod_new_kernel.get_function("clause_update")
        self.kernel_clause_update.prepare("PPPPPPPPPPi")

        self.kernel_transform = mod_new_kernel.get_function("transform")
        self.kernel_transform.prepare("PPPiP")

        self.kernel_transform_patchwise = mod_new_kernel.get_function("transform_patchwise")
        self.kernel_transform_patchwise.prepare("PPPiP")

        # Allocate GPU memory
        self.ta_state_gpu = mem_alloc(self.number_of_clauses * self.number_of_literals * 4)
        self.clause_weights_gpu = mem_alloc(self.number_of_clauses * self.number_of_outputs * 4)
        self.patch_weights_gpu = mem_alloc(self.number_of_clauses * self.number_of_patches * 4)

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

    #### CAUSE and WEIGHT OPERATIONS ####
    def get_ta_state(self):
        ta_state = np.empty(self.number_of_clauses * self.number_of_literals, dtype=np.uint32)
        memcpy_dtoh(ta_state, self.ta_state_gpu)
        if self.coalesced:
            return ta_state.reshape((self.number_of_clauses, self.number_of_literals))
        else:
            return ta_state.reshape((self.number_of_clause_banks, self.number_of_clauses_per_class, self.number_of_literals))

    def get_literals(self):
        ta_states = self.get_ta_state()
        return (ta_states > (self.number_of_ta_states // 2)).astype(np.uint32)

    def get_weights(self):
        clause_weights = np.empty(self.number_of_clauses * self.number_of_outputs, dtype=np.float32)
        memcpy_dtoh(clause_weights, self.clause_weights_gpu)
        if self.coalesced:
            return clause_weights.reshape((self.number_of_clauses, self.number_of_outputs))
        else:
            # NOTE: This will mostly be zeros. Maybe this should be returned as a smaller array?
            return clause_weights.reshape((self.number_of_clause_banks, self.number_of_clauses_per_class, self.number_of_outputs))

    def get_patch_weights(self):
        patch_weights = np.empty(self.number_of_clauses * self.number_of_patches, dtype=np.int32)
        memcpy_dtoh(patch_weights, self.patch_weights_gpu)
        if self.coalesced:
            return patch_weights.reshape((self.number_of_clauses, self.number_of_patches))
        else:
            return patch_weights.reshape((self.number_of_clause_banks, self.number_of_clauses_per_class, self.number_of_patches))

    ######## TRANSFORM #######

    def transform(self, X, is_X_encoded: bool = False, block_size: int | None = None):
        encoded_X = X if is_X_encoded else self.encode(X)
        if block_size is None:
            block_size = self.block_size

        N = encoded_X.shape[0]
        max_uint32 = np.iinfo(np.uint32).max
        max_safe_N = max_uint32 // self.number_of_clauses
        if N > max_safe_N:
            raise OverflowError(
                f"X has too many samples ({N}). Maximum of {max_safe_N} samples can be processed with current number_of_clauses. Call this method multiple times with smaller batches of X."
            )

        encoded_X_gpu = mem_alloc(N * self.number_of_patches * self.number_of_literal_chunks * 4)
        memcpy_htod(encoded_X_gpu, encoded_X)

        packed_clauses_gpu, includes_gpu = self._pack_clauses_gpu()

        clause_outputs_gpu = mem_alloc(N * self.number_of_clauses * 4)
        self.kernel_transform.prepared_call(
            *kernel_config(N * self.number_of_clauses, device_props, block_size),
            packed_clauses_gpu,
            includes_gpu,
            encoded_X_gpu,
            np.int32(N),
            clause_outputs_gpu,
        )
        ctx.synchronize()

        clause_outputs = np.zeros((N * self.number_of_clauses), dtype=np.uint32)
        memcpy_dtoh(clause_outputs, clause_outputs_gpu)

        encoded_X_gpu.free()
        packed_clauses_gpu.free()
        includes_gpu.free()
        clause_outputs_gpu.free()

        return clause_outputs.reshape((N, self.number_of_clauses))

    def transform_patchwise(self, X, is_X_encoded: bool = False, block_size: int | None = None):
        encoded_X = X if is_X_encoded else self.encode(X)
        if block_size is None:
            block_size = self.block_size

        N = encoded_X.shape[0]
        max_uint32 = np.iinfo(np.uint32).max
        max_safe_N = max_uint32 // (self.number_of_clauses * self.number_of_patches)
        if N > max_safe_N:
            raise OverflowError(
                f"X has too many samples ({N}). Maximum of {max_safe_N} samples can be processed with current number_of_clauses * number_of_patches. Call this method multiple times with smaller batches of X."
            )

        encoded_X_gpu = mem_alloc(N * self.number_of_patches * self.number_of_literal_chunks * 4)
        memcpy_htod(encoded_X_gpu, encoded_X)

        packed_clauses_gpu = mem_alloc(self.number_of_clauses * self.number_of_literal_chunks * 4)
        includes_gpu = mem_alloc(self.number_of_clauses * 4)

        packed_clauses_gpu, includes_gpu = self._pack_clauses_gpu()

        clause_outputs_gpu = mem_alloc(N * self.number_of_clauses * self.number_of_patches * 4)
        self.kernel_transform_patchwise.prepared_call(
            *kernel_config(N * self.number_of_clauses * self.number_of_patches, device_props, block_size),
            packed_clauses_gpu,
            includes_gpu,
            encoded_X_gpu,
            np.int32(N),
            clause_outputs_gpu,
        )
        ctx.synchronize()
        clause_outputs = np.zeros((N * self.number_of_clauses * self.number_of_patches), dtype=np.uint32)
        memcpy_dtoh(clause_outputs, clause_outputs_gpu)

        encoded_X_gpu.free()
        packed_clauses_gpu.free()
        includes_gpu.free()
        clause_outputs_gpu.free()
        return clause_outputs.reshape((N, self.number_of_clauses, self.number_of_patches))

    ##########################

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
            number_of_clauses_per_class=args["number_of_clauses"],
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
        ta_state = np.empty(
            self.number_of_clauses * self.number_of_literals,
            dtype=np.uint32,
        )
        clause_weights = np.empty(self.number_of_clauses * self.number_of_outputs, dtype=np.float32)
        patch_weights = np.empty(self.number_of_clauses * self.number_of_patches, dtype=np.int32)

        memcpy_dtoh(ta_state, self.ta_state_gpu)
        memcpy_dtoh(clause_weights, self.clause_weights_gpu)
        memcpy_dtoh(patch_weights, self.patch_weights_gpu)

        state_dict = {
            "ta_state": ta_state,
            "clause_weights": clause_weights,
            "patch_weights": patch_weights,
            "min_y": self.min_y,
            "max_y": self.max_y,
        }

        return state_dict

    def load_state_dict(self, state):
        state_dict = state["state"]
        ta_state = state_dict["ta_state"]
        clause_weights = state_dict["clause_weights"]
        patch_weights = state_dict["patch_weights"]
        self.min_y = state_dict["min_y"]
        self.max_y = state_dict["max_y"]

        memcpy_htod(self.ta_state_gpu, ta_state)
        memcpy_htod(self.clause_weights_gpu, clause_weights)
        memcpy_htod(self.patch_weights_gpu, patch_weights)
