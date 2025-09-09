import pathlib
from typing import TypedDict, Unpack

import numpy as np
import pycuda.autoinit  # noqa: F401
import pycuda.curandom as curandom
from pycuda.compiler import SourceModule
from pycuda.driver import Context as ctx  # pyright: ignore[reportAttributeAccessIssue]
from pycuda.driver import mem_alloc, memcpy_dtoh, memcpy_htod, memcpy_dtod, memset_d32  # pyright: ignore[reportAttributeAccessIssue]
from pycuda.gpuarray import to_gpu
from tqdm import tqdm

from .cuda_utils import kernel_config, get_kernel, device_props


class BaseTMOptArgs(TypedDict, total=False):
    q: float
    patch_dim: tuple[int, int]
    number_of_ta_states: int
    max_included_literals: int | None
    append_negated: bool
    init_neg_weights: bool
    negative_polarity: bool
    encode_loc: bool
    coalesced: bool
    h: float | list[float] # Experimental
    bias: bool  # Does not work
    seed: int | None
    block_size: int


class FitOptArgs(TypedDict, total=False):
    block_size: int
    true_mod: list[float] | np.ndarray[tuple[int], np.dtype[np.float64]]  # Experimental
    false_mod: list[float] | np.ndarray[tuple[int], np.dtype[np.float64]]  # Experimental
    clause_drop_p: float
    norm_true_update_prob: bool  # Do not use
    norm_false_update_prob: bool  # Do not use


class BaseTM:
    def __init__(
        self,
        number_of_clauses_per_class: int,
        T: int,
        s: float,
        dim: tuple[int, int, int],
        n_classes: int,
        **opt_args: Unpack[BaseTMOptArgs],
    ):
        # Required arguments
        self.init_args = {
            "number_of_clauses_per_class": number_of_clauses_per_class,
            "T": T,
            "s": s,
            "dim": dim,
            "n_classes": n_classes,
        }
        # Optional arguments -- Needed to set defaults here
        self.opt_args: BaseTMOptArgs = {
            "q": opt_args.get("q", 1.0),
            "patch_dim": opt_args.get("patch_dim", (dim[0], dim[1])),
            "number_of_ta_states": opt_args.get("number_of_ta_states", 256),
            "max_included_literals": opt_args.get("max_included_literals", None),
            "append_negated": opt_args.get("append_negated", True),
            "init_neg_weights": opt_args.get("init_neg_weights", True),
            "negative_polarity": opt_args.get("negative_polarity", True),
            "encode_loc": opt_args.get("encode_loc", True),
            "coalesced": opt_args.get("coalesced", True),
            "h": opt_args.get("h", 1.0),
            "bias": opt_args.get("bias", False),
            "seed": opt_args.get("seed", None),
            "block_size": opt_args.get("block_size", 128),
        }

        self.number_of_clauses_per_class = number_of_clauses_per_class
        self.T = T
        self.s = s
        self.dim = dim
        self.number_of_outputs = n_classes
        self.q = min(self.opt_args["q"], float(self.number_of_outputs))
        self.patch_dim = self.opt_args["patch_dim"]
        self.number_of_ta_states = self.opt_args["number_of_ta_states"]
        self.max_included_literals = self.opt_args["max_included_literals"]
        self.append_negated = self.opt_args["append_negated"]
        self.init_neg_weights = self.opt_args["init_neg_weights"]
        self.negative_clauses = self.opt_args["negative_polarity"]
        self.encode_loc = self.opt_args["encode_loc"]
        self.coalesced = self.opt_args["coalesced"]
        self.bias = self.opt_args["bias"]
        self.seed = self.opt_args["seed"]
        self.block_size = self.opt_args["block_size"]

        self.number_of_clause_banks = 1 if self.coalesced else self.number_of_outputs
        self.number_of_clauses = self.number_of_clause_banks * self.number_of_clauses_per_class

        if isinstance(self.opt_args["h"], list):
            assert len(self.opt_args["h"]) == n_classes, "If h is a list, it must have length equal to n_classes."
            self.h = np.asarray(self.opt_args["h"], dtype=np.float64)
        else:
            self.h = np.asarray([self.opt_args["h"]] * n_classes, dtype=np.float64)

        if not hasattr(self, "min_y"):
            self.min_y = None
        if not hasattr(self, "max_y"):
            self.max_y = None

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
        self.number_of_patches = int((self.dim[0] - self.patch_dim[0] + 1) * (self.dim[1] - self.patch_dim[1] + 1))

        self.rng = np.random.default_rng(self.seed)

        self._gpu_init()

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
        #define MAX_INCLUDED_LITERALS {self.number_of_literals if self.max_included_literals is None else self.max_included_literals}
        #define APPEND_NEGATED {1 if self.append_negated else 0}
        #define INIT_NEG_WEIGHTS {1 if self.init_neg_weights else 0}
        #define NEGATIVE_CLAUSES {1 if self.negative_clauses else 0}
        #define CLASSES {self.number_of_outputs}
        #define MAX_TA_STATE {self.number_of_ta_states}
        #define ENCODE_LOC {1 if self.encode_loc else 0}
        #define COALESCED {1 if self.coalesced else 0}
        #define CLAUSE_BANKS {self.number_of_clause_banks}
        __constant__ const double H[{self.number_of_outputs}] = {{{",".join(map(str, self.h))}}};
        #define BIAS {1 if self.bias else 0}
        """
        current_dir = pathlib.Path(__file__).parent
        kernel_str = get_kernel("cuda/kernel.cu", current_dir)
        mod_new_kernel = SourceModule(
            self.gpu_macro_string + kernel_str,
            options=["-O3", "--use_fast_math"],
            no_extern_c=True,
        )

        self.kernel_init = mod_new_kernel.get_function("init_weights")
        self.kernel_init.prepare("PP")

        self.kernel_encode_batch = mod_new_kernel.get_function("encode_batch")
        self.kernel_encode_batch.prepare("PPi")

        self.kernel_pack_clauses = mod_new_kernel.get_function("pack_clauses")
        self.kernel_pack_clauses.prepare("PPP")

        self.kernel_fast_eval = mod_new_kernel.get_function("fast_eval")
        self.kernel_fast_eval.prepare("PPPPPi")

        self.kernel_select_active = mod_new_kernel.get_function("select_active")
        self.kernel_select_active.prepare("PPPPPP")

        self.kernel_calc_class_sums_infer_batch = mod_new_kernel.get_function("calc_class_sums_infer_batch")
        self.kernel_calc_class_sums_infer_batch.prepare("PPPPiP")

        self.kernel_clause_update = mod_new_kernel.get_function("clause_update")
        self.kernel_clause_update.prepare("PPPPPPPPPPPPiii")

        self.kernel_transform = mod_new_kernel.get_function("transform")
        self.kernel_transform.prepare("PPPiP")

        self.kernel_transform_patchwise = mod_new_kernel.get_function("transform_patchwise")
        self.kernel_transform_patchwise.prepare("PPPiP")

        # Allocate GPU memory
        self.ta_state_gpu = mem_alloc(self.number_of_clauses * self.number_of_literals * 4)
        self.clause_weights_gpu = mem_alloc(self.number_of_clauses * self.number_of_outputs * 4)
        self.patch_weights_gpu = mem_alloc(self.number_of_clauses * self.number_of_patches * 4)
        self.bias_weights_gpu = mem_alloc(self.number_of_outputs * 4)

        # RNG
        self.rng_gpu = (
            curandom.XORWOWRandomNumberGenerator()
            if self.seed is None
            else curandom.XORWOWRandomNumberGenerator(
                lambda count: to_gpu(np.array([(self.seed + i) for i in range(1, count + 1)], dtype=np.int32))  # pyright: ignore[reportOptionalOperand]
            )
        )

        # Init ta_state to number_of_ta_states // 2
        self._reset_clauses()
        self._reset_weights(True)
        memset_d32(self.patch_weights_gpu, 0, self.number_of_clauses * self.number_of_patches)

    def _reset_clauses(self):
        memset_d32(self.ta_state_gpu, self.number_of_ta_states // 2, self.number_of_clauses * self.number_of_literals)

    def _reset_weights(self, reset_bias: bool = True):
        # TODO: Implement
        self.kernel_init.prepared_call(
            *kernel_config(self.number_of_clauses, device_props, self.block_size),
            self.rng_gpu.state,
            # self.ta_state_gpu,
            self.clause_weights_gpu,
        )
        ctx.synchronize()
        if reset_bias:
            memset_d32(self.bias_weights_gpu, 0, self.number_of_outputs)
        pass

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
    def _fit(self, encoded_X, encoded_Y, **opt_args: Unpack[FitOptArgs]):
        # Process optional arguments
        block_size = opt_args.get("block_size", 128)
        true_mod = np.asarray(opt_args.get("true_mod", np.ones(self.number_of_outputs)), dtype=np.float64)
        false_mod = np.asarray(opt_args.get("false_mod", np.ones(self.number_of_outputs)), dtype=np.float64)
        clause_drop_p = opt_args.get("clause_drop_p", 0.0)
        norm_true_update_prob = opt_args.get("norm_true_update_prob", False)  # In case of multi-label
        norm_false_update_prob = opt_args.get("norm_false_update_prob", False)

        N = encoded_X.shape[0]
        encoded_X_gpu = mem_alloc(encoded_X.nbytes)
        encoded_Y_gpu = mem_alloc(encoded_Y.nbytes)
        memcpy_htod(encoded_X_gpu, encoded_X)
        memcpy_htod(encoded_Y_gpu, encoded_Y)

        # Class probability balancer
        true_mod_gpu = mem_alloc(true_mod.nbytes)
        false_mod_gpu = mem_alloc(false_mod.nbytes)
        memcpy_htod(true_mod_gpu, true_mod)
        memcpy_htod(false_mod_gpu, false_mod)

        # Drop clauses
        if clause_drop_p > 0.0:
            clause_drop_mask = (self.rng.random(self.number_of_clauses) <= clause_drop_p).astype(np.uint32)
        else:
            clause_drop_mask = np.zeros(self.number_of_clauses, dtype=np.uint32)
        clause_drop_mask_gpu = mem_alloc(clause_drop_mask.nbytes)
        memcpy_htod(clause_drop_mask_gpu, clause_drop_mask)

        # Initialize GPU memory for temporary data
        packed_clauses_gpu = mem_alloc(self.number_of_clauses * self.number_of_literal_chunks * 4)
        class_sum_gpu = mem_alloc(self.number_of_outputs * 4)
        clause_outputs_gpu = mem_alloc(self.number_of_clauses * self.number_of_patches * 4)
        selected_patch_ids_gpu = mem_alloc(self.number_of_clauses * 4)
        num_includes_gpu = mem_alloc(self.number_of_clauses * 4)

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
                clause_drop_mask_gpu,
                encoded_X_gpu,
                clause_outputs_gpu,
                np.int32(e),
            )
            ctx.synchronize()

            # memset_d32(class_sum_gpu, 0, self.number_of_outputs)
            memcpy_dtod(class_sum_gpu, self.bias_weights_gpu, self.number_of_outputs * 4)
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
                self.bias_weights_gpu,
                class_sum_gpu,
                selected_patch_ids_gpu,
                num_includes_gpu,
                true_mod_gpu,
                false_mod_gpu,
                clause_drop_mask_gpu,
                encoded_X_gpu,
                encoded_Y_gpu,
                np.int32(e),
                np.int32(1) if norm_true_update_prob else np.int32(0),
                np.int32(1) if norm_false_update_prob else np.int32(0),
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

        # Initialize class sums with bias weights
        # class_sums = np.zeros((N, self.number_of_outputs), dtype=np.float32)
        bias_weights = np.empty((1, self.number_of_outputs), dtype=np.float32)
        memcpy_dtoh(bias_weights, self.bias_weights_gpu)
        class_sums = np.tile(bias_weights, (N, 1)).astype(np.float32)

        for i in range(0, N, max_safe_N):
            X_safe = encoded_X[i : i + max_safe_N]
            X_gpu = mem_alloc(X_safe.nbytes)
            memcpy_htod(X_gpu, X_safe)

            # class_sums_gpu = mem_alloc(X_safe.shape[0] * self.number_of_outputs * 4)
            # memset_d32(class_sums_gpu, 0, X_safe.shape[0] * self.number_of_outputs)
            cs_safe = class_sums[i : i + max_safe_N]
            class_sums_gpu = mem_alloc(cs_safe.nbytes)
            memcpy_htod(class_sums_gpu, cs_safe)
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

    #### CAUSE and WEIGHT OPERATIONS ####
    def get_ta_state(self):
        ta_state = np.empty(self.number_of_clauses * self.number_of_literals, dtype=np.uint32)
        memcpy_dtoh(ta_state, self.ta_state_gpu)
        if self.coalesced:
            return ta_state.reshape((self.number_of_clauses, self.number_of_literals))
        else:
            return ta_state.reshape(
                (self.number_of_clause_banks, self.number_of_clauses_per_class, self.number_of_literals)
            )

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
            return clause_weights.reshape(
                (self.number_of_clause_banks, self.number_of_clauses_per_class, self.number_of_outputs)
            )

    def get_patch_weights(self):
        patch_weights = np.empty(self.number_of_clauses * self.number_of_patches, dtype=np.int32)
        memcpy_dtoh(patch_weights, self.patch_weights_gpu)
        if self.coalesced:
            return patch_weights.reshape((self.number_of_clauses, self.number_of_patches))
        else:
            return patch_weights.reshape(
                (self.number_of_clause_banks, self.number_of_clauses_per_class, self.number_of_patches)
            )

    def get_bias_weights(self):
        bias_weights = np.empty(self.number_of_outputs, dtype=np.float32)
        memcpy_dtoh(bias_weights, self.bias_weights_gpu)
        return bias_weights

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
            **self.init_args,
            **self.opt_args,
        }
        state_dict = self.get_state_dict()
        return {"args": args, "state": state_dict}

    def __setstate__(self, state):
        args = state["args"]
        self.__init__(**args)
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
