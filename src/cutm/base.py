import pathlib
from typing import Literal, TypedDict, Unpack

import numpy as np
import pycuda.autoinit  # noqa: F401
import pycuda.curandom as curandom
from pycuda.compiler import SourceModule
from pycuda.driver import Context as ctx  # pyright: ignore[reportAttributeAccessIssue]
from pycuda.driver import mem_alloc, memcpy_dtoh, memcpy_htod, memset_d32  # pyright: ignore[reportAttributeAccessIssue]
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
    weighted: bool
    max_weight: float
    s_neg_polarity: float
    h: float | list[float]
    allow_polarity_change: bool
    initial_weight: float
    initial_state: int | Literal["random"] | Literal["middle"]
    include_state: int | Literal["middle"]
    type1a_fb: bool
    type1b_fb: bool
    type2_fb: bool
    split_class_sum: bool
    seed: int | None
    block_size: int
    grid_size: int | None


class FitOptArgs(TypedDict, total=False):
    block_size: int
    grid_size: int
    clause_drop_p: float
    shuffle: bool
    label_sampling: bool | int
    g_pos: float | list[float]
    g_neg: float | list[float]
    log: bool


def _float_or_list_to_array(x: float | list[float], size: int):
    if isinstance(x, float) or isinstance(x, int):
        return np.array([x] * size, dtype=np.float64)
    elif isinstance(x, list):
        assert len(x) == size, f"List must be of size {size}."
        return np.array(x, dtype=np.float64)
    else:
        raise ValueError("x must be a float or a list of floats.")


class BaseTM:
    """Implementing tsetlin machine

    Attributes
    ----------
    number_of_clauses_per_class : int
        Number of clauses per class. If using coalesced clauses(default), then the clauses are shared among all classes, and total_num_clauses = number_of_clauses_per_clas. Else,(non-coalesced), total_num_clauses = number_of_clauses_per_class * num_classes.
    T : int | float
        The voting threshold.
    s : float
        The specificity parameter. Should be >= 1.0.
    dim : tuple[int, int, int]
        Input dimensions. In cases where the input is image like, this is (height, width, channels) or (height, weidth, 1). But, in general, this should be (number_of_features, 1, 1). MUST BE A TUPLE WITH 3 VALUES.
    n_classes : int
        Number of output classes.
    q : float
        Hyperparameter Q
    patch_dim : tuple[int, int], optional, default=(dim[0], dim[1])
        Only use when using convolution.
    number_of_ta_states : int, optional, default=256
        Number of states per Tsetlin Automaton.
    max_included_literals : int | None, optional, default=None
        Maximum number of literals that can be included in a clause. If None, then no limit.
    append_negated : bool, optional, default=True
        Whether to append negated features to the input.
    init_neg_weights : bool, optional, default=True
        Option to turn off initialization of negative clause weights.
    negative_polarity : bool, optional, default=True
        Whether to use negative polarity clauses.
    encode_loc : bool, optional, default=True
        When using convolution, whether to encode the patch location as literals.
    coalesced : bool, optional, default=True
        Whether to use coalesced clauses.
    weighted : bool, optional, default=True
        Whether to use weighted or unweighted clauses.
    max_weight : float, optional, default=np.finfo(np.float32).max
        Maximum absolute value for clause weights. Defaults to maximum float32 value.
    s_neg_polarity : float, optional, default=s
        Specificity parameter for negative polarity clauses. Defaults to s.
    h : float | list[float], optional, default=0.5
        Experimental. DO NOT USE.
    allow_polarity_change : bool, optional, default=True
        Whether to allow polarity change during training.
    initial_weight : float, optional, default=1.0
        Absolute value of initial clause weights.
    initial_state : int | Literal["random"] | Literal["middle"], optional, default="middle"
        Initial state of Tsetlin Automata. If "middle", then all TAs are initialized to the middle state. If "random", then all TAs are initialized randomly. Else, should be an integer between 0 and number_of_ta_states - 1.
    include_state : int | Literal["middle"], optional, default="middle"
        State threshold for including literals. If "middle", then the threshold is set to the middle state + 1. Else, should be an integer between 0 and number_of_ta_states - 1.
    type1a_fb : bool, optional, default=True
        Option to disable Type 1a feedback.
    type1b_fb : bool, optional, default=True
        Option to disable Type 1b feedback.
    type2_fb : bool, optional, default=True
        Option to disable Type 2 feedback.
    split_class_sum: bool, optional, default=False
        Experimental. DO NOT USE.
    seed : int | None, optional, default=None
        Random seed. Does not gaurantee reproducibility, because of GPU parallelism. But the initialization of clauses and weights should be the same for the same seed.
    block_size : int, optional, default=128
        CUDA kernel parameter
    grid_size : int | None, optional, default=None
        CUDA kernel parameter
    """

    def __init__(
        self,
        number_of_clauses_per_class: int,
        T: int | float,
        s: float,
        dim: tuple[int, int, int],
        n_classes: int,
        **opt_args: Unpack[BaseTMOptArgs],
    ):
        unexpected_args = set(opt_args.keys()) - set(BaseTMOptArgs.__annotations__.keys())
        if unexpected_args:
            raise TypeError(f"Unexpected keyword arguments: {unexpected_args}")

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
            "weighted": opt_args.get("weighted", True),
            "max_weight": opt_args.get("max_weight", float(np.finfo(np.float32).max)),
            "s_neg_polarity": opt_args.get("s_neg_polarity", s),
            "h": opt_args.get("h", 0.5),
            "allow_polarity_change": opt_args.get("allow_polarity_change", True),
            "initial_weight": opt_args.get("initial_weight", 1.0),
            "initial_state": opt_args.get("initial_state", "middle"),
            "include_state": opt_args.get("include_state", "middle"),
            "type1a_fb": opt_args.get("type1a_fb", True),
            "type1b_fb": opt_args.get("type1b_fb", True),
            "type2_fb": opt_args.get("type2_fb", True),
            "split_class_sum": opt_args.get("split_class_sum", False),
            "seed": opt_args.get("seed", None),
            "block_size": opt_args.get("block_size", 128),
            "grid_size": opt_args.get("grid_size", None),
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
        self.weighted = self.opt_args["weighted"]
        self.max_weight = self.opt_args["max_weight"]
        self.s_neg_polarity = self.opt_args["s_neg_polarity"]
        self.h = _float_or_list_to_array(self.opt_args["h"], self.number_of_outputs)
        self.allow_polarity_change = self.opt_args["allow_polarity_change"]
        self.initial_weight = self.opt_args["initial_weight"]
        self.type1a_fb = self.opt_args["type1a_fb"]
        self.type1b_fb = self.opt_args["type1b_fb"]
        self.type2_fb = self.opt_args["type2_fb"]
        self.split_class_sum = self.opt_args["split_class_sum"]
        self.seed = self.opt_args["seed"]
        self.block_size = self.opt_args["block_size"]
        self.grid_size = self.opt_args["grid_size"]

        if self.opt_args["initial_state"] == "middle":
            self.initial_state = (self.number_of_ta_states - 1) // 2
        elif self.opt_args["initial_state"] == "random":
            self.initial_state = -1
        elif (
            isinstance(self.opt_args["initial_state"], int)
            and 0 <= self.opt_args["initial_state"] < self.number_of_ta_states
        ):
            self.initial_state = self.opt_args["initial_state"]
        else:
            raise ValueError(
                "initial_state must be 'middle', 'random', or an integer between 0 and number_of_ta_states - 1."
            )

        if self.opt_args["include_state"] == "middle":
            self.include_state = ((self.number_of_ta_states - 1) // 2) + 1
        elif (
            isinstance(self.opt_args["include_state"], int)
            and 0 <= self.opt_args["include_state"] < self.number_of_ta_states
        ):
            self.include_state = self.opt_args["include_state"]
        else:
            raise ValueError("include_state must be 'middle' or an integer between 0 and number_of_ta_states - 1.")

        self.number_of_clause_banks = 1 if self.coalesced else self.number_of_outputs
        self.number_of_clauses = self.number_of_clause_banks * self.number_of_clauses_per_class

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
        #define CLAUSES {self.number_of_clauses}ULL
        #define THRESH {self.T}
        #define S {self.s}
        #define Q {self.q}
        #define DIM0 {self.dim[0]}ULL
        #define DIM1 {self.dim[1]}ULL
        #define DIM2 {self.dim[2]}ULL
        #define PATCH_DIM0 {self.patch_dim[0]}
        #define PATCH_DIM1 {self.patch_dim[1]}
        #define PATCHES {self.number_of_patches}ULL
        #define LITERALS {self.number_of_literals}ULL
        #define MAX_INCLUDED_LITERALS {self.number_of_literals if self.max_included_literals is None else self.max_included_literals}ULL
        #define APPEND_NEGATED {1 if self.append_negated else 0}
        #define NEGATIVE_CLAUSES {1 if self.negative_clauses else 0}
        #define CLASSES {self.number_of_outputs}
        #define MAX_TA_STATE {self.number_of_ta_states - 1}
        #define ENCODE_LOC {1 if self.encode_loc else 0}
        #define COALESCED {1 if self.coalesced else 0}
        #define CLAUSE_BANKS {self.number_of_clause_banks}
        #define WEIGHTED {1 if self.weighted else 0}
        #define MAX_WEIGHT {self.max_weight}
        #define S_NEG_POLARITY {self.s_neg_polarity}
        #define ALLOW_POLARITY_CHANGE {1 if self.allow_polarity_change else 0}
        #define INCLUDE_TA_STATE {self.include_state}
        #define TYPE1A_FB {1 if self.type1a_fb else 0}
        #define TYPE1B_FB {1 if self.type1b_fb else 0}
        #define TYPE2_FB {1 if self.type2_fb else 0}
        #define SPLIT_CLASS_SUM {1 if self.split_class_sum else 0}
        __device__ double H[{self.number_of_outputs}] = {{{", ".join([str(h) for h in self.h])}}};
        """
        current_dir = pathlib.Path(__file__).parent
        kernel_str = get_kernel("cuda/kernel.cu", current_dir)
        mod_new_kernel = SourceModule(
            self.gpu_macro_string + kernel_str,
            options=["-O3", "--use_fast_math"],
            no_extern_c=True,
        )

        self.kernel_encode_batch = mod_new_kernel.get_function("encode_batch")
        self.kernel_encode_batch.prepare("PPi")

        self.kernel_pack_clauses = mod_new_kernel.get_function("pack_clauses")
        self.kernel_pack_clauses.prepare("PPP")

        self.kernel_fast_eval = mod_new_kernel.get_function("fast_eval")
        self.kernel_fast_eval.prepare("PPPPPPi")

        self.kernel_select_active = mod_new_kernel.get_function("select_active")
        self.kernel_select_active.prepare("PPPPPPP")

        self.kernel_calc_class_sums_infer_batch = mod_new_kernel.get_function("calc_class_sums_infer_batch")
        self.kernel_calc_class_sums_infer_batch.prepare("PPPPiPP")

        self.kernel_evidence_to_update_prob = mod_new_kernel.get_function("evidence_to_update_prob")
        self.kernel_evidence_to_update_prob.prepare("PPPPPiPP")

        self.kernel_clause_update = mod_new_kernel.get_function("clause_update")
        self.kernel_clause_update.prepare("PPPPPPPPPPPi")

        self.kernel_transform = mod_new_kernel.get_function("transform")
        self.kernel_transform.prepare("PPPiPP")

        self.kernel_transform_patchwise = mod_new_kernel.get_function("transform_patchwise")
        self.kernel_transform_patchwise.prepare("PPPiPP")

        # Allocate GPU memory
        self.ta_state_gpu = mem_alloc(self.number_of_clauses * self.number_of_literals * 4)
        self.clause_weights_gpu = mem_alloc(self.number_of_clauses * self.number_of_outputs * 4)
        self.patch_weights_gpu = mem_alloc(self.number_of_clauses * self.number_of_patches * 4)
        self.literal_mask_gpu = mem_alloc(self.number_of_literal_chunks * 4)

        # RNG
        self.rng_gpu = (
            curandom.XORWOWRandomNumberGenerator()
            if self.seed is None
            else curandom.XORWOWRandomNumberGenerator(
                lambda count: to_gpu(np.array([(self.seed + i) for i in range(1, count + 1)], dtype=np.int32))  # pyright: ignore[reportOptionalOperand]
            )
        )

        self._reset_clauses()
        self._reset_weights()
        memset_d32(self.patch_weights_gpu, 0, self.number_of_clauses * self.number_of_patches)
        memset_d32(self.literal_mask_gpu, 0xFFFFFFFF, self.number_of_literal_chunks)

    def _init_clauses(self):
        if self.initial_state == -1:
            # Random initialization
            ta_states = self.rng.integers(
                low=0,
                high=self.number_of_ta_states,
                size=(self.number_of_clauses * self.number_of_literals),
                dtype=np.uint32,
            )
        else:
            # Fixed initialization
            ta_states = np.full(
                (self.number_of_clauses * self.number_of_literals),
                self.initial_state,
                dtype=np.uint32,
            )
        memcpy_htod(self.ta_state_gpu, ta_states)

    def _init_weights(self):
        num_neg_polarity = self.number_of_clauses_per_class // 2
        if self.coalesced:
            weights = np.zeros((self.number_of_clauses, self.number_of_outputs), dtype=np.float32)
            for i in range(self.number_of_outputs):
                wt = np.ones((self.number_of_clauses,), dtype=np.float32) * self.initial_weight
                if self.init_neg_weights:
                    wt[num_neg_polarity:] = -1.0 * self.initial_weight
                weights[:, i] = self.rng.permutation(wt)
        else:
            w = []
            for i in range(self.number_of_outputs):
                wt = np.zeros((self.number_of_clauses_per_class, self.number_of_outputs), dtype=np.float32)
                wt[:, i] = 1.0 * self.initial_weight
                if self.init_neg_weights:
                    wt[num_neg_polarity:, i] = -1.0 * self.initial_weight
                w.append(wt)
            weights = np.vstack(w)

        memcpy_htod(self.clause_weights_gpu, weights)

    def _reset_clauses(self):
        self._init_clauses()

    def _reset_weights(self):
        self._init_weights()

    def encode(
        self,
        X: np.ndarray,
        input_type: Literal["binary", "ternary"] = "binary",
        block_size: int | None = None,
        grid_size: int | None = None,
    ) -> np.ndarray[tuple[int, int, int], np.dtype[np.uint32]]:
        """
        Encoding input X into bit-packed format, which is suitable for the TM.
        In case of convolutional TM, this also extracts patches from the input
        and appends the pathch location if encode_loc is True.

        Parameters
        ----------
        X : np.ndarray
            Input data. Internally reshaped to 2D array of shape (N, dim[0] * dim[1] * dim[2]), and data type converted to np.int32.
        input_type : Literal["binary", "ternary"]
            Input data can be "binary" (the normal case), or "ternary" (dont know the use case yet, so dont use).
        block_size : int | None
            CUDA kernel parameter
        grid_size : int | None
            CUDA kernel parameter

        Returns
        -------
        np.ndarray[tuple[int, int, int], np.dtype[np.uint32]]
            Encoded X of shape (N, number_of_patches, number_of_literal_chunks), of dtype np.uint32. Each value is a bit-packed representation of 32 literals.

        """

        # Convert to proper data type and shape
        X = X.astype(np.int32).reshape((X.shape[0], self.dim[0], self.dim[1], self.dim[2]))

        # Validate `input_type`, and convert {0,1} to {-1,1} if binary
        if input_type == "binary":
            assert np.min(X) >= 0 and np.max(X) <= 1, "X must be binary (contain only 0 or 1)."
            X = X * 2 - 1  # Convert to -1, 1
        elif input_type == "ternary":
            assert np.min(X) >= -1 and np.max(X) <= 1, "X must be ternary (contain only -1, 0 or 1)."
        else:
            raise ValueError("input_type must be 'binary' or 'ternary'.")

        N = X.shape[0]

        if block_size is None:
            block_size = self.block_size
        if grid_size is None:
            grid_size = self.grid_size

        # Calculate maximum number of samples that can be processed at once. Necessary to avoid GPU memory issues.
        max_uint32 = np.iinfo(np.uint32).max
        max_safe_N = max_uint32 // self.number_of_patches

        encoded_X = np.empty((N, self.number_of_patches, self.number_of_literal_chunks), dtype=np.uint32)
        for i in range(0, N, max_safe_N):
            X_safe = X[i : i + max_safe_N]
            X_gpu = mem_alloc(X_safe.nbytes)

            # Copy data to GPU
            memcpy_htod(X_gpu, X_safe)

            # Allocate memory to store output and initialize to zero
            encoded_X_gpu = mem_alloc(X_safe.shape[0] * self.number_of_patches * self.number_of_literal_chunks * 4)
            memset_d32(encoded_X_gpu, 0, X_safe.shape[0] * self.number_of_patches * self.number_of_literal_chunks)

            # Encode kernel
            self.kernel_encode_batch.prepared_call(
                *kernel_config(X_safe.shape[0] * self.number_of_patches, device_props, block_size, grid_size),
                X_gpu,
                encoded_X_gpu,
                np.int32(X_safe.shape[0]),
            )
            ctx.synchronize()

            # Copy back to CPU
            encoded_X_safe = np.empty(
                (X_safe.shape[0] * self.number_of_patches * self.number_of_literal_chunks), dtype=np.uint32
            )
            memcpy_dtoh(encoded_X_safe, encoded_X_gpu)

            encoded_X[i : i + max_safe_N] = encoded_X_safe.reshape(
                (X_safe.shape[0], self.number_of_patches, self.number_of_literal_chunks)
            )

        return encoded_X

    def _target_sampling(self, Y, label_sampling: bool | int):
        N = Y.shape[0]
        targets = np.zeros_like(Y, dtype=np.int32)

        if label_sampling is False or label_sampling <= 0:
            # Original behaviour
            targets[Y > 0] = 1
            p = self.q / max(1, self.number_of_outputs - 1)
            for i in range(N):
                # Each negeative class can get feedback with prob (q / number_of_outputs - 1)
                false_classes = np.where(Y[i, :] <= 0)[0]
                if len(false_classes) > 0:
                    not_skip = self.rng.random(size=len(false_classes)) <= p
                    targets[i, false_classes[not_skip]] = -1
        else:
            # Drop label such that the TM sees the same number of labels for each class
            per_class_counts = np.sum(Y > 0, axis=0)

            # min_cnt is the number of true labels to keep for each class.
            if isinstance(label_sampling, bool) and label_sampling is True:
                min_cnt = np.min(per_class_counts)
            else:
                min_cnt = min(N, label_sampling)

            # Select min-cnt true labels for each class
            for i in range(self.number_of_outputs):
                inds = np.where(Y[:, i] > 0)[0]
                if len(inds) > min_cnt:
                    sel = self.rng.choice(inds, size=min_cnt, replace=False)
                    targets[sel, i] = 1
                else:
                    targets[inds, i] = 1

            # False label selection based on q
            p = self.q / max(1, self.number_of_outputs - 1)
            for i in range(N):
                true_classes = np.where(targets[i, :] > 0)[0]
                if len(true_classes) == 0:
                    continue
                false_classes = np.where(Y[i, :] <= 0)[0]
                if len(false_classes) > 0:
                    not_skip = self.rng.random(size=len(false_classes)) <= p
                    targets[i, false_classes[not_skip]] = -1

        return targets

    def freeze_literals(self, literal_inds: list[int]):
        """
        Freeze the given literals, so that they are not used in evaluation or update steps.
        Literals not in the list are unfrozen. So, passing an empty list unfreezes all literals.

        Parameters
        ----------
        literal_inds : list[int]
            List of literal indices to be frozen.
        """
        # Create a literal_mask from literal_inds, and update a global self.literal_mask and self.literal_mask_gpu variable.
        lit_msk = np.full((self.number_of_literal_chunks,), 0xFFFFFFFF, dtype=np.uint32)
        if len(literal_inds) > 0:
            for litid in literal_inds:
                chunk_nr = litid // 32
                chunk_pos = litid % 32
                lit_msk[chunk_nr] &= ~(np.uint32(1) << chunk_pos)
        self.literal_mask = lit_msk
        memcpy_htod(self.literal_mask_gpu, self.literal_mask)

    def _validate_fit_args(self, **opt_args: Unpack[FitOptArgs]):
        unexpected_args = set(opt_args.keys()) - set(FitOptArgs.__annotations__.keys())
        if unexpected_args:
            raise TypeError(f"Unexpected keyword arguments: {unexpected_args}")
        args = {
            "block_size": opt_args.get("block_size", self.block_size),
            "grid_size": opt_args.get("grid_size", self.grid_size),
            "clause_drop_p": opt_args.get("clause_drop_p", 0.0),
            "shuffle": opt_args.get("shuffle", True),
            "label_sampling": opt_args.get("label_sampling", False),
            "g_pos": _float_or_list_to_array(opt_args.get("g_pos", 1.0), self.number_of_outputs),
            "g_neg": _float_or_list_to_array(opt_args.get("g_neg", 1.0), self.number_of_outputs),
            "log": opt_args.get("log", False),
        }
        return args

    def _fit_allocate_gpu(self, args, encoded_X, targets):
        # Drop clauses
        clause_drop_p = args["clause_drop_p"]
        if clause_drop_p > 0.0:
            clause_drop_mask = (self.rng.random(self.number_of_clauses) <= clause_drop_p).astype(np.uint32)
        else:
            clause_drop_mask = np.zeros(self.number_of_clauses, dtype=np.uint32)
        gpu_buffers = {
            "encoded_X_gpu": mem_alloc(encoded_X.nbytes),
            "targets_gpu": mem_alloc(targets.nbytes),
            "clause_drop_mask_gpu": mem_alloc(clause_drop_mask.nbytes),
            "g_pos_gpu": mem_alloc(args["g_pos"].nbytes),
            "g_neg_gpu": mem_alloc(args["g_neg"].nbytes),
            "packed_clauses_gpu": mem_alloc(self.number_of_clauses * self.number_of_literal_chunks * 4),
            "positive_evidence_gpu": mem_alloc(self.number_of_outputs * 4),
            "negative_evidence_gpu": mem_alloc(self.number_of_outputs * 4),
            "pprob_gpu": mem_alloc(self.number_of_outputs * 8),  # double
            "nprob_gpu": mem_alloc(self.number_of_outputs * 8),  # double
            "clause_outputs_gpu": mem_alloc(self.number_of_clauses * self.number_of_patches * 4),
            "selected_patch_ids_gpu": mem_alloc(self.number_of_clauses * 4),
            "num_includes_gpu": mem_alloc(self.number_of_clauses * 4),
        }

        # Copy stuff to GPU
        memcpy_htod(gpu_buffers["encoded_X_gpu"], encoded_X)
        memcpy_htod(gpu_buffers["targets_gpu"], targets)
        memcpy_htod(gpu_buffers["clause_drop_mask_gpu"], clause_drop_mask)
        memcpy_htod(gpu_buffers["g_pos_gpu"], args["g_pos"])
        memcpy_htod(gpu_buffers["g_neg_gpu"], args["g_neg"])

        return gpu_buffers

    def _fit_sample(self, gpu_buffers: dict, e: int, kconfs: dict):
        # Bit-pack included literals, so that we can use bitwise operations to process INT_SIZE literals at once. Also calculate number of includes per clause.
        self.kernel_pack_clauses.prepared_call(
            *kconfs["config_n_clauses"],
            self.ta_state_gpu,
            gpu_buffers["packed_clauses_gpu"],
            gpu_buffers["num_includes_gpu"],
        )
        ctx.synchronize()

        # Evaluate clauses. Calculates the clause_outputs_gpu
        self.kernel_fast_eval.prepared_call(
            *kconfs["config_patchwise"],
            gpu_buffers["packed_clauses_gpu"],
            gpu_buffers["num_includes_gpu"],
            gpu_buffers["clause_drop_mask_gpu"],
            self.literal_mask_gpu,
            gpu_buffers["encoded_X_gpu"],
            gpu_buffers["clause_outputs_gpu"],
            np.int32(e),
        )
        ctx.synchronize()

        # Reset class sums.
        memset_d32(gpu_buffers["positive_evidence_gpu"], 0, self.number_of_outputs)
        memset_d32(gpu_buffers["negative_evidence_gpu"], 0, self.number_of_outputs)

        # Select a patch for each clause, and also calculate the class sum.
        self.kernel_select_active.prepared_call(
            *kconfs["config_n_clauses"],
            self.rng_gpu.state,
            self.clause_weights_gpu,
            gpu_buffers["clause_outputs_gpu"],
            self.patch_weights_gpu,
            gpu_buffers["selected_patch_ids_gpu"],
            gpu_buffers["positive_evidence_gpu"],
            gpu_buffers["negative_evidence_gpu"]
        )
        ctx.synchronize()

        # Calculate the class sums to update probabilities.
        # This also implements a curved uprob functions.
        self.kernel_evidence_to_update_prob.prepared_call(
            *kconfs["config_outputs"],
            gpu_buffers["positive_evidence_gpu"],
            gpu_buffers["negative_evidence_gpu"],
            gpu_buffers["targets_gpu"],
            gpu_buffers["g_pos_gpu"],
            gpu_buffers["g_neg_gpu"],
            np.int32(e),
            gpu_buffers["pprob_gpu"],
            gpu_buffers["nprob_gpu"],
        )
        ctx.synchronize()

        # Finally, we can update the clauses.
        self.kernel_clause_update.prepared_call(
            *kconfs["config_n_clauses"],
            self.rng_gpu.state,
            self.ta_state_gpu,
            self.clause_weights_gpu,
            gpu_buffers["selected_patch_ids_gpu"],
            gpu_buffers["num_includes_gpu"],
            gpu_buffers["clause_drop_mask_gpu"],
            self.literal_mask_gpu,
            gpu_buffers["encoded_X_gpu"],
            gpu_buffers["targets_gpu"],
            gpu_buffers["pprob_gpu"],
            gpu_buffers["nprob_gpu"],
            np.int32(e),
        )
        ctx.synchronize()

    #### FIT AND SCORE ####
    def _fit(self, encoded_X, encoded_Y, **opt_args: Unpack[FitOptArgs]):
        """Main fit function.

        Parameters
        ----------
        encoded_X : np.ndarray
            Encoded input data of shape (N, number_of_patches, number_of_literal_chunks), of dtype np.uint32. It should already be bit packed and processed using the `encode` method.
        encoded_Y : np.ndarray
            Encoded target data of shape (N, number_of_outputs), of dtype np.int32. It should contain 1 for positive class, and 0 for negative class.
        clause_drop_p : float
            Probability of dropping a clause during each training sample.
        shuffle : bool
            Whether to shuffle the training data.
        label_sampling : bool | int
            Experimental. Do not use. Randomly drop class labels.
        g_pos : float | list[float]
            Experimental. DO NOT USE.
        g_neg : float | list[float]
            Experimental. DO NOT USE.
        log : bool
            Some logs for me. DO NOT USE.
        """
        N = encoded_X.shape[0]
        args = self._validate_fit_args(**opt_args)

        # Shuffle input
        iota = np.arange(N)
        if args["shuffle"]:
            self.rng.shuffle(iota)
        encoded_X = encoded_X[iota]
        encoded_Y = encoded_Y[iota]

        # Precompute targets for each sample. 1 means the sample belongs to the class. -1 means, selected not classes. 0 means ignore.
        targets = self._target_sampling((encoded_Y > 0).astype(np.int32), args["label_sampling"])

        # Allocate GPU buffers and copy data
        gpu_buffers = self._fit_allocate_gpu(args, encoded_X, targets)

        # Prepare logging dictionary
        logged_data = {}
        log = opt_args.get("log", False)
        if log:
            logged_data["opt_args"] = opt_args
            logged_data["iota"] = iota
            logged_data["targets"] = targets
            class_sums = np.zeros((N, self.number_of_outputs), dtype=np.float32)
            update_probs = np.zeros((N, self.number_of_outputs), dtype=np.float64)

        # Kernel configurations
        kconfs = {
            "config_n_clauses": kernel_config(
                self.number_of_clauses, device_props, args["block_size"], args["grid_size"]
            ),
            "config_patchwise": kernel_config(
                self.number_of_clauses * self.number_of_patches, device_props, args["block_size"], args["grid_size"]
            ),
            "config_outputs": kernel_config(
                self.number_of_outputs, device_props, args["block_size"], args["grid_size"]
            ),
        }

        # Main fit loop over all the samples
        pbar = tqdm(range(N), desc="Fitting Batch", leave=False, dynamic_ncols=True)
        for e in pbar:
            # If all the targets are zero, then there is nothing to learn, so skip.
            if np.all(targets[e, :] == 0):
                continue

            self._fit_sample(gpu_buffers, e, kconfs)

            # If saving the logs, copy class sums and update probs back to CPU
            if log:
                memcpy_dtoh(class_sums[e : e + 1], gpu_buffers["class_sum_gpu"])  # pyright: ignore[reportPossiblyUnboundVariable]
                memcpy_dtoh(update_probs[e : e + 1], gpu_buffers["update_probs_gpu"])  # pyright: ignore[reportPossiblyUnboundVariable]

        if log:
            logged_data["class_sums"] = class_sums  # pyright: ignore[reportPossiblyUnboundVariable]
            logged_data["update_probs"] = update_probs  # pyright: ignore[reportPossiblyUnboundVariable]

        return logged_data

    def _pack_clauses_gpu(self, block_size, grid_size):
        packed_clauses_gpu = mem_alloc(self.number_of_clauses * self.number_of_literal_chunks * 4)
        includes_gpu = mem_alloc(self.number_of_clauses * 4)

        self.kernel_pack_clauses.prepared_call(
            *kernel_config(self.number_of_clauses, device_props, block_size, grid_size),
            self.ta_state_gpu,
            packed_clauses_gpu,
            includes_gpu,
        )
        ctx.synchronize()

        return packed_clauses_gpu, includes_gpu

    def _score_batch(
        self, encoded_X, block_size: int | None = None, grid_size: int | None = None
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float32]]:
        N = encoded_X.shape[0]

        # Kernel parameters
        if block_size is None:
            block_size = self.block_size
        if grid_size is None:
            grid_size = self.grid_size

        # Find the maximum number of samples that can be safely processed parallelly
        max_uint32 = np.iinfo(np.uint32).max
        max_safe_N = max_uint32 // self.number_of_clauses

        # Pack clauses
        packed_clauses_gpu, includes_gpu = self._pack_clauses_gpu(block_size, grid_size)

        class_sums = np.zeros((N, self.number_of_outputs), dtype=np.float32)

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
                *kernel_config(X_safe.shape[0] * self.number_of_clauses, device_props, block_size, grid_size),
                packed_clauses_gpu,
                self.clause_weights_gpu,
                includes_gpu,
                X_gpu,
                np.int32(X_safe.shape[0]),
                class_sums_gpu,
                self.literal_mask_gpu,
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
        return (ta_states > ((self.number_of_ta_states - 1) // 2)).astype(np.uint32)

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

    ######## TRANSFORM #######
    def transform(self, X, is_X_encoded: bool = False, block_size: int | None = None, grid_size: int | None = None):
        encoded_X = X if is_X_encoded else self.encode(X)
        if block_size is None:
            block_size = self.block_size
        if grid_size is None:
            grid_size = self.grid_size

        N = encoded_X.shape[0]
        max_uint32 = np.iinfo(np.uint32).max
        max_safe_N = max_uint32 // self.number_of_clauses
        if N > max_safe_N:
            raise OverflowError(
                f"X has too many samples ({N}). Maximum of {max_safe_N} samples can be processed with current number_of_clauses. Call this method multiple times with smaller batches of X."
            )

        encoded_X_gpu = mem_alloc(N * self.number_of_patches * self.number_of_literal_chunks * 4)
        memcpy_htod(encoded_X_gpu, encoded_X)

        packed_clauses_gpu, includes_gpu = self._pack_clauses_gpu(block_size, grid_size)

        clause_outputs_gpu = mem_alloc(N * self.number_of_clauses * 4)
        self.kernel_transform.prepared_call(
            *kernel_config(N * self.number_of_clauses, device_props, block_size, grid_size),
            packed_clauses_gpu,
            includes_gpu,
            encoded_X_gpu,
            np.int32(N),
            clause_outputs_gpu,
            self.literal_mask_gpu,
        )
        ctx.synchronize()

        clause_outputs = np.zeros((N * self.number_of_clauses), dtype=np.uint32)
        memcpy_dtoh(clause_outputs, clause_outputs_gpu)

        encoded_X_gpu.free()
        packed_clauses_gpu.free()
        includes_gpu.free()
        clause_outputs_gpu.free()

        return clause_outputs.reshape((N, self.number_of_clauses))

    def transform_patchwise(
        self, X, is_X_encoded: bool = False, block_size: int | None = None, grid_size: int | None = None
    ):
        encoded_X = X if is_X_encoded else self.encode(X)
        if block_size is None:
            block_size = self.block_size

        if grid_size is None:
            grid_size = self.grid_size

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

        packed_clauses_gpu, includes_gpu = self._pack_clauses_gpu(block_size, grid_size)

        clause_outputs_gpu = mem_alloc(N * self.number_of_clauses * self.number_of_patches * 4)
        self.kernel_transform_patchwise.prepared_call(
            *kernel_config(N * self.number_of_clauses * self.number_of_patches, device_props, block_size, grid_size),
            packed_clauses_gpu,
            includes_gpu,
            encoded_X_gpu,
            np.int32(N),
            clause_outputs_gpu,
            self.literal_mask_gpu,
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
