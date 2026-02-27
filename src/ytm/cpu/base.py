import pathlib
import shutil
import subprocess
import tempfile
import os
from typing import Literal, TypedDict, Unpack

import numpy as np
from ctypes import CDLL, POINTER, c_uint32, c_int, c_float, c_double, c_uint64, byref
from tqdm import tqdm


def get_kernel(filename: str, current_dir: pathlib.Path) -> str:
    kernel_path = current_dir / filename
    with open(kernel_path, "r") as f:
        return f.read()


def compile_code(code: str, output_path: str) -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".c", delete=False) as f:
        f.write(code)
        temp_c = f.name

    try:
        if shutil.which("clang"):
            subprocess.run(
                ["clang", "-shared", "-fPIC", "-O3", "-fopenmp", "-lomp", "-pthread", temp_c, "-o", output_path],
                check=True,
                capture_output=True,
            )
        elif shutil.which("gcc"):
            subprocess.run(
                ["gcc", "-shared", "-fPIC", "-O3", "-lgomp", "-pthread", "-fopenmp", temp_c, "-o", output_path],
                check=True,
                capture_output=True,
            )
        else:
            raise EnvironmentError("Neither clang nor gcc compiler found.")
    except subprocess.CalledProcessError as e:
        print("Compilation failed:")
        print(e.stderr.decode())
        raise
    finally:
        os.unlink(temp_c)

    return output_path


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
    num_threads: int


class FitOptArgs(TypedDict, total=False):
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

        self.init_args = {
            "number_of_clauses_per_class": number_of_clauses_per_class,
            "T": T,
            "s": s,
            "dim": dim,
            "n_classes": n_classes,
        }
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
            "num_threads": opt_args.get("num_threads", 1),
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
        self.num_threads = self.opt_args["num_threads"]

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

        self._init_lib()

    def _init_lib(self):
        self.cpu_macro_string = f"""
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
        """
        current_dir = pathlib.Path(__file__).parent
        kernel_str = get_kernel("src.c", current_dir)

        with tempfile.TemporaryDirectory() as tmpdir:
            so_path = os.path.join(tmpdir, "libtm.so")
            compile_code(self.cpu_macro_string + kernel_str, so_path)
            self.lib = CDLL(so_path)

        self.lib.set_H(self.h.ctypes.data_as(POINTER(c_double)))
        self.lib.set_num_threads(self.num_threads)

        self.lib.encode_batch.argtypes = [POINTER(c_int), POINTER(c_uint32), c_int]
        self.lib.encode_batch.restype = None

        self.lib.pack_clauses.argtypes = [POINTER(c_uint32), POINTER(c_uint32), POINTER(c_int)]
        self.lib.pack_clauses.restype = None

        self.lib.fast_eval.argtypes = [
            POINTER(c_uint32),
            POINTER(c_int),
            POINTER(c_uint32),
            POINTER(c_uint32),
            POINTER(c_uint32),
            POINTER(c_uint32),
            c_int,
        ]
        self.lib.fast_eval.restype = None

        self.lib.select_active.argtypes = [
            POINTER(c_uint64),
            POINTER(c_float),
            POINTER(c_uint32),
            POINTER(c_int),
            POINTER(c_int),
            POINTER(c_float),
            POINTER(c_float),
        ]
        self.lib.select_active.restype = None

        self.lib.calc_class_sums_infer_batch.argtypes = [
            POINTER(c_uint32),
            POINTER(c_float),
            POINTER(c_int),
            POINTER(c_uint32),
            c_int,
            POINTER(c_float),
            POINTER(c_uint32),
        ]
        self.lib.calc_class_sums_infer_batch.restype = None

        self.lib.transform.argtypes = [
            POINTER(c_uint32),
            POINTER(c_int),
            POINTER(c_uint32),
            c_int,
            POINTER(c_uint32),
            POINTER(c_uint32),
        ]
        self.lib.transform.restype = None

        self.lib.transform_patchwise.argtypes = [
            POINTER(c_uint32),
            POINTER(c_int),
            POINTER(c_uint32),
            c_int,
            POINTER(c_uint32),
            POINTER(c_uint32),
        ]
        self.lib.transform_patchwise.restype = None

        self.lib.evidence_to_update_prob.argtypes = [
            POINTER(c_float),
            POINTER(c_float),
            POINTER(c_int),
            POINTER(c_double),
            POINTER(c_double),
            c_int,
            POINTER(c_double),
            POINTER(c_double),
        ]
        self.lib.evidence_to_update_prob.restype = None

        self.lib.clause_update.argtypes = [
            POINTER(c_uint64),
            POINTER(c_uint32),
            POINTER(c_float),
            POINTER(c_int),
            POINTER(c_int),
            POINTER(c_uint32),
            POINTER(c_uint32),
            POINTER(c_uint32),
            POINTER(c_int),
            POINTER(c_double),
            POINTER(c_double),
            c_int,
            c_int,
        ]
        self.lib.clause_update.restype = None

        self._init_clauses()
        self._init_weights()

        self.literal_mask = np.full((self.number_of_literal_chunks,), 0xFFFFFFFF, dtype=np.uint32)

    def _init_clauses(self):
        if self.initial_state == -1:
            self.ta_states = self.rng.integers(
                low=0,
                high=self.number_of_ta_states,
                size=(self.number_of_clauses * self.number_of_literals),
                dtype=np.uint32,
            )
        else:
            self.ta_states = np.full(
                (self.number_of_clauses * self.number_of_literals),
                self.initial_state,
                dtype=np.uint32,
            )

    def _init_weights(self):
        num_neg_polarity = self.number_of_clauses_per_class // 2
        self.clause_weights = np.zeros((self.number_of_outputs, self.number_of_clauses_per_class), dtype=np.float32)
        for i in range(self.number_of_outputs):
            wt = np.ones((self.number_of_clauses_per_class,), dtype=np.float32) * self.initial_weight
            if self.init_neg_weights:
                wt[num_neg_polarity:] *= -1.0
            self.clause_weights[i, :] = self.rng.permutation(wt) if self.coalesced else wt

        self.patch_weights = np.zeros((self.number_of_clauses, self.number_of_patches), dtype=np.int32)

    def encode(
        self,
        X: np.ndarray,
        input_type: Literal["binary", "ternary"] = "binary",
    ) -> np.ndarray[tuple[int, int, int], np.dtype[np.uint32]]:
        X = X.astype(np.int32).reshape((X.shape[0], self.dim[0], self.dim[1], self.dim[2]))

        if input_type == "binary":
            assert np.min(X) >= 0 and np.max(X) <= 1, "X must be binary (contain only 0 or 1)."
            X = X * 2 - 1
        elif input_type == "ternary":
            assert np.min(X) >= -1 and np.max(X) <= 1, "X must be ternary (contain only -1, 0 or 1)."
        else:
            raise ValueError("input_type must be 'binary' or 'ternary'.")

        N = X.shape[0]
        encoded_X = np.empty((N, self.number_of_patches, self.number_of_literal_chunks), dtype=np.uint32)

        X_flat = X.ravel()
        encoded_X_flat = encoded_X.ravel()

        self.lib.encode_batch(
            X_flat.ctypes.data_as(POINTER(c_int)),
            encoded_X_flat.ctypes.data_as(POINTER(c_uint32)),
            c_int(N),
        )

        return encoded_X

    def _target_sampling(self, Y, label_sampling: bool | int):
        N = Y.shape[0]
        targets = np.zeros_like(Y, dtype=np.int32)

        if label_sampling is False or label_sampling <= 0:
            targets[Y > 0] = 1
            p = self.q / max(1, self.number_of_outputs - 1)
            for i in range(N):
                false_classes = np.where(Y[i, :] <= 0)[0]
                if len(false_classes) > 0:
                    not_skip = self.rng.random(size=len(false_classes)) <= p
                    targets[i, false_classes[not_skip]] = -1
        else:
            per_class_counts = np.sum(Y > 0, axis=0)

            if isinstance(label_sampling, bool) and label_sampling is True:
                min_cnt = np.min(per_class_counts)
            else:
                min_cnt = min(N, label_sampling)

            for i in range(self.number_of_outputs):
                inds = np.where(Y[:, i] > 0)[0]
                if len(inds) > min_cnt:
                    sel = self.rng.choice(inds, size=min_cnt, replace=False)
                    targets[sel, i] = 1
                else:
                    targets[inds, i] = 1

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
        lit_msk = np.full((self.number_of_literal_chunks,), 0xFFFFFFFF, dtype=np.uint32)
        if len(literal_inds) > 0:
            for litid in literal_inds:
                chunk_nr = litid // 32
                chunk_pos = litid % 32
                lit_msk[chunk_nr] &= ~(np.uint32(1) << chunk_pos)
        self.literal_mask = lit_msk

    def _validate_fit_args(self, **opt_args: Unpack[FitOptArgs]):
        unexpected_args = set(opt_args.keys()) - set(FitOptArgs.__annotations__.keys())
        if unexpected_args:
            raise TypeError(f"Unexpected keyword arguments: {unexpected_args}")
        args = {
            "clause_drop_p": opt_args.get("clause_drop_p", 0.0),
            "shuffle": opt_args.get("shuffle", True),
            "label_sampling": opt_args.get("label_sampling", False),
            "g_pos": _float_or_list_to_array(opt_args.get("g_pos", 1.0), self.number_of_outputs),
            "g_neg": _float_or_list_to_array(opt_args.get("g_neg", 1.0), self.number_of_outputs),
            "log": opt_args.get("log", False),
        }
        return args

    def _pack_clauses_cpu(self):
        packed_clauses = np.zeros((self.number_of_clauses, self.number_of_literal_chunks), dtype=np.uint32)
        num_includes = np.zeros(self.number_of_clauses, dtype=np.int32)

        self.lib.pack_clauses(
            self.ta_states.ctypes.data_as(POINTER(c_uint32)),
            packed_clauses.ctypes.data_as(POINTER(c_uint32)),
            num_includes.ctypes.data_as(POINTER(c_int)),
        )

        return packed_clauses, num_includes

    def _fit_sample(self, encoded_X, targets, e, clause_drop_mask, g_pos, g_neg):
        packed_clauses, num_includes = self._pack_clauses_cpu()

        clause_outputs = np.zeros((self.number_of_clauses, self.number_of_patches), dtype=np.uint32)
        self.lib.fast_eval(
            packed_clauses.ctypes.data_as(POINTER(c_uint32)),
            num_includes.ctypes.data_as(POINTER(c_int)),
            clause_drop_mask.ctypes.data_as(POINTER(c_uint32)),
            self.literal_mask.ctypes.data_as(POINTER(c_uint32)),
            encoded_X.ctypes.data_as(POINTER(c_uint32)),
            clause_outputs.ctypes.data_as(POINTER(c_uint32)),
            c_int(e),
        )

        positive_evidence = np.zeros(self.number_of_outputs, dtype=np.float32)
        negative_evidence = np.zeros(self.number_of_outputs, dtype=np.float32)
        selected_patch_ids = np.full(self.number_of_clauses, -1, dtype=np.int32)

        num_threads = self.num_threads
        rng_states = np.random.randint(0, 2**31, size=num_threads, dtype=np.uint64)

        self.lib.select_active(
            rng_states.ctypes.data_as(POINTER(c_uint64)),
            self.clause_weights.ctypes.data_as(POINTER(c_float)),
            clause_outputs.ctypes.data_as(POINTER(c_uint32)),
            self.patch_weights.ravel().ctypes.data_as(POINTER(c_int)),
            selected_patch_ids.ctypes.data_as(POINTER(c_int)),
            positive_evidence.ctypes.data_as(POINTER(c_float)),
            negative_evidence.ctypes.data_as(POINTER(c_float)),
        )

        pprob = np.zeros(self.number_of_outputs, dtype=np.float64)
        nprob = np.zeros(self.number_of_outputs, dtype=np.float64)

        self.lib.evidence_to_update_prob(
            positive_evidence.ctypes.data_as(POINTER(c_float)),
            negative_evidence.ctypes.data_as(POINTER(c_float)),
            targets.ctypes.data_as(POINTER(c_int)),
            g_pos.ctypes.data_as(POINTER(c_double)),
            g_neg.ctypes.data_as(POINTER(c_double)),
            c_int(e),
            pprob.ctypes.data_as(POINTER(c_double)),
            nprob.ctypes.data_as(POINTER(c_double)),
        )

        self.lib.clause_update(
            rng_states.ctypes.data_as(POINTER(c_uint64)),
            self.ta_states.ctypes.data_as(POINTER(c_uint32)),
            self.clause_weights.ctypes.data_as(POINTER(c_float)),
            selected_patch_ids.ctypes.data_as(POINTER(c_int)),
            num_includes.ctypes.data_as(POINTER(c_int)),
            clause_drop_mask.ctypes.data_as(POINTER(c_uint32)),
            self.literal_mask.ctypes.data_as(POINTER(c_uint32)),
            encoded_X.ctypes.data_as(POINTER(c_uint32)),
            targets.ctypes.data_as(POINTER(c_int)),
            pprob.ctypes.data_as(POINTER(c_double)),
            nprob.ctypes.data_as(POINTER(c_double)),
            c_int(e),
            c_int(num_threads),
        )

    def _fit(self, encoded_X, encoded_Y, **opt_args: Unpack[FitOptArgs]):
        N = encoded_X.shape[0]
        args = self._validate_fit_args(**opt_args)

        iota = np.arange(N)
        if args["shuffle"]:
            self.rng.shuffle(iota)
        encoded_X = encoded_X[iota]
        encoded_Y = encoded_Y[iota]

        targets = self._target_sampling((encoded_Y > 0).astype(np.int32), args["label_sampling"])

        clause_drop_p = args["clause_drop_p"]
        if clause_drop_p > 0.0:
            clause_drop_mask = (self.rng.random(self.number_of_clauses) <= clause_drop_p).astype(np.uint32)
        else:
            clause_drop_mask = np.zeros(self.number_of_clauses, dtype=np.uint32)

        g_pos = args["g_pos"]
        g_neg = args["g_neg"]

        logged_data = {}
        log = opt_args.get("log", False)
        if log:
            logged_data["opt_args"] = opt_args
            logged_data["iota"] = iota
            logged_data["targets"] = targets
            class_sums = np.zeros((N, self.number_of_outputs), dtype=np.float32)
            update_probs = np.zeros((N, self.number_of_outputs), dtype=np.float64)

        pbar = tqdm(range(N), desc="Fitting Batch", leave=False, dynamic_ncols=True)
        for e in pbar:
            if np.all(targets[e, :] == 0):
                continue

            self._fit_sample(encoded_X, targets, e, clause_drop_mask, g_pos, g_neg)

            if log:
                pass

        return logged_data

    def _score_batch(self, encoded_X):
        N = encoded_X.shape[0]
        packed_clauses, num_includes = self._pack_clauses_cpu()

        class_sums = np.zeros((N, self.number_of_outputs), dtype=np.float32)

        self.lib.calc_class_sums_infer_batch(
            packed_clauses.ctypes.data_as(POINTER(c_uint32)),
            self.clause_weights.ctypes.data_as(POINTER(c_float)),
            num_includes.ctypes.data_as(POINTER(c_int)),
            encoded_X.ctypes.data_as(POINTER(c_uint32)),
            c_int(N),
            class_sums.ctypes.data_as(POINTER(c_float)),
            self.literal_mask.ctypes.data_as(POINTER(c_uint32)),
        )

        return class_sums

    def get_ta_state(self):
        if self.coalesced:
            return self.ta_states.reshape((self.number_of_clauses, self.number_of_literals))
        else:
            return self.ta_states.reshape(
                (self.number_of_clause_banks, self.number_of_clauses_per_class, self.number_of_literals)
            )

    def get_literals(self):
        ta_states = self.get_ta_state()
        return (ta_states > ((self.number_of_ta_states - 1) // 2)).astype(np.uint32)

    def get_weights(self) -> np.ndarray[tuple[int, int], np.dtype[np.float32]]:
        return self.clause_weights.copy()

    def get_patch_weights(self):
        if self.coalesced:
            return self.patch_weights.reshape((self.number_of_clauses, self.number_of_patches))
        else:
            return self.patch_weights.reshape(
                (self.number_of_clause_banks, self.number_of_clauses_per_class, self.number_of_patches)
            )

    def transform(self, X, is_X_encoded: bool = False):
        encoded_X = X if is_X_encoded else self.encode(X)

        N = encoded_X.shape[0]

        packed_clauses, num_includes = self._pack_clauses_cpu()

        clause_outputs = np.zeros((N, self.number_of_clauses), dtype=np.uint32)
        self.lib.transform(
            packed_clauses.ctypes.data_as(POINTER(c_uint32)),
            num_includes.ctypes.data_as(POINTER(c_int)),
            encoded_X.ctypes.data_as(POINTER(c_uint32)),
            c_int(N),
            clause_outputs.ctypes.data_as(POINTER(c_uint32)),
            self.literal_mask.ctypes.data_as(POINTER(c_uint32)),
        )

        return clause_outputs

    def transform_patchwise(self, X, is_X_encoded: bool = False):
        encoded_X = X if is_X_encoded else self.encode(X)

        N = encoded_X.shape[0]

        packed_clauses, num_includes = self._pack_clauses_cpu()

        clause_outputs = np.zeros((N, self.number_of_clauses, self.number_of_patches), dtype=np.uint32)
        self.lib.transform_patchwise(
            packed_clauses.ctypes.data_as(POINTER(c_uint32)),
            num_includes.ctypes.data_as(POINTER(c_int)),
            encoded_X.ctypes.data_as(POINTER(c_uint32)),
            c_int(N),
            clause_outputs.ctypes.data_as(POINTER(c_uint32)),
            self.literal_mask.ctypes.data_as(POINTER(c_uint32)),
        )

        return clause_outputs

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

    def get_state_dict(self):
        state_dict = {
            "ta_state": self.ta_states.copy(),
            "clause_weights": self.clause_weights.copy(),
            "patch_weights": self.patch_weights.copy(),
            "min_y": self.min_y,
            "max_y": self.max_y,
        }
        return state_dict

    def load_state_dict(self, state):
        state_dict = state["state"]
        self.ta_states = state_dict["ta_state"]
        self.clause_weights = state_dict["clause_weights"]
        self.patch_weights = state_dict["patch_weights"]
        self.min_y = state_dict["min_y"]
        self.max_y = state_dict["max_y"]
