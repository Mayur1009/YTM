import os
import shutil
import subprocess
import tempfile
from ctypes import CDLL, POINTER, c_float, c_int, c_int32, c_uint32, c_int8

import numpy as np
from tqdm import tqdm

from .. import BaseDevice, FitBuffers

int8_p = POINTER(c_int8)
int32_p = POINTER(c_int32)
uint32_p = POINTER(c_uint32)
float_p = POINTER(c_float)


class CPUDevice(BaseDevice):
    def dev_init(self):
        self.np_rng = np.random.default_rng(self.args.seed)
        self.rng = np.array([self.args.seed + i for i in range(self.args.n_threads)], dtype=np.uint32)
        self.p_rng = self.rng.ctypes.data_as(uint32_p)

        self._init_clauses()
        self._init_weights()

        cur_dir = os.path.dirname(os.path.abspath(__file__))
        so_file = self._compile_code(
            os.path.join(cur_dir, "src.c"),
            header=f"""
            #define USE_OMP {1 if self.args.n_threads > 1 else 0}
            #define TOTAL_CLAUSES {int(self.total_clauses)}
            #define THRESH {int(self.args.T)}
            #define S {float(self.args.s)}
            #define DIM0 {int(self.args.dim[0])}
            #define DIM1 {int(self.args.dim[1])}
            #define DIM2 {int(self.args.dim[2])}
            #define CLASSES {int(self.args.n_classes)}
            #define PATCH_DIM0 {int(self.args.patch_dim[0])}
            #define PATCH_DIM1 {int(self.args.patch_dim[1])}
            #define WEIGHTED {1 if self.args.weighted else 0}
            #define MAX_WEIGHT {float(self.args.max_weight)}f
            #define COALESCED {1 if self.args.coalesced else 0}
            #define NEGATED_LITERALS {1 if self.args.negated_literals else 0}
            #define POSITION_LITERALS {1 if self.args.position_literals else 0}
            #define NEGATIVE_CLAUSES {1 if self.args.negative_clauses else 0}
            #define ALLOW_POLARITY_CHANGE {1 if self.args.allow_polarity_change else 0}
            #define MAX_INCLUDED_LITERALS {int(self.args.max_included_literals)}
            #define MAX_TA_STATE {int(self.args.n_states - 1)}
            #define INCLUDE_STATE {int(self.args.include_state)}
            #define TYPE1A_FB {0 if self.args.skip_t1a_fb else 1}
            #define TYPE1B_FB {0 if self.args.skip_t1b_fb else 1}
            #define TYPE2_FB {0 if self.args.skip_t2_fb else 1}
            #define PATCHES {int(self.n_patches)}
            #define LITERALS {int(self.n_literals)}
            """,
        )
        dll = CDLL(so_file)
        self.lib_encode = dll.encode
        self.lib_decode = dll.decode
        self.lib_pack_clauses = dll.pack_clauses
        self.lib_eval_clauses = dll.eval_clauses
        self.lib_select_patch = dll.select_patch_and_count_votes
        self.lib_calc_update_prob = dll.evidence_to_update_prob
        self.lib_update_clauses = dll.update_clauses
        self.lib_clause_inference = dll.clause_inference

        self.lib_encode.argtypes = [int8_p, c_int, uint32_p]
        self.lib_decode.argtypes = [uint32_p, c_int, int8_p]
        self.lib_pack_clauses.argtypes = [uint32_p, uint32_p, uint32_p]
        self.lib_eval_clauses.argtypes = [uint32_p, uint32_p, uint32_p, uint32_p, c_int, uint32_p]
        self.lib_select_patch.argtypes = [uint32_p, float_p, uint32_p, int32_p, int32_p, float_p, float_p]
        self.lib_calc_update_prob.argtypes = [float_p, float_p, int8_p, c_int, float_p]
        self.lib_update_clauses.argtypes = [
            uint32_p,
            int32_p,
            uint32_p,
            uint32_p,
            uint32_p,
            int8_p,
            float_p,
            c_int,
            uint32_p,
            float_p,
        ]
        self.lib_clause_inference.argtypes = [uint32_p, float_p, uint32_p, uint32_p, c_int, float_p]

        if self.args.n_threads > 1:
            lib_set_num_threads = dll.set_num_threads
            lib_set_num_threads.argtypes = [c_int]
            lib_set_num_threads(self.args.n_threads)

    def _init_clauses(self):
        include_state = self.args.include_state if self.args.include_state is not None else 128
        self.ta_states = np.full(
            (self.total_clauses, self.n_literals),
            include_state - 1,
            dtype=np.uint32,
        )

    def _init_weights(self):
        n_neg_polarity = self.args.n_clauses // 2
        self.clause_weights = np.zeros((self.args.n_classes, self.args.n_clauses), dtype=np.float32)
        for i in range(self.args.n_classes):
            wt = np.ones((self.args.n_clauses,), dtype=np.float32) * 1.0
            wt[n_neg_polarity:] *= -1.0
            self.clause_weights[i, :] = self.np_rng.permutation(wt) if self.args.coalesced else wt

        self.patch_weights = np.zeros((self.total_clauses, self.n_patches), dtype=np.int32)

    def _compile_code(self, fname, header: str):
        with open(fname, "r") as f:
            code = f.read()

        with tempfile.NamedTemporaryFile(suffix=".c", mode="w", delete=False) as f:
            f.write(header)
            f.write(code)
            c_file = f.name

        so_file = c_file.replace(".c", ".so")

        if shutil.which("clang"):
            compiler = "clang"
            base_args = ["-shared", "-fPIC", "-O3", "-ffast-math", "-march=native"]
            omp_args = ["-fopenmp", "-lomp"]
        elif shutil.which("gcc"):
            compiler = "gcc"
            base_args = ["-shared", "-fPIC", "-O3", "-ffast-math", "-march=native"]
            omp_args = ["-fopenmp", "-lgomp"]
        else:
            raise RuntimeError("No suitable C compiler found (clang or gcc)")

        if self.args.n_threads > 1:
            try:
                subprocess.run(
                    [compiler] + base_args + omp_args + [c_file, "-o", so_file],
                    check=True,
                    capture_output=True,
                )
            except subprocess.CalledProcessError as e:
                raise RuntimeError(
                    f"Failed to compile. Compiler Output:\n{e.stdout.decode()}\nError: {e.stderr.decode()}"
                )
        else:
            try:
                subprocess.run(
                    [compiler] + base_args + [c_file, "-o", so_file],
                    check=True,
                    capture_output=True,
                )
            except subprocess.CalledProcessError as e:
                raise RuntimeError(
                    f"Failed to compile. Compiler Output:\n{e.stdout.decode()}\nError: {e.stderr.decode()}"
                )

        return so_file

    def encode(self, X: np.ndarray[tuple[int, int], np.dtype[np.int8]]):
        N = X.shape[0]
        encoded_X = np.zeros((N, self.n_patches, self.n_literal_chunks), dtype=np.uint32)

        self.lib_encode(
            X.astype(np.int8).ctypes.data_as(int8_p),
            N,
            encoded_X.ctypes.data_as(uint32_p),
        )

        return encoded_X

    def decode(self, encoded_X):
        N = encoded_X.shape[0]
        X = np.zeros((N, self.args.dim[0] * self.args.dim[1] * self.args.dim[2]), dtype=np.int8)

        self.lib_decode(
            encoded_X.ctypes.data_as(uint32_p),
            N,
            X.ctypes.data_as(int8_p),
        )

        return X

    def prepare_fit_buffers(self, encoded_X, targets, clause_drop_mask) -> FitBuffers:
        return FitBuffers(
            encoded_X=encoded_X.astype(np.uint32),
            targets=targets.astype(np.int8),
            packed_clauses=np.empty((self.total_clauses, self.n_literal_chunks), dtype=np.uint32),
            n_includes=np.empty((self.total_clauses,), dtype=np.uint32),
            clause_outputs=np.empty((self.total_clauses * self.n_patches,), dtype=np.uint32),
            selected_patch_ids=np.empty((self.total_clauses,), dtype=np.int32),
            pos_votes=np.empty((self.args.n_classes,), dtype=np.float32),
            neg_votes=np.empty((self.args.n_classes,), dtype=np.float32),
            update_probs=np.empty((self.args.n_classes,), dtype=np.float32),
            clause_drop_mask=clause_drop_mask.astype(np.uint32),
        )

    def pack_clauses(self, packed_clauses: np.ndarray, n_includes: np.ndarray):
        packed_clauses.fill(0)
        self.lib_pack_clauses(
            self.ta_states.ctypes.data_as(uint32_p),
            packed_clauses.ctypes.data_as(uint32_p),
            n_includes.ctypes.data_as(uint32_p),
        )

    def eval_clauses(
        self,
        packed_clauses: np.ndarray,
        n_includes: np.ndarray,
        clause_drop_mask: np.ndarray,
        clause_outputs: np.ndarray,
        encoded_X: np.ndarray,
        e: int,
    ):
        self.lib_eval_clauses(
            packed_clauses.ctypes.data_as(uint32_p),
            n_includes.ctypes.data_as(uint32_p),
            clause_drop_mask.ctypes.data_as(uint32_p),
            encoded_X.ctypes.data_as(uint32_p),
            c_int(e),
            clause_outputs.ctypes.data_as(uint32_p),
        )

    def select_patch_and_count_votes(
        self,
        clause_outputs: np.ndarray,
        selected_patch_ids: np.ndarray,
        pos_votes: np.ndarray,
        neg_votes: np.ndarray,
    ):
        pos_votes.fill(0)
        neg_votes.fill(0)
        self.lib_select_patch(
            self.p_rng,
            self.clause_weights.ctypes.data_as(float_p),
            clause_outputs.ctypes.data_as(uint32_p),
            self.patch_weights.ctypes.data_as(int32_p),
            selected_patch_ids.ctypes.data_as(int32_p),
            pos_votes.ctypes.data_as(float_p),
            neg_votes.ctypes.data_as(float_p),
        )

    def calc_update_prob(
        self, pos_votes: np.ndarray, neg_votes: np.ndarray, targets: np.ndarray, update_probs: np.ndarray, e: int
    ):
        self.lib_calc_update_prob(
            pos_votes.ctypes.data_as(float_p),
            neg_votes.ctypes.data_as(float_p),
            targets.ctypes.data_as(int8_p),
            c_int(e),
            update_probs.ctypes.data_as(float_p),
        )

    def update_clauses(
        self,
        n_includes: np.ndarray,
        selected_patch_ids: np.ndarray,
        clause_drop_mask: np.ndarray,
        update_probs: np.ndarray,
        encoded_X: np.ndarray,
        targets: np.ndarray,
        e: int,
    ):
        self.lib_update_clauses(
            self.p_rng,
            selected_patch_ids.ctypes.data_as(int32_p),
            n_includes.ctypes.data_as(uint32_p),
            clause_drop_mask.ctypes.data_as(uint32_p),
            encoded_X.ctypes.data_as(uint32_p),
            targets.ctypes.data_as(int8_p),
            update_probs.ctypes.data_as(float_p),
            c_int(e),
            self.ta_states.ctypes.data_as(uint32_p),
            self.clause_weights.ctypes.data_as(float_p),
        )

    def infer(self, encoded_X: np.ndarray, batch_size: int = -1) -> np.ndarray:
        N = encoded_X.shape[0]
        if batch_size == -1:
            batch_size = N

        packed_clauses = np.empty((self.total_clauses, self.n_literal_chunks), dtype=np.uint32)
        n_includes = np.empty((self.total_clauses,), dtype=np.uint32)

        self.pack_clauses(packed_clauses, n_includes)

        class_sums = np.zeros((N, self.args.n_classes), dtype=np.float32)
        for i in tqdm(range(0, N, batch_size), desc="Inference", leave=False, dynamic_ncols=True):
            batch = encoded_X[i : i + batch_size]
            cs_batch = np.zeros((batch.shape[0], self.args.n_classes), dtype=np.float32)

            self.lib_clause_inference(
                packed_clauses.ctypes.data_as(uint32_p),
                self.clause_weights.ctypes.data_as(float_p),
                n_includes.ctypes.data_as(uint32_p),
                batch.ctypes.data_as(uint32_p),
                c_int(batch.shape[0]),
                cs_batch.ctypes.data_as(float_p),
            )

            class_sums[i : i + batch.shape[0]] = cs_batch

        return class_sums

    def get_state_dict(self):
        return {
            "ta_states": self.ta_states,
            "clause_weights": self.clause_weights,
            "patch_weights": self.patch_weights,
        }

    def load_state_dict(self, state_dict):
        self.ta_states = state_dict["ta_states"]
        self.clause_weights = state_dict["clause_weights"]
        self.patch_weights = state_dict["patch_weights"]
