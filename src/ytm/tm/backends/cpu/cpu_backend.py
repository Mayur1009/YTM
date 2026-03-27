import os
import shutil
import subprocess
import tempfile
import warnings
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

        self.header = f"""
            #define USE_OMP {1 if self.args.n_threads > 1 else 0}
            #define TOTAL_CLAUSES {int(self.total_clauses)}
            #define THRESH {int(self.args.T)}
            #define S {float(self.args.s)}
            #define HEIGHT {int(self.args.dim[0])}
            #define WIDTH {int(self.args.dim[1])}
            #define DEPTH {int(self.args.dim[2])}
            #define CLASSES {int(self.args.n_classes)}
            #define PATCH_HEIGHT {int(self.args.patch_dim[0])}
            #define PATCH_WIDTH {int(self.args.patch_dim[1])}
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
            #define TRACK_PATCH_WEIGHTS {1 if self.args.track_patch_weights else 0}
            #define BOOST_TP_FB {1 if self.args.boost_tp_fb else 0}
        """

        cur_dir = os.path.dirname(os.path.abspath(__file__))
        so_file = self._compile_code(os.path.join(cur_dir, "src.c"), self.header)
        dll = CDLL(so_file)
        self.lib_encode = dll.encode
        self.lib_pack_clauses = dll.pack_clauses
        self.lib_eval_clauses = dll.eval_clauses
        self.lib_select_patch = dll.select_patch_and_count_votes
        self.lib_calc_update_prob = dll.evidence_to_update_prob
        self.lib_update_clauses = dll.update_clauses
        self.lib_clause_inference = dll.clause_inference

        self.lib_encode.argtypes = [int8_p, c_int, uint32_p]
        self.lib_pack_clauses.argtypes = [uint32_p, uint32_p, uint32_p]
        self.lib_eval_clauses.argtypes = [uint32_p, uint32_p, int8_p, uint32_p, c_int, int8_p]
        self.lib_select_patch.argtypes = [uint32_p, float_p, int8_p, int32_p, int32_p, float_p, float_p]
        self.lib_calc_update_prob.argtypes = [float_p, float_p, float_p, c_int, float_p]
        self.lib_update_clauses.argtypes = [
            uint32_p,
            int32_p,
            uint32_p,
            int8_p,
            uint32_p,
            float_p,
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
        self.ta_states = np.full(
            (self.total_clauses, self.n_literals),
            self.args.include_state - 1,
            dtype=np.uint32,
        )

    def _init_weights(self):
        n_neg_polarity = self.args.n_clauses // 2
        self.clause_weights = np.zeros((self.args.n_classes, self.args.n_clauses), dtype=np.float32)
        for i in range(self.args.n_classes):
            wt = np.ones((self.args.n_clauses,), dtype=np.float32) * 1.0
            wt[n_neg_polarity:] *= -1.0
            self.clause_weights[i, :] = self.np_rng.permutation(wt) if self.args.coalesced else wt

        if self.args.track_patch_weights:
            self.patch_weights = np.zeros((self.total_clauses, self.n_patches), dtype=np.int32)
        else:
            self.patch_weights = np.zeros((1, 1), dtype=np.int32)

    def _compile_code(self, fname, header: str):
        with open(fname, "r") as f:
            code = f.read()

        with tempfile.NamedTemporaryFile(suffix=".c", mode="w", delete=False) as f:
            f.write(header)
            f.write(code)
            c_file = f.name

        so_file = c_file.replace(".c", ".so")

        compiler_flags = list(self.args.compile_flags)

        if shutil.which("clang"):
            compiler = "clang"
            omp_args = ["-fopenmp", "-lomp"]
        elif shutil.which("gcc"):
            compiler = "gcc"
            omp_args = ["-fopenmp", "-lgomp"]
        else:
            raise RuntimeError("No suitable C compiler found (clang or gcc)")

        if self.args.n_threads > 1:
            try:
                subprocess.run(
                    [compiler] + compiler_flags + omp_args + [c_file, "-o", so_file],
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
                    [compiler] + compiler_flags + [c_file, "-o", so_file],
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

    def prepare_fit_buffers(self, encoded_X, targets, clause_drop_mask) -> FitBuffers:
        return FitBuffers(
            encoded_X=encoded_X.astype(np.uint32),
            targets=targets.astype(np.float32),
            packed_clauses=np.empty((self.total_clauses, self.n_literal_chunks), dtype=np.uint32),
            n_includes=np.empty((self.total_clauses,), dtype=np.uint32),
            clause_outputs=np.empty((self.total_clauses * self.n_patches,), dtype=np.int8),
            selected_patch_ids=np.empty((self.total_clauses,), dtype=np.int32),
            pos_votes=np.empty((self.args.n_classes,), dtype=np.float32),
            neg_votes=np.empty((self.args.n_classes,), dtype=np.float32),
            update_probs=np.empty((self.args.n_classes,), dtype=np.float32),
            clause_drop_mask=clause_drop_mask.astype(np.int8),
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
            clause_drop_mask.ctypes.data_as(int8_p),
            encoded_X.ctypes.data_as(uint32_p),
            c_int(e),
            clause_outputs.ctypes.data_as(int8_p),
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
            clause_outputs.ctypes.data_as(int8_p),
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
            targets.ctypes.data_as(float_p),
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
            clause_drop_mask.ctypes.data_as(int8_p),
            encoded_X.ctypes.data_as(uint32_p),
            targets.ctypes.data_as(float_p),
            update_probs.ctypes.data_as(float_p),
            c_int(e),
            self.ta_states.ctypes.data_as(uint32_p),
            self.clause_weights.ctypes.data_as(float_p),
        )

    def fit_epoch(self, encoded_X, targets, clause_drop_p):
        N = encoded_X.shape[0]
        if clause_drop_p > 0.0:
            clause_drop_mask = (self.np_rng.random(self.total_clauses) <= clause_drop_p).astype(np.int8)
        else:
            clause_drop_mask = np.zeros(self.total_clauses, dtype=np.int8)

        dev_buffers: FitBuffers = self.prepare_fit_buffers(encoded_X, targets, clause_drop_mask)

        pbar = tqdm(range(N), desc="Fitting Batch", leave=False, dynamic_ncols=True)
        for e in pbar:
            # If all the targets are zero, then there is nothing to learn, so skip.
            if np.all(targets[e, :] == 0):
                continue

            self.pack_clauses(dev_buffers.packed_clauses, dev_buffers.n_includes)
            self.eval_clauses(
                dev_buffers.packed_clauses,
                dev_buffers.n_includes,
                dev_buffers.clause_drop_mask,
                dev_buffers.clause_outputs,
                dev_buffers.encoded_X,
                e,
            )
            self.select_patch_and_count_votes(
                dev_buffers.clause_outputs,
                dev_buffers.selected_patch_ids,
                dev_buffers.pos_votes,
                dev_buffers.neg_votes,
            )
            self.calc_update_prob(
                dev_buffers.pos_votes,
                dev_buffers.neg_votes,
                dev_buffers.targets,
                dev_buffers.update_probs,
                e,
            )
            self.update_clauses(
                dev_buffers.n_includes,
                dev_buffers.selected_patch_ids,
                dev_buffers.clause_drop_mask,
                dev_buffers.update_probs,
                dev_buffers.encoded_X,
                dev_buffers.targets,
                e,
            )

    def fit_epoch2(self, X: np.ndarray, targets: np.ndarray, clause_drop_p: float, batch_size: int = -1):
        """
        Memory-efficient training that computes patch matching on-the-fly.
        Does not pre-encode patches - processes samples in batches.

        Args:
            X: Raw input data of shape (N, HEIGHT, WIDTH, DEPTH) as int8
            targets: Target labels of shape (N, n_classes) as int8
            clause_drop_p: Probability of dropping a clause
            batch_size: Number of samples to process per batch. -1 means all at once.
        """
        # Lazy initialization of src2 library
        if not hasattr(self, "lib2_fit_batch"):
            self._init_src2()

        N = X.shape[0]
        if batch_size == -1:
            batch_size = N

        # Generate clause drop mask
        if clause_drop_p > 0.0:
            clause_drop_mask = (self.np_rng.random(self.total_clauses) <= clause_drop_p).astype(np.int8)
        else:
            clause_drop_mask = np.zeros(self.total_clauses, dtype=np.int8)

        # Process in batches
        for i in tqdm(range(0, N, batch_size), desc="Fitting (no-enc)", leave=False, dynamic_ncols=True):
            batch_end = min(i + batch_size, N)
            batch_X = np.ascontiguousarray(X[i:batch_end], dtype=np.int8)
            batch_targets = np.ascontiguousarray(targets[i:batch_end], dtype=np.float32)

            self.lib2_fit_batch(
                self.p_rng,
                self.ta_states.ctypes.data_as(uint32_p),
                self.clause_weights.ctypes.data_as(float_p),
                self.patch_weights.ctypes.data_as(int32_p),
                clause_drop_mask.ctypes.data_as(int8_p),
                batch_X.ctypes.data_as(int8_p),
                batch_targets.ctypes.data_as(float_p),
                c_int(batch_end - i),
            )

    def _init_src2(self):
        """Initialize the src2 library for memory-efficient training."""
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        so_file = self._compile_code(os.path.join(cur_dir, "src2.c"), self.header)
        dll2 = CDLL(so_file)

        self.lib2_fit_sample = dll2.fit_sample
        self.lib2_fit_sample.argtypes = [
            uint32_p,  # rng
            uint32_p,  # global_ta_states
            float_p,  # clause_weights
            int32_p,  # patch_weights
            int8_p,  # clause_drop_mask
            int8_p,  # X (single sample)
            float_p,  # targets
        ]

        self.lib2_infer_sample = dll2.infer_sample
        self.lib2_infer_sample.argtypes = [
            uint32_p,  # rng
            int8_p,  # X (single sample)
            uint32_p,  # global_ta_states
            float_p,  # clause_weights
            float_p,  # class_sums (output)
        ]

        self.lib2_infer_batch = dll2.infer_batch
        self.lib2_infer_batch.argtypes = [
            uint32_p,  # rng
            int8_p,  # X (batch)
            uint32_p,  # global_ta_states
            float_p,  # clause_weights
            float_p,  # class_sums (output)
            c_int,  # N (batch size)
        ]

        self.lib2_fit_batch = dll2.fit_batch
        self.lib2_fit_batch.argtypes = [
            uint32_p,  # rng
            uint32_p,  # global_ta_states
            float_p,  # clause_weights
            int32_p,  # patch_weights
            int8_p,  # clause_drop_mask
            int8_p,  # X (batch)
            float_p,  # targets (batch)
            c_int,  # N (batch size)
        ]

        if self.args.n_threads > 1:
            lib2_set_num_threads = dll2.set_num_threads
            lib2_set_num_threads.argtypes = [c_int]
            lib2_set_num_threads(self.args.n_threads)

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

    def infer2(self, X: np.ndarray, batch_size: int = -1) -> np.ndarray:
        """
        Memory-efficient inference that computes patch matching on-the-fly.
        Does not require pre-encoded data.

        Args:
            X: Raw input data of shape (N, HEIGHT, WIDTH, DEPTH) as int8
            batch_size: Number of samples to process per batch. -1 means all at once.

        Returns:
            class_sums: Array of shape (N, n_classes) with vote sums per class
        """
        # Lazy initialization of src2 library
        if not hasattr(self, "lib2_infer_batch"):
            self._init_src2()

        N = X.shape[0]
        if batch_size == -1:
            batch_size = N

        class_sums = np.zeros((N, self.args.n_classes), dtype=np.float32)

        for i in tqdm(range(0, N, batch_size), desc="Inference (no-enc)", leave=False, dynamic_ncols=True):
            batch_end = min(i + batch_size, N)
            batch_X = np.ascontiguousarray(X[i:batch_end], dtype=np.int8)

            self.lib2_infer_batch(
                self.p_rng,
                batch_X.ctypes.data_as(int8_p),
                self.ta_states.ctypes.data_as(uint32_p),
                self.clause_weights.ctypes.data_as(float_p),
                class_sums[i:batch_end].ctypes.data_as(float_p),
                c_int(batch_end - i),
            )

        return class_sums

    def get_weights(self) -> np.ndarray[tuple[int, int], np.dtype[np.float32]]:
        return self.clause_weights

    def get_ta_states(self) -> np.ndarray[tuple[int, int, int], np.dtype[np.uint32]]:
        n_clause_banks = 1 if self.args.coalesced else self.args.n_classes
        return self.ta_states.reshape((n_clause_banks, self.args.n_clauses, self.n_literals))

    def get_patch_weights(self) -> np.ndarray:
        if not self.args.track_patch_weights:
            warnings.warn("track_patch_weights is False, so no patch_weights were saved.")
            return self.patch_weights
        return self.patch_weights.reshape(self.total_clauses, self.n_patches_y, self.n_patches_x)

    def transform_patchwise(
        self, encoded_X: np.ndarray[tuple[int, int, int], np.dtype[np.uint32]]
    ) -> np.ndarray[tuple[int, int, int, int], np.dtype[np.bool]]:
        N = encoded_X.shape[0]
        co_patchwise = np.zeros((N, self.total_clauses, self.n_patches), dtype=np.int8)

        packed_clauses = np.empty((self.total_clauses, self.n_literal_chunks), dtype=np.uint32)
        n_includes = np.empty((self.total_clauses,), dtype=np.uint32)
        clause_drop_mask = np.zeros((self.total_clauses,), dtype=np.int8)
        clause_outputs = np.empty((self.total_clauses * self.n_patches,), dtype=np.int8)
        self.pack_clauses(packed_clauses, n_includes)

        for i in tqdm(range(N), desc="Patchwise Transform", leave=False, dynamic_ncols=True):
            self.lib_eval_clauses(
                packed_clauses.ctypes.data_as(uint32_p),
                n_includes.ctypes.data_as(uint32_p),
                clause_drop_mask.ctypes.data_as(int8_p),
                encoded_X.ctypes.data_as(uint32_p),
                c_int(i),
                clause_outputs.ctypes.data_as(int8_p),
            )

            co_patchwise[i] = clause_outputs.reshape((self.total_clauses, self.n_patches))

        n_clause_banks = 1 if self.args.coalesced else self.args.n_classes
        return co_patchwise.astype(bool).reshape((N, n_clause_banks, self.args.n_clauses, self.n_patches))

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
