import os
import shutil
import subprocess
import tempfile
from ctypes import CDLL, POINTER, c_bool, c_float, c_int, c_int8, c_int32, c_uint32

import numpy as np
from tqdm import tqdm

from ..base import BaseDevice

bool_p = POINTER(c_bool)
int8_p = POINTER(c_int8)
int32_p = POINTER(c_int32)
uint32_p = POINTER(c_uint32)
float_p = POINTER(c_float)


class CPUDevice(BaseDevice):
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

        return CDLL(so_file)

    def _build_header(self):
        header = f"""
#define USE_OMP {1 if self.args.n_threads > 1 else 0}
#define TOTAL_CLAUSES {self.total_clauses}
#define THRESH {self.args.T}
#define S {self.args.s}
#define CLASSES {self.args.n_classes}
#define HEIGHT {self.args.dim[0]}
#define WIDTH {self.args.dim[1]}
#define DEPTH {self.args.dim[2]}
#define PATCH_HEIGHT {self.args.patch_dim[0]}
#define PATCH_WIDTH {self.args.patch_dim[1]}
#define STRIDE_Y {self.args.stride[0]}
#define STRIDE_X {self.args.stride[1]}
#define NEGATED_LITERALS {1 if self.args.negated_literals else 0}
#define POSITION_LITERALS {1 if self.args.position_literals else 0}
#define COALESCED {1 if self.args.coalesced else 0}
#define WEIGHTED {1 if self.args.weighted else 0}
#define MAX_WEIGHT {self.args.max_weight}
#define NEGATIVE_CLAUSES {1 if self.args.negative_clauses else 0}
#define ALLOW_POLARITY_CHANGE {1 if self.args.allow_polarity_change else 0}
#define MAX_INCLUDED_LITERALS {self.args.max_includes}
#define INCLUDE_STATE {self.args.include_state}
#define MAX_TA_STATE {self.args.n_states - 1}
#define TYPE1A_FB {0 if self.args.skip_t1a_fb else 1}
#define TYPE1B_FB {0 if self.args.skip_t1b_fb else 1}
#define TYPE2_FB {0 if self.args.skip_t2_fb else 1}
#define N_RAW_PATCH_FEATS {self.n_raw_patch_feats}
#define N_PATCH_FEATS {self.n_patch_feats}
#define N_POSITION_FEATS {self.n_position_feats}
#define N_PATCHES_Y {self.n_patches_y}
#define N_PATCHES_X {self.n_patches_x}
#define N_PATCHES {self.n_patches}
#define N_LITERALS {self.n_literals}
"""
        return header

    def _init_lib(self):
        src_file = os.path.join(os.path.dirname(__file__), "src.c")
        header = self._build_header()
        self.lib = self._compile_code(src_file, header)

        # fit_batch signature
        self.lib_fit_batch = self.lib.fit_batch
        self.lib_fit_batch.argtypes = [
            uint32_p,  # rng
            uint32_p,  # global_ta_states
            float_p,  # clause_weights
            int32_p,  # patch_weights
            int8_p,  # clause_drop_mask
            int32_p,  # X
            int8_p,  # targets
            c_int,  # N
            int32_p,  # feat_mins
            int32_p,  # literal_offsets
        ]
        self.lib_fit_batch.restype = None

        # infer_clauses signature
        self.lib_infer_clauses = self.lib.infer_clauses
        self.lib_infer_clauses.argtypes = [
            uint32_p,  # global_ta_states
            int32_p,  # clause_positions
            int32_p,  # valid_feat_ranges
            bool_p,  # clause_valid
            int32_p,  # feat_mins
            int32_p,  # literal_offsets
            uint32_p,  # num_includes
            int32_p,  # interesting_fids
            int32_p,  # interesting_fid_len
        ]
        self.lib_infer_clauses.restype = None

        # infer_batch signature
        self.lib_infer_batch = self.lib.infer_batch
        self.lib_infer_batch.argtypes = [
            int32_p,  # X
            float_p,  # clause_weights
            float_p,  # class_sums
            c_int,  # N
            int32_p,  # feat_mins
            int32_p,  # clause_positions
            int32_p,  # valid_feat_ranges
            bool_p,  # clause_valid
            uint32_p,  # num_includes
            int32_p,  # interesting_fids
            int32_p,  # interesting_fid_lens
        ]
        self.lib_infer_batch.restype = None

        self.lib_transform_patchwise = self.lib.transform_patchwise
        self.lib_transform_patchwise.argtypes = [
            int32_p,  # X
            int8_p,  # patch_output
            c_int,  # N
            int32_p,  # feat_mins
            int32_p,  # clause_positions
            int32_p,  # valid_feat_ranges
            bool_p,  # clause_valid
            uint32_p,  # num_includes
            int32_p,  # interesting_fids
            int32_p,  # interesting_fid_lens
        ]
        self.lib_transform_patchwise.restype = None

        if self.args.n_threads > 1:
            self.lib_set_num_threads = self.lib.set_num_threads
            self.lib_set_num_threads.argtypes = [c_int]
            self.lib_set_num_threads(self.args.n_threads)

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

        self.patch_weights = np.zeros((self.total_clauses, self.n_patches), dtype=np.int32)

    def dev_init(self):
        self.rng = np.array([self.args.seed + i for i in range(self.args.n_threads)], dtype=np.uint32)

        self._init_clauses()
        self._init_weights()
        self._init_lib()

        # Prepare metadata arrays for C
        self.p_rng = self.rng.ctypes.data_as(uint32_p)
        self.p_ta_states = self.ta_states.ctypes.data_as(uint32_p)
        self.p_clause_weights = self.clause_weights.ctypes.data_as(float_p)
        self.p_patch_weights = self.patch_weights.ctypes.data_as(int32_p)
        self.p_feat_mins = self.args.feat_mins.ctypes.data_as(int32_p)
        self.p_literal_offsets = self.literal_offsets.ctypes.data_as(int32_p)

    def fit_epoch(self, X: np.ndarray, targets: np.ndarray, clause_drop_p: float, batch_size: int):
        N = X.shape[0]
        if batch_size == -1:
            batch_size = N

        if clause_drop_p > 0.0:
            clause_drop_mask = (self.np_rng.random(self.total_clauses) <= clause_drop_p).astype(np.int8)
        else:
            clause_drop_mask = np.zeros(self.total_clauses, dtype=np.int8)

        p_clause_drop_mask = clause_drop_mask.ctypes.data_as(int8_p)

        for i in tqdm(range(0, N, batch_size), desc="Fit batch", leave=False, dynamic_ncols=True):
            batch_end = min(i + batch_size, N)
            batch_X = np.ascontiguousarray(X[i:batch_end], dtype=np.int32)
            batch_targets = np.ascontiguousarray(targets[i:batch_end], dtype=np.int8)

            p_X = batch_X.ctypes.data_as(int32_p)
            p_targets = batch_targets.ctypes.data_as(int8_p)

            self.lib_fit_batch(
                self.p_rng,
                self.p_ta_states,
                self.p_clause_weights,
                self.p_patch_weights,
                p_clause_drop_mask,
                p_X,
                p_targets,
                batch_end - i,
                self.p_feat_mins,
                self.p_literal_offsets,
            )

    def infer_clauses(self):
        clause_positions = np.zeros((self.total_clauses, 4), dtype=np.int32)
        valid_feat_ranges = np.zeros((self.total_clauses, self.n_raw_patch_feats * 2), dtype=np.int32)
        clause_valid = np.zeros(self.total_clauses, dtype=np.bool_)
        num_includes = np.zeros(self.total_clauses, dtype=np.uint32)
        interesting_fids = np.zeros((self.total_clauses, self.n_raw_patch_feats), dtype=np.int32)
        interesting_fid_lens = np.zeros(self.total_clauses, dtype=np.int32)
        self.lib_infer_clauses(
            self.p_ta_states,
            clause_positions.ctypes.data_as(int32_p),
            valid_feat_ranges.ctypes.data_as(int32_p),
            clause_valid.ctypes.data_as(bool_p),
            self.p_feat_mins,
            self.p_literal_offsets,
            num_includes.ctypes.data_as(uint32_p),
            interesting_fids.ctypes.data_as(int32_p),
            interesting_fid_lens.ctypes.data_as(int32_p),
        )
        return {
            "clause_positions": clause_positions,
            "valid_feat_ranges": valid_feat_ranges,
            "clause_valid": clause_valid,
            "num_includes": num_includes,
            "interesting_fids": interesting_fids,
            "interesting_fid_lens": interesting_fid_lens,
        }

    def infer(self, X: np.ndarray, batch_size: int):
        N = X.shape[0]
        if batch_size == -1:
            batch_size = N

        class_sums = np.zeros((N, self.args.n_classes), dtype=np.float32)

        bufs = self.infer_clauses()
        p_clause_positions = bufs["clause_positions"].ctypes.data_as(int32_p)
        p_valid_feat_ranges = bufs["valid_feat_ranges"].ctypes.data_as(int32_p)
        p_clause_valid = bufs["clause_valid"].ctypes.data_as(bool_p)
        p_num_includes = bufs["num_includes"].ctypes.data_as(uint32_p)
        p_interesting_fids = bufs["interesting_fids"].ctypes.data_as(int32_p)
        p_interesting_fid_lens = bufs["interesting_fid_lens"].ctypes.data_as(int32_p)

        for i in tqdm(range(0, N, batch_size), desc="Infer batch", leave=False, dynamic_ncols=True):
            batch_end = min(i + batch_size, N)
            batch_X = np.ascontiguousarray(X[i:batch_end], dtype=np.int32)
            batch_class_sums = np.ascontiguousarray(class_sums[i:batch_end], dtype=np.float32)

            p_X = batch_X.ctypes.data_as(int32_p)
            p_class_sums = batch_class_sums.ctypes.data_as(float_p)

            self.lib_infer_batch(
                p_X,
                self.p_clause_weights,
                p_class_sums,
                batch_end - i,
                self.p_feat_mins,
                p_clause_positions,
                p_valid_feat_ranges,
                p_clause_valid,
                p_num_includes,
                p_interesting_fids,
                p_interesting_fid_lens,
            )

            class_sums[i:batch_end] = batch_class_sums

        return class_sums

    def transform_patchwise(self, X: np.ndarray, batch_size: int):
        N = X.shape[0]
        if batch_size == -1:
            batch_size = N

        patch_outputs = np.zeros((N, self.total_clauses, self.n_patches), dtype=np.int32)

        bufs = self.infer_clauses()
        p_clause_positions = bufs["clause_positions"].ctypes.data_as(int32_p)
        p_valid_feat_ranges = bufs["valid_feat_ranges"].ctypes.data_as(int32_p)
        p_clause_valid = bufs["clause_valid"].ctypes.data_as(bool_p)
        p_num_includes = bufs["num_includes"].ctypes.data_as(uint32_p)
        p_interesting_fids = bufs["interesting_fids"].ctypes.data_as(int32_p)
        p_interesting_fid_lens = bufs["interesting_fid_lens"].ctypes.data_as(int32_p)

        for i in tqdm(range(0, N, batch_size), desc="Infer batch", leave=False, dynamic_ncols=True):
            batch_end = min(i + batch_size, N)
            batch_X = np.ascontiguousarray(X[i:batch_end], dtype=np.int32)
            batch_po = np.ascontiguousarray(patch_outputs[i:batch_end], dtype=np.int8)

            p_X = batch_X.ctypes.data_as(int32_p)
            p_po = batch_po.ctypes.data_as(int32_p)

            self.lib_transform_patchwise(
                p_X,
                p_po,
                batch_end - i,
                self.p_feat_mins,
                p_clause_positions,
                p_valid_feat_ranges,
                p_clause_valid,
                p_num_includes,
                p_interesting_fids,
                p_interesting_fid_lens,
            )

            patch_outputs[i:batch_end] = batch_po

        return patch_outputs.reshape((N, self.n_clause_banks, self.args.n_clauses, self.n_patches))

    def get_weights(self):
        return self.clause_weights

    def get_ta_states(self):
        return self.ta_states.reshape((self.n_clause_banks, self.args.n_clauses, self.n_literals))

    def get_clauses(self):
        # Use infer clauses to get clause ranges. Then create a array of (TOTAL_CLAUSES, N_RAW_PATCH_FEATS) (or *2 if negated_literals) and depending on the range set the thermometer bin value. The positive literals will mean this clause matches >= that value, and the neagated literal value will mean, the clauses matches < that value.
        bufs = self.infer_clauses()

        if self.args.position_literals:
            translated_pos_positions = np.zeros((self.total_clauses, 2))

            if self.args.negated_literals:
                translated_neg_positions = np.zeros((self.total_clauses, 2))

        # TODO:


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

        # Update pointers after loading
        self.p_ta_states = self.ta_states.ctypes.data_as(uint32_p)
        self.p_clause_weights = self.clause_weights.ctypes.data_as(float_p)
        self.p_patch_weights = self.patch_weights.ctypes.data_as(int32_p)
