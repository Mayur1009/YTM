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

        if self.args.n_threads > 1:
            self.lib.set_num_threads(self.args.n_threads)

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
        self.p_lit_to_fid = self.lit_to_fid.ctypes.data_as(int32_p)

    def fit_epoch(self, X: np.ndarray, targets: np.ndarray, clause_drop_p: float, batch_size: int):
        N = X.shape[0]

        if clause_drop_p > 0.0:
            clause_drop_mask = (self.np_rng.random(self.total_clauses) <= clause_drop_p).astype(np.int8)
        else:
            clause_drop_mask = np.zeros(self.total_clauses, dtype=np.int8)

        X = X.astype(np.int32)
        targets = targets.astype(np.int8)

        clause_positions = np.empty((self.total_clauses, 4), dtype=np.int32)
        clause_feat_min = np.empty((self.total_clauses, self.n_raw_patch_feats), dtype=np.int32)
        clause_feat_max = np.empty((self.total_clauses, self.n_raw_patch_feats), dtype=np.int32)
        constrained_fids = np.empty((self.total_clauses, self.n_raw_patch_feats), dtype=np.int32)
        n_constrained = np.empty(self.total_clauses, dtype=np.int32)
        num_includes = np.empty(self.total_clauses, dtype=np.uint32)
        is_clause_valid = np.empty(self.total_clauses, dtype=np.int8)
        is_clause_synced = np.zeros(self.total_clauses, dtype=np.int8)  # Make sure this is initialized to 0
        selected_pids = np.empty(self.total_clauses, dtype=np.int32)
        votes = np.empty(self.total_clauses, dtype=np.float32)
        prob = np.empty(self.args.n_classes, dtype=np.float32)

        for e in tqdm(range(N), desc="Fit", leave=False):
            self.lib.fit_sample(
                self.p_rng,
                self.p_ta_states,
                self.p_clause_weights,
                self.p_patch_weights,
                self.p_feat_mins,
                self.p_literal_offsets,
                clause_drop_mask.ctypes.data_as(int8_p),
                X.ctypes.data_as(int32_p),
                targets.ctypes.data_as(int8_p),
                c_int(e),
                clause_positions.ctypes.data_as(int32_p),
                clause_feat_min.ctypes.data_as(int32_p),
                clause_feat_max.ctypes.data_as(int32_p),
                constrained_fids.ctypes.data_as(int32_p),
                n_constrained.ctypes.data_as(int32_p),
                num_includes.ctypes.data_as(uint32_p),
                is_clause_valid.ctypes.data_as(int8_p),
                is_clause_synced.ctypes.data_as(int8_p),
                selected_pids.ctypes.data_as(int32_p),
                votes.ctypes.data_as(float_p),
                prob.ctypes.data_as(float_p),
            )

    def pack_clauses(self):
        clause_positions = np.empty((self.total_clauses, 4), dtype=np.int32)
        clause_feat_min = np.empty((self.total_clauses, self.n_raw_patch_feats), dtype=np.int32)
        clause_feat_max = np.empty((self.total_clauses, self.n_raw_patch_feats), dtype=np.int32)
        constrained_fids = np.empty((self.total_clauses, self.n_raw_patch_feats), dtype=np.int32)
        n_constrained = np.empty(self.total_clauses, dtype=np.int32)
        num_includes = np.empty(self.total_clauses, dtype=np.uint32)
        is_clause_valid = np.empty(self.total_clauses, dtype=np.int8)
        is_clause_synced = np.zeros(self.total_clauses, dtype=np.int8)

        self.lib.pack_clauses(
            self.p_ta_states,
            self.p_literal_offsets,
            clause_positions.ctypes.data_as(int32_p),
            clause_feat_min.ctypes.data_as(int32_p),
            clause_feat_max.ctypes.data_as(int32_p),
            constrained_fids.ctypes.data_as(int32_p),
            n_constrained.ctypes.data_as(int32_p),
            num_includes.ctypes.data_as(uint32_p),
            is_clause_valid.ctypes.data_as(int8_p),
            is_clause_synced.ctypes.data_as(int8_p),
        )
        return {
            "clause_positions": clause_positions,
            "clause_feat_min": clause_feat_min,
            "clause_feat_max": clause_feat_max,
            "constrained_fids": constrained_fids,
            "n_constrained": n_constrained,
            "num_includes": num_includes,
            "is_clause_valid": is_clause_valid,
            "is_clause_synced": is_clause_synced,
        }

    def infer(self, X: np.ndarray, batch_size: int):
        N = X.shape[0]
        X = X.astype(np.int32)
        class_sums = np.zeros((N, self.args.n_classes), dtype=np.float32)
        bufs = self.pack_clauses()

        for e in tqdm(range(N), desc="Infer", leave=False):
            self.lib.infer_sample(
                self.p_clause_weights,
                bufs["clause_positions"].ctypes.data_as(int32_p),
                bufs["clause_feat_min"].ctypes.data_as(int32_p),
                bufs["clause_feat_max"].ctypes.data_as(int32_p),
                bufs["constrained_fids"].ctypes.data_as(int32_p),
                bufs["n_constrained"].ctypes.data_as(int32_p),
                bufs["num_includes"].ctypes.data_as(uint32_p),
                bufs["is_clause_valid"].ctypes.data_as(int8_p),
                self.p_feat_mins,
                X.ctypes.data_as(int32_p),
                c_int(e),
                class_sums.ctypes.data_as(float_p),
            )

        return class_sums

    def transform_patchwise(self, X: np.ndarray, batch_size: int):
        N = X.shape[0]
        X = X.astype(np.int32)
        patch_outputs = np.zeros((N, self.total_clauses, self.n_patches_y, self.n_patches_x), dtype=np.int8)

        # Pack clauses once for all samples
        bufs = self.pack_clauses()

        for e in tqdm(range(N), desc="Transform", leave=False):
            self.lib.eval_sample_patchwise(
                self.p_clause_weights,
                bufs["clause_positions"].ctypes.data_as(int32_p),
                bufs["clause_feat_min"].ctypes.data_as(int32_p),
                bufs["clause_feat_max"].ctypes.data_as(int32_p),
                bufs["constrained_fids"].ctypes.data_as(int32_p),
                bufs["n_constrained"].ctypes.data_as(int32_p),
                bufs["num_includes"].ctypes.data_as(uint32_p),
                bufs["is_clause_valid"].ctypes.data_as(int8_p),
                self.p_feat_mins,
                X.ctypes.data_as(int32_p),
                c_int(e),
                patch_outputs.ctypes.data_as(int8_p),
            )

        return patch_outputs.reshape((N, self.n_clause_banks, self.args.n_clauses, self.n_patches_y, self.n_patches_x))

    def get_weights(self):
        return self.clause_weights

    def get_ta_states(self):
        return self.ta_states.reshape((self.n_clause_banks, self.args.n_clauses, self.n_literals))

    def get_clauses(self):
        """
        Returns human-interpretable clause constraints.

        Returns:
            dict with:
            - "feature_bounds": (total_clauses, n_raw_patch_feats, 2)
               [..., 0] = lower bound (value >= this), inclusive, in original feature space
               [..., 1] = upper bound (value <= this), inclusive, in original feature space
            - "position_bounds": (total_clauses, 4) if position_literals else None
               [min_patch_y, max_patch_y, min_patch_x, max_patch_x], inclusive bounds
            - "is_valid": (total_clauses,) bool - False if clause has contradictory constraints
        """
        bufs = self.pack_clauses()

        feat_mins = self.args.feat_mins
        feat_maxs = self.args.feat_maxs

        # Convert from shifted bounds to original feature space
        # clause_feat_min/max are in shifted space (value - feat_min)
        feature_bounds = np.zeros((self.total_clauses, self.n_raw_patch_feats, 2), dtype=np.int32)
        feature_bounds[:, :, 0] = bufs["clause_feat_min"] + feat_mins  # lower bounds
        feature_bounds[:, :, 1] = bufs["clause_feat_max"] + feat_mins  # upper bounds

        # Position bounds (convert from exclusive max to inclusive)
        position_bounds = None
        if self.args.position_literals:
            position_bounds = bufs["clause_positions"].copy()
            # Convert [min, max) to [min, max] (inclusive)
            position_bounds[:, 1] -= 1  # max_y
            position_bounds[:, 3] -= 1  # max_x

        # Check validity from packed buffer
        is_valid = bufs["is_clause_valid"].astype(bool)

        return {
            "feature_bounds": feature_bounds,
            "position_bounds": position_bounds,
            "is_valid": is_valid,
        }

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
