import platform
import pathlib
import os
import shutil
import subprocess
import tempfile
import warnings
from ctypes import CDLL, POINTER, c_float, c_int, c_int8, c_int32, c_uint32, c_uint64

import numpy as np

from ..base import BaseDevice, PackedClauses, tqdm_bar

int8_p = POINTER(c_int8)
int32_p = POINTER(c_int32)
uint32_p = POINTER(c_uint32)
float_p = POINTER(c_float)

omp_flags = {
    "gcc": ["-fopenmp"],
    "clang": ["-fopenmp"],
}


def _check_openmp_support(compiler: str) -> bool:
    with tempfile.NamedTemporaryFile(suffix=".c", mode="w") as f:
        f.write("""
#include <omp.h>
int main() {
    int n_threads = omp_get_max_threads();
    return 0;
}
        """)
        f.flush()
        ext = ".out"
        out_file = f.name.replace(".c", ext)
        result = subprocess.run(
            [compiler] + omp_flags.get(compiler, []) + [f.name, "-o", out_file],
            capture_output=True,
        )

        if result.returncode == 0:
            os.unlink(out_file)
            supported = True
        else:
            warnings.warn(
                f"OpenMP support check failed for compiler '{compiler}'. Compiler Output:\n{result.stdout.decode()}\nError: {result.stderr.decode()}\n"
                "Proceeding without OpenMP support."
            )
            supported = False

    return supported


class CPUDevice(BaseDevice):
    def _select_compiler(self):
        if shutil.which("clang"):
            self.compiler = "clang"
        elif shutil.which("gcc"):
            self.compiler = "gcc"
        else:
            raise RuntimeError("No suitable C compiler found (clang or gcc)")

        if self.args.compile_flags is None:
            self.compiler_flags = ["-shared", "-fPIC", "-lm", "-O3", "-ffast-math", "-march=native", "-mtune=native"]
        else:
            self.compiler_flags = self.args.compile_flags

    def _openmp_flags(self):
        if self.args.n_threads > 1 and _check_openmp_support(self.compiler):
            self.omp_flags = omp_flags[self.compiler]
        else:
            self.args.n_threads = 1
            self.omp_flags = []

    def _compile_code(self, code: str):
        with tempfile.NamedTemporaryFile(suffix=".c", mode="w", delete=False) as f:
            f.write(code)
            c_file = f.name

        ext = ".dll" if platform.system() == "Windows" else ".so"
        so_file = c_file.replace(".c", ext)

        try:
            subprocess.run(
                [self.compiler] + self.compiler_flags + self.omp_flags + [c_file, "-o", so_file],
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to compile. Compiler Output:\n{e.stdout.decode()}\nError: {e.stderr.decode()}")
        return CDLL(so_file)

    def _build_header(self):
        header = f"""
#define TOTAL_CLAUSES {self.total_clauses}
#define T_MIN {float(self.args.T_min)}f
#define T_MAX {float(self.args.T_max)}f
#define S {float(self.args.s)}f
#define CLASSES {self.args.n_classes}
#define Q {float(self.args.q)}f
#define HEIGHT {self.args.dim[0]}
#define WIDTH {self.args.dim[1]}
#define DEPTH {self.args.dim[2]}
#define PATCH_HEIGHT {self.args.patch_dim[0]}
#define PATCH_WIDTH {self.args.patch_dim[1]}
#define STRIDE_Y {self.args.stride[0]}
#define STRIDE_X {self.args.stride[1]}
#define MAX_WEIGHT {float(self.args.max_weight)}f
#define MAX_INCLUDED_LITERALS {self.args.max_includes}
#define INCLUDE_STATE {self.args.include_state}
#define MAX_TA_STATE {self.args.n_states - 1}
#define N_RAW_PATCH_FEATS {self.n_raw_patch_feats}
#define N_PATCH_FEATS {self.n_patch_feats}
#define N_POSITION_FEATS {self.n_position_feats}
#define N_PATCHES_Y {self.n_patches_y}
#define N_PATCHES_X {self.n_patches_x}
#define N_PATCHES {self.n_patches}
#define N_LITERALS {self.n_literals}
#define NEGATED_LITERALS {1 if self.args.negated_literals else 0}
#define POSITION_LITERALS {1 if self.args.position_literals else 0}
#define COALESCED {1 if self.args.coalesced else 0}
#define WEIGHTED {1 if self.args.weighted else 0}
#define NEGATIVE_CLAUSES {1 if self.args.negative_clauses else 0}
#define ALLOW_POLARITY_CHANGE {1 if self.args.allow_polarity_change else 0}
#define TYPE1A_FB {0 if self.args.skip_t1a_fb else 1}
#define TYPE1B_FB {0 if self.args.skip_t1b_fb else 1}
#define TYPE2_FB {0 if self.args.skip_t2_fb else 1}
#define TRACK_PATCH_WEIGHTS {1 if self.args.track_patch_weights else 0}
#define BOOST_TP_FB {1 if self.args.boost_tp_fb else 0}
"""
        return header

    def _read_file(self, path: pathlib.Path) -> str:
        with open(path, "r") as f:
            return f.read()

    def _init_lib(self):
        dir_path = pathlib.Path(__file__).parent
        code = f"""
{self._build_header()}
{self._read_file(dir_path / "common.c")}
{self._read_file(dir_path / "pack_clauses.c")}
{self._read_file(dir_path / "evaluate.c")}
{self._read_file(dir_path / "update.c")}
{self._read_file(dir_path / "inference.c")}
        """
        self.lib = self._compile_code(code)

    def _init_clauses(self):
        self.ta_states = np.full(
            (self.total_clauses, self.n_literals),
            self.args.include_state - 1,
            dtype=np.uint32,
        )

    def _init_weights(self):
        self.clause_weights = np.ones((self.args.n_classes, self.args.n_clauses), dtype=np.float32)
        if self.args.negative_clauses:
            n_neg_polarity = self.args.n_clauses // 2
            if self.args.coalesced:
                for i in range(self.args.n_classes):
                    wt = np.ones((self.args.n_clauses,), dtype=np.float32) * 1.0
                    wt[n_neg_polarity:] *= -1.0
                    self.clause_weights[i, :] = self.np_rng.permutation(wt)
            else:
                self.clause_weights[:, n_neg_polarity:] *= -1.0

        if self.args.track_patch_weights:
            self.patch_weights = np.zeros((self.total_clauses, self.n_patches), dtype=np.int32)
        else:
            self.patch_weights = np.zeros((1, 1), dtype=np.int32)

    def _init_packed_clauses(self):
        self.packed_clauses = PackedClauses(
            clause_position_bounds=np.empty((self.total_clauses, 4), dtype=np.int32),
            clause_feat_bounds=np.empty((self.total_clauses, self.n_raw_patch_feats, 2), dtype=np.int32),
            bounded_feat_ids=np.empty((self.total_clauses, self.n_raw_patch_feats), dtype=np.int32),
            n_bounded_feats=np.empty(self.total_clauses, dtype=np.int32),
            clause_density=np.empty(self.total_clauses, dtype=np.int32),
            is_clause_synced=np.zeros(self.total_clauses, dtype=np.int8),
        )

    def _init_frozen_clauses(self):
        self.frozen_clauses = np.zeros((self.n_clause_banks, self.args.n_clauses), dtype=np.int8)

    def _init_pointers(self):
        self.p_clause_position_bounds = self.packed_clauses.clause_position_bounds.ctypes.data_as(int32_p)
        self.p_clause_feat_bounds = self.packed_clauses.clause_feat_bounds.ctypes.data_as(int32_p)
        self.p_bounded_feat_ids = self.packed_clauses.bounded_feat_ids.ctypes.data_as(int32_p)
        self.p_n_bounded_feats = self.packed_clauses.n_bounded_feats.ctypes.data_as(int32_p)
        self.p_clause_density = self.packed_clauses.clause_density.ctypes.data_as(int32_p)
        self.p_is_clause_synced = self.packed_clauses.is_clause_synced.ctypes.data_as(int8_p)

        self.p_ta_states = self.ta_states.ctypes.data_as(uint32_p)
        self.p_clause_weights = self.clause_weights.ctypes.data_as(float_p)
        self.p_patch_weights = self.patch_weights.ctypes.data_as(int32_p)
        self.p_feat_mins = np.asarray(self.args.feat_mins).ctypes.data_as(int32_p)
        self.p_feat_maxs = np.asarray(self.args.feat_maxs).ctypes.data_as(int32_p)
        self.p_literal_offsets = self.literal_offsets.ctypes.data_as(int32_p)

    def dev_init(self):
        self._select_compiler()
        self._openmp_flags()
        self._init_clauses()
        self._init_weights()
        self._init_packed_clauses()
        self._init_frozen_clauses()
        self._init_lib()
        self._init_pointers()

        self.set_threads(self.args.n_threads)

    def set_threads(self, n: int):
        self.lib.set_num_threads(c_int(n))

    def freeze_clauses(self, class_id: int, clause_ids: list[int] | np.ndarray):
        clause_ids = np.asarray(clause_ids, dtype=np.int32)
        self.frozen_clauses[class_id, clause_ids] = 1

    def unfreeze_clauses(self):
        self.frozen_clauses.fill(0)

    def fit_epoch(self, X: np.ndarray, encoded_Y: np.ndarray, clause_drop_p: float, batch_size: int, label_probs: np.ndarray, rng_state: int):
        N = X.shape[0]

        if clause_drop_p > 0.0:
            clause_drop_mask = (self.np_rng.random(self.total_clauses) <= clause_drop_p).astype(np.int8)
        else:
            clause_drop_mask = np.zeros(self.total_clauses, dtype=np.int8)

        clause_drop_mask = np.logical_or(clause_drop_mask, self.frozen_clauses.flatten()).astype(np.int8)

        X = X.astype(np.int32)
        encoded_Y = encoded_Y.astype(np.float32)
        label_probs = label_probs.astype(np.float32)

        selected_pids = np.empty(self.total_clauses, dtype=np.int32)
        votes = np.empty(self.args.n_classes, dtype=np.float32)
        prob = np.empty(self.args.n_classes, dtype=np.float32)

        p_X = X.ctypes.data_as(int32_p)
        p_encoded_Y = encoded_Y.ctypes.data_as(float_p)
        p_label_probs = label_probs.ctypes.data_as(float_p)
        p_clause_drop_mask = clause_drop_mask.ctypes.data_as(int8_p)
        p_selected_pids = selected_pids.ctypes.data_as(int32_p)
        p_votes = votes.ctypes.data_as(float_p)
        p_prob = prob.ctypes.data_as(float_p)

        for e in tqdm_bar(range(N), desc="Fit"):
            self.lib.pack_clauses(
                self.p_ta_states,
                self.p_feat_mins,
                self.p_feat_maxs,
                self.p_literal_offsets,
                self.p_clause_position_bounds,
                self.p_clause_feat_bounds,
                self.p_bounded_feat_ids,
                self.p_n_bounded_feats,
                self.p_clause_density,
                self.p_is_clause_synced,
            )
            self.lib.evaluate(
                p_X,
                c_int(e),
                p_clause_drop_mask,
                self.p_clause_position_bounds,
                self.p_clause_feat_bounds,
                self.p_bounded_feat_ids,
                self.p_n_bounded_feats,
                self.p_clause_density,
                c_uint64(rng_state),
                p_selected_pids,
                self.p_patch_weights,
            )
            self.lib.count_votes(
                p_selected_pids,
                self.p_clause_weights,
                p_votes,
            )
            self.lib.calc_update_prob(
                p_votes,
                p_encoded_Y,
                c_int(e),
                p_prob,
            )
            self.lib.update_clauses(
                c_uint64(rng_state),
                p_selected_pids,
                self.p_clause_density,
                p_clause_drop_mask,
                p_X,
                p_encoded_Y,
                c_int(e),
                p_prob,
                p_label_probs,
                self.p_ta_states,
                self.p_clause_weights,
                self.p_feat_mins,
                self.p_literal_offsets,
                self.p_is_clause_synced,
            )

    def pack_clauses(self, force_repack: bool = False):
        if force_repack:
            self.packed_clauses.is_clause_synced.fill(0)

        self.lib.pack_clauses(
            self.p_ta_states,
            self.p_feat_mins,
            self.p_feat_maxs,
            self.p_literal_offsets,
            self.p_clause_position_bounds,
            self.p_clause_feat_bounds,
            self.p_bounded_feat_ids,
            self.p_n_bounded_feats,
            self.p_clause_density,
            self.p_is_clause_synced,
        )

    def infer(self, X: np.ndarray, batch_size: int):
        N = X.shape[0]
        X = X.astype(np.int32)
        p_X = X.ctypes.data_as(int32_p)
        class_sums = np.zeros((N, self.args.n_classes), dtype=np.float32)
        self.pack_clauses()

        for e in tqdm_bar(range(N), desc="Infer"):
            self.lib.infer_sample(
                self.p_clause_weights,
                self.p_clause_position_bounds,
                self.p_clause_feat_bounds,
                self.p_bounded_feat_ids,
                self.p_n_bounded_feats,
                self.p_clause_density,
                p_X,
                c_int(e),
                class_sums.ctypes.data_as(float_p),
            )

        return class_sums

    def transform(self, X: np.ndarray, batch_size: int):
        N = X.shape[0]
        X = X.astype(np.int32)
        clause_outputs = np.zeros((N, self.total_clauses), dtype=np.int8)
        clause_drop_mask = np.zeros(self.total_clauses, dtype=np.int8)
        selected_pids = np.empty(self.total_clauses, dtype=np.int32)
        self.pack_clauses()

        p_X = X.ctypes.data_as(int32_p)
        p_clause_drop_mask = clause_drop_mask.ctypes.data_as(int8_p)
        p_selected_pids = selected_pids.ctypes.data_as(int32_p)

        for e in tqdm_bar(range(N), desc="Transform"):
            self.lib.evaluate(
                p_X,
                c_int(e),
                p_clause_drop_mask,
                self.p_clause_position_bounds,
                self.p_clause_feat_bounds,
                self.p_bounded_feat_ids,
                self.p_n_bounded_feats,
                self.p_clause_density,
                c_uint64(self.args.seed),
                p_selected_pids,
                self.p_patch_weights,
            )
            clause_outputs[e] = (selected_pids >= 0).astype(np.int8)

        return clause_outputs

    def transform_patchwise(self, X: np.ndarray, batch_size: int):
        N = X.shape[0]
        X = X.astype(np.int32)
        p_X = X.ctypes.data_as(int32_p)
        patch_outputs = np.zeros((N, self.total_clauses, self.n_patches_y, self.n_patches_x), dtype=np.int8)
        self.pack_clauses()

        for e in tqdm_bar(range(N), desc="Transform"):
            self.lib.eval_sample_patchwise(
                self.p_clause_position_bounds,
                self.p_clause_feat_bounds,
                self.p_bounded_feat_ids,
                self.p_n_bounded_feats,
                self.p_clause_density,
                p_X,
                c_int(e),
                patch_outputs.ctypes.data_as(int8_p),
            )

        return patch_outputs.reshape((N, self.n_clause_banks, self.args.n_clauses, self.n_patches_y, self.n_patches_x))

    def get_weights(self):
        return self.clause_weights

    def get_ta_states(self):
        return self.ta_states.reshape((self.n_clause_banks, self.args.n_clauses, self.n_literals))

    def get_patch_weights(self):
        if not self.args.track_patch_weights:
            warnings.warn("track_patch_weights is False, so no patch_weights were saved.")
            return self.patch_weights
        return self.patch_weights.reshape(self.n_clause_banks, self.args.n_clauses, self.n_patches_y, self.n_patches_x)

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
        self.packed_clauses.is_clause_synced.fill(0)
        self._init_pointers()
