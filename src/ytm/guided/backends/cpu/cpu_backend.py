import os
import pathlib
import platform
import shutil
import subprocess
import tempfile
import warnings
from ctypes import CDLL, POINTER, c_float, c_int, c_int8, c_int32, c_uint8, c_uint32, c_uint64

import numpy as np

from ..base import BaseDevice, tqdm_bar

int8_p = POINTER(c_int8)
int32_p = POINTER(c_int32)
uint32_p = POINTER(c_uint32)
uint8_p = POINTER(c_uint8)
float_p = POINTER(c_float)

omp_flags = {
    "gcc": ["-fopenmp"],
    "clang": ["-fopenmp"],
}


def _run_compiler(cmd: list[str]) -> None:
    subprocess.run(cmd, capture_output=True, check=True)


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
        out_file = f.name.replace(".c", ".out")
        try:
            _run_compiler([compiler] + omp_flags.get(compiler, []) + [f.name, "-o", out_file])
        except subprocess.CalledProcessError as e:
            warnings.warn(
                f"OpenMP support check failed for compiler '{compiler}'. Compiler Output:\n{e.stdout.decode()}\nError: {e.stderr.decode()}\n"
                "Proceeding without OpenMP support."
            )
            return False

        os.unlink(out_file)
        return True


class CPUDevice(BaseDevice):
    def _select_compiler(self):
        if shutil.which("clang"):
            self.compiler = "clang"
        elif shutil.which("gcc"):
            self.compiler = "gcc"
        else:
            raise RuntimeError("No suitable C compiler found (clang or gcc)")

        if self.args.compile_flags is None:
            self.compiler_flags = ["-shared", "-fPIC", "-lm", "-O3", "-march=native", "-mtune=native"]
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
            _run_compiler([self.compiler] + self.compiler_flags + self.omp_flags + [c_file, "-o", so_file])
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to compile. Compiler Output:\n{e.stdout.decode()}\nError: {e.stderr.decode()}")
        return CDLL(so_file)

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
{self._read_file(dir_path / "losses.c")}
{self._read_file(dir_path / "update.c")}
{self._read_file(dir_path / "inference.c")}
        """
        self.lib = self._compile_code(code)

    def _init_pointers(self):
        self.p_clause_position_bounds = self.packed_clauses.clause_position_bounds.ctypes.data_as(int32_p)
        self.p_clause_feat_bounds = self.packed_clauses.clause_feat_bounds.ctypes.data_as(int32_p)
        self.p_bounded_feat_ids = self.packed_clauses.bounded_feat_ids.ctypes.data_as(int32_p)
        self.p_n_bounded_feats = self.packed_clauses.n_bounded_feats.ctypes.data_as(int32_p)
        self.p_clause_density = self.packed_clauses.clause_density.ctypes.data_as(int32_p)
        self.p_is_clause_synced = self.packed_clauses.is_clause_synced.ctypes.data_as(int8_p)

        self.p_ta_states = self.ta_states.ctypes.data_as(uint32_p)
        self.p_clause_weights = self.clause_weights.ctypes.data_as(float_p)
        self.p_bias = self.bias.ctypes.data_as(float_p)
        self.p_patch_weights = self.patch_weights.ctypes.data_as(int32_p)
        self.p_feat_mins = np.asarray(self.args.feat_mins).ctypes.data_as(int32_p)
        self.p_feat_maxs = np.asarray(self.args.feat_maxs).ctypes.data_as(int32_p)
        self.p_literal_offsets = self.literal_offsets.ctypes.data_as(int32_p)
        self.p_feedback_type = self.feedback_type.ctypes.data_as(uint8_p)

    def dev_init(self):
        self.xp = np
        self._select_compiler()
        self._openmp_flags()
        self._init_clauses()
        self._init_weights()
        self._init_bias()
        self._init_packed_clauses()
        self._init_frozen_clauses()
        self._init_lib()
        self.feedback_type = np.zeros((self.total_clauses, self.args.n_classes), dtype=np.uint8)
        self._init_pointers()
        self.set_threads(self.args.n_threads)
        self._init_loss_fn()

    def _to_host(self, arr):
        return arr.copy()

    def _init_loss_fn(self):
        super()._init_loss_fn()
        self.p_loss_class_weights = self._loss_class_weights.ctypes.data_as(float_p)

    def set_threads(self, n: int):
        self.lib.set_num_threads(c_int(n))

    def fit_epoch(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        clause_drop_p: float,
        batch_size: int,
        lr: float | None = None,
        lambda_: float | None = None,
    ):
        N = X.shape[0]

        if clause_drop_p > 0.0:
            clause_drop_mask = (self._rng.random(self.total_clauses) <= clause_drop_p).astype(np.int8)
        else:
            clause_drop_mask = np.zeros(self.total_clauses, dtype=np.int8)

        clause_drop_mask = np.logical_or(clause_drop_mask, self.frozen_clauses.flatten()).astype(np.int8)

        X = X.astype(np.int32)
        Y = Y.astype(np.float32)

        _lr = lr if lr is not None else self.args.lr
        selected_pids = np.empty(self.total_clauses, dtype=np.int32)
        votes = np.empty(self.args.n_classes, dtype=np.float32)
        grad = np.empty(self.args.n_classes, dtype=np.float32)
        y_hat = np.empty(self.args.n_classes, dtype=np.float32)
        loss_val = np.empty(1, dtype=np.float32)
        running_loss = 0.0

        p_X = X.ctypes.data_as(int32_p)
        p_clause_drop_mask = clause_drop_mask.ctypes.data_as(int8_p)
        p_selected_pids = selected_pids.ctypes.data_as(int32_p)
        p_votes = votes.ctypes.data_as(float_p)
        p_grad = grad.ctypes.data_as(float_p)
        p_y_hat = y_hat.ctypes.data_as(float_p)
        p_loss = loss_val.ctypes.data_as(float_p)

        pbar = tqdm_bar(range(N), desc="Fit")
        for e in pbar:
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
            _rng_key = c_uint64(int(self._rng.integers(1, 1 << 63, dtype=np.uint64)))

            self.lib.evaluate(
                _rng_key,
                p_X,
                c_int(e),
                p_clause_drop_mask,
                self.p_clause_position_bounds,
                self.p_clause_feat_bounds,
                self.p_bounded_feat_ids,
                self.p_n_bounded_feats,
                self.p_clause_density,
                p_selected_pids,
                self.p_patch_weights,
            )
            self.lib.count_votes(
                p_selected_pids,
                self.p_clause_weights,
                self.p_bias,
                p_votes,
            )
            self.lib.compute_act_loss_grad(
                p_votes,
                Y[e].ctypes.data_as(float_p),
                self.p_loss_class_weights,
                p_y_hat,
                p_grad,
                p_loss,
                c_int(-1),
                c_float(0.0),
            )
            self.lib.decide_feedback(
                _rng_key,
                p_votes,
                Y[e].ctypes.data_as(float_p),
                self.p_loss_class_weights,
                p_loss,
                self.p_clause_weights,
                self.p_clause_density,
                p_selected_pids,
                p_clause_drop_mask,
                c_float(self.args.lambda_),
                self.p_feedback_type,
                self.p_is_clause_synced,
            )
            self.lib.update_weights(
                p_selected_pids,
                p_clause_drop_mask,
                p_grad,
                c_float(_lr),
                self.p_clause_weights,
            )
            self.lib.update_bias(
                p_grad,
                c_float(_lr),
                self.p_bias,
            )
            self.lib.update_clauses(
                _rng_key,
                p_selected_pids,
                p_X,
                c_int(e),
                self.p_feat_mins,
                self.p_literal_offsets,
                self.p_feedback_type,
                self.p_ta_states,
            )

            running_loss += float(loss_val[0])
            pbar.set_postfix(loss=f"{running_loss / (e + 1):.4f}")

        return running_loss / N

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
        votes = np.zeros((N, self.args.n_classes), dtype=np.float32)
        self.pack_clauses()

        for e in tqdm_bar(range(N), desc="Infer"):
            self.lib.infer_sample(
                self.p_clause_weights,
                self.p_bias,
                self.p_clause_position_bounds,
                self.p_clause_feat_bounds,
                self.p_bounded_feat_ids,
                self.p_n_bounded_feats,
                self.p_clause_density,
                p_X,
                c_int(e),
                votes.ctypes.data_as(float_p),
            )

        y_hat = np.empty((N, self.args.n_classes), dtype=np.float32)
        self.lib.apply_act_batch(votes.ctypes.data_as(float_p), c_int(N), y_hat.ctypes.data_as(float_p))
        return y_hat

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

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self._init_pointers()
