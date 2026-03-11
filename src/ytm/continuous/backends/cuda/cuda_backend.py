import numpy as np
import os
from tqdm import tqdm
import pycuda.gpuarray as ga
from pycuda.compiler import SourceModule
from pycuda.curandom import XORWOWRandomNumberGenerator
from pycuda.driver import Context, device_attribute  # pyright: ignore # ty: ignore
from ..base import BaseDevice


def read_file(path):
    with open(path, "r") as f:
        return f.read()


class CUDADevice(BaseDevice):
    def _build_header(self):
        header = f"""
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

    def _init_kernels(self):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        header = self._build_header()

        mod = SourceModule(
            header + "\n" + read_file(os.path.join(cur_dir, "kernel.cu")),
            options=["-O3", "--use_fast_math"],
            no_extern_c=True,
        )

        # Get kernel functions
        self.k_infer_clauses = mod.get_function("infer_clauses")
        self.k_infer_sample = mod.get_function("infer_sample")
        self.k_eval_clauses = mod.get_function("eval_clauses")
        self.k_count_votes = mod.get_function("count_votes")
        self.k_calc_update_prob = mod.get_function("calc_update_prob")
        self.k_update_clauses = mod.get_function("update_clauses")
        self.k_transform_patchwise = mod.get_function("transform_patchwise")

    def _init_clauses(self):
        self.ta_states = ga.to_gpu(
            np.full(
                (self.total_clauses, self.n_literals),
                self.args.include_state - 1,
                dtype=np.uint32,
            )
        )

    def _init_weights(self):
        n_neg_polarity = self.args.n_clauses // 2
        clause_weights = np.zeros((self.args.n_classes, self.args.n_clauses), dtype=np.float32)
        for i in range(self.args.n_classes):
            wt = np.ones((self.args.n_clauses,), dtype=np.float32) * 1.0
            wt[n_neg_polarity:] *= -1.0
            clause_weights[i, :] = self.np_rng.permutation(wt) if self.args.coalesced else wt

        self.clause_weights = ga.to_gpu(clause_weights)
        self.patch_weights = ga.to_gpu(np.zeros((self.total_clauses, self.n_patches), dtype=np.int32))

    def _init_clause_arrays(self):
        """Allocate GPU arrays for pre-computed clause information."""
        self.clause_positions = ga.zeros((self.total_clauses, 4), dtype=np.int32)
        self.valid_feat_ranges = ga.zeros((self.total_clauses, self.n_raw_patch_feats * 2), dtype=np.int32)
        self.clause_valid = ga.zeros(self.total_clauses, dtype=np.bool_)
        self.num_includes = ga.zeros(self.total_clauses, dtype=np.uint32)
        self.interesting_fids = ga.zeros((self.total_clauses, self.n_raw_patch_feats), dtype=np.int32)
        self.interesting_fid_lens = ga.zeros(self.total_clauses, dtype=np.int32)

        # Upload constant arrays to GPU
        self.d_feat_mins = ga.to_gpu(self.args.feat_mins.astype(np.int32))
        self.d_literal_offsets = ga.to_gpu(self.literal_offsets.astype(np.int32))

    def _init_training_arrays(self):
        """Allocate GPU arrays for training."""
        self.selected_patch_ids = ga.zeros(self.total_clauses, dtype=np.int32)
        self.clause_num_includes = ga.zeros(self.total_clauses, dtype=np.uint32)
        self.votes = ga.zeros(self.args.n_classes, dtype=np.float32)
        self.prob = ga.zeros(self.args.n_classes, dtype=np.float32)

    def _kernel_config(self, n) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        # Ensure hardware compliance
        bs = min(self.args.block_size, self.cuda_props["max_threads_per_block"])

        if self.args.grid_size is None:
            # Calculate grid size
            gs = (n + bs - 1) // bs

            # Limit grid size to reasonable bounds
            max_blocks = min(65535, self.cuda_props["multiprocessor_count"] * 4)
            gs = min(gs, max_blocks)
        else:
            gs = self.args.grid_size

        return (gs, 1, 1), (bs, 1, 1)

    def dev_init(self):
        import pycuda.autoprimaryctx  # noqa: F401

        self.ctx: Context = pycuda.autoprimaryctx.context

        self.rng = XORWOWRandomNumberGenerator(
            lambda count: ga.to_gpu(np.array([(self.args.seed + i) for i in range(1, count + 1)], dtype=np.int32))
        )
        device = self.ctx.get_device()
        attrs = device.get_attributes()
        self.cuda_props = {
            "max_threads_per_block": attrs[device_attribute.MAX_THREADS_PER_BLOCK],
            "max_block_dim_x": attrs[device_attribute.MAX_BLOCK_DIM_X],
            "max_grid_dim_x": attrs[device_attribute.MAX_GRID_DIM_X],
            "warp_size": attrs[device_attribute.WARP_SIZE],
            "multiprocessor_count": attrs[device_attribute.MULTIPROCESSOR_COUNT],
            "max_shared_memory_per_block": attrs[device_attribute.MAX_SHARED_MEMORY_PER_BLOCK],
        }
        self._init_clauses()
        self._init_weights()
        self._init_clause_arrays()
        self._init_training_arrays()
        self._init_kernels()

    def _run_infer_clauses(self):
        """Pre-compute clause information (call once before inference batch)."""
        grid, block = self._kernel_config(self.total_clauses)
        self.k_infer_clauses(
            self.ta_states,
            self.clause_positions,
            self.valid_feat_ranges,
            self.clause_valid,
            self.d_feat_mins,
            self.d_literal_offsets,
            self.num_includes,
            self.interesting_fids,
            self.interesting_fid_lens,
            block=block,
            grid=grid,
        )

    def fit_epoch(self, X: np.ndarray, targets: np.ndarray, clause_drop_p: float, batch_size: int):
        N = X.shape[0]
        if batch_size == -1:
            batch_size = N

        if clause_drop_p > 0.0:
            clause_drop_mask = (self.np_rng.random(self.total_clauses) <= clause_drop_p).astype(np.int8)
        else:
            clause_drop_mask = np.zeros(self.total_clauses, dtype=np.int8)

        d_clause_drop_mask = ga.to_gpu(clause_drop_mask)

        # Get RNG state for CUDA - need curandState array
        # XORWOWRandomNumberGenerator uses its own state, we need to get the underlying state
        grid, block = self._kernel_config(self.total_clauses)
        grid_classes, block_classes = self._kernel_config(self.args.n_classes)

        # Allocate curandState if not already done
        if not hasattr(self, "d_rng_states"):
            # Use the RNG's internal state
            self.d_rng_states = self.rng.state

        for i in tqdm(range(0, N, batch_size), desc="Fit batch", leave=False, dynamic_ncols=True):
            batch_end = min(i + batch_size, N)
            batch_X = np.ascontiguousarray(X[i:batch_end], dtype=np.int32)
            batch_targets = np.ascontiguousarray(targets[i:batch_end], dtype=np.int8)

            d_X = ga.to_gpu(batch_X)
            d_targets = ga.to_gpu(batch_targets)

            # Process each sample in the batch sequentially (to preserve learning dynamics)
            for j in range(batch_end - i):
                sample_X = int(d_X.gpudata) + j * self.args.dim[0] * self.args.dim[1] * self.args.dim[2] * 4
                sample_targets = int(d_targets.gpudata) + j * self.args.n_classes

                # Reset votes
                self.votes.fill(0)

                # 1. Evaluate clauses and select patches
                self.k_eval_clauses(
                    self.d_rng_states,
                    np.intp(sample_X),
                    self.ta_states,
                    d_clause_drop_mask,
                    self.selected_patch_ids,
                    self.clause_num_includes,
                    self.d_feat_mins,
                    self.d_literal_offsets,
                    block=block,
                    grid=grid,
                )

                # 2. Count votes
                self.k_count_votes(
                    self.selected_patch_ids,
                    self.clause_weights,
                    self.patch_weights,
                    self.votes,
                    block=block,
                    grid=grid,
                )

                # 3. Calculate update probabilities
                self.k_calc_update_prob(
                    self.votes,
                    np.intp(sample_targets),
                    self.prob,
                    block=block_classes,
                    grid=grid_classes,
                )

                # 4. Update clauses
                self.k_update_clauses(
                    self.d_rng_states,
                    self.selected_patch_ids,
                    self.clause_num_includes,
                    d_clause_drop_mask,
                    np.intp(sample_X),
                    np.intp(sample_targets),
                    self.prob,
                    self.ta_states,
                    self.clause_weights,
                    self.d_feat_mins,
                    self.d_literal_offsets,
                    block=block,
                    grid=grid,
                )

    def infer(self, X: np.ndarray, batch_size: int):
        N = X.shape[0]
        if batch_size == -1:
            batch_size = N

        class_sums = np.zeros((N, self.args.n_classes), dtype=np.float32)

        # Pre-compute clause information once
        self._run_infer_clauses()

        for i in tqdm(range(0, N, batch_size), desc="Infer batch", leave=False, dynamic_ncols=True):
            batch_end = min(i + batch_size, N)
            batch_X = np.ascontiguousarray(X[i:batch_end], dtype=np.int32)
            batch_size_actual = batch_end - i

            d_X = ga.to_gpu(batch_X)
            d_class_sums = ga.zeros((batch_size_actual, self.args.n_classes), dtype=np.float32)

            grid, block = self._kernel_config(batch_size_actual * self.total_clauses)

            self.k_infer_sample(
                d_X,
                self.clause_weights,
                d_class_sums,
                np.int32(batch_size_actual),
                self.d_feat_mins,
                self.clause_positions,
                self.valid_feat_ranges,
                self.clause_valid,
                self.num_includes,
                self.interesting_fids,
                self.interesting_fid_lens,
                block=block,
                grid=grid,
            )

            class_sums[i:batch_end] = d_class_sums.get()

        return class_sums

    def transform_patchwise(self, X: np.ndarray, batch_size: int):
        N = X.shape[0]
        if batch_size == -1:
            batch_size = N

        patch_output = np.zeros((N, self.total_clauses, self.n_patches), dtype=np.int8)

        # Pre-compute clause information once
        self._run_infer_clauses()

        for i in tqdm(range(0, N, batch_size), desc="Transform batch", leave=False, dynamic_ncols=True):
            batch_end = min(i + batch_size, N)
            batch_X = np.ascontiguousarray(X[i:batch_end], dtype=np.int32)
            batch_size_actual = batch_end - i

            d_X = ga.to_gpu(batch_X)
            d_patch_output = ga.zeros((batch_size_actual, self.total_clauses, self.n_patches), dtype=np.int8)

            grid, block = self._kernel_config(batch_size_actual * self.total_clauses)

            self.k_transform_patchwise(
                d_X,
                d_patch_output,
                np.int32(batch_size_actual),
                self.d_feat_mins,
                self.clause_positions,
                self.valid_feat_ranges,
                self.clause_valid,
                self.num_includes,
                self.interesting_fids,
                self.interesting_fid_lens,
                block=block,
                grid=grid,
            )

            patch_output[i:batch_end] = d_patch_output.get()

        return patch_output

    def get_weights(self):
        return self.clause_weights.get()

    def get_ta_states(self):
        return self.ta_states.get().reshape((self.n_clause_banks, self.args.n_clauses, self.n_literals))

    def get_state_dict(self):
        return {
            "ta_states": self.ta_states.get(),
            "clause_weights": self.clause_weights.get(),
            "patch_weights": self.patch_weights.get(),
        }

    def load_state_dict(self, state_dict: dict):
        self.ta_states = ga.to_gpu(state_dict["ta_states"])
        self.clause_weights = ga.to_gpu(state_dict["clause_weights"])
        self.patch_weights = ga.to_gpu(state_dict["patch_weights"])
