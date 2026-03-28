import os
import warnings

import numpy as np
from tqdm import tqdm
import pycuda.gpuarray as ga
from pycuda.compiler import SourceModule
from pycuda.curandom import XORWOWRandomNumberGenerator
from pycuda.driver import Context, device_attribute, memset_d32_async  # pyright: ignore # ty: ignore
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
#define TRACK_PATCH_WEIGHTS {1 if self.args.track_patch_weights else 0}
#define BOOST_TP_FB {1 if self.args.boost_tp_fb else 0}
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

        self.k_pack_clauses = mod.get_function("pack_clauses")
        self.k_pack_clauses.prepare("PPPPPPPPPP")

        self.k_eval_clauses = mod.get_function("eval_clauses")
        self.k_eval_clauses.prepare("PiPPPPPPPPPP")

        self.k_select_patch_and_count_votes = mod.get_function("select_patch_and_count_votes")
        self.k_select_patch_and_count_votes.prepare("PPPPPP")

        self.k_calc_update_prob = mod.get_function("calc_update_prob")
        self.k_calc_update_prob.prepare("PPiP")

        self.k_update_clauses = mod.get_function("update_clauses")
        self.k_update_clauses.prepare("PPPPPPPPPPPPP")

        self.k_infer_batch = mod.get_function("infer_batch")
        self.k_infer_batch.prepare("PPPiPPPPPPPP")

        self.k_transform_patchwise = mod.get_function("transform_patchwise")
        self.k_transform_patchwise.prepare("PPiPPPPPPPP")

        self.kconf_clauses = self._kernel_config(self.total_clauses)
        self.kconf_clause_patches = self._kernel_config(self.total_clauses * self.n_patches)
        self.kconf_classes = self._kernel_config(self.args.n_classes)

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
            wt = np.ones((self.args.n_clauses,), dtype=np.float32)
            wt[n_neg_polarity:] *= -1.0
            clause_weights[i, :] = self.np_rng.permutation(wt) if self.args.coalesced else wt

        self.clause_weights = ga.to_gpu(clause_weights)
        if self.args.track_patch_weights:
            self.patch_weights = ga.to_gpu(np.zeros((self.total_clauses, self.n_patches), dtype=np.int32))
        else:
            self.patch_weights = ga.to_gpu(np.zeros((1, 1), dtype=np.int32))

    def _kernel_config(self, n) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        bs = min(self.args.block_size, self.cuda_props["max_threads_per_block"])

        if self.args.grid_size is None:
            gs = (n + bs - 1) // bs
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
        self._init_kernels()
        self.feat_mins_gpu = ga.to_gpu(np.asarray(self.args.feat_mins, dtype=np.int32))
        self.literal_offsets_gpu = ga.to_gpu(self.literal_offsets.astype(np.int32))

    def fit_epoch(self, X: np.ndarray, targets: np.ndarray, clause_drop_p: float, batch_size: int):
        N = X.shape[0]
        if batch_size == -1:
            batch_size = N

        # Clause dropout mask (same for entire epoch)
        if clause_drop_p > 0.0:
            clause_drop_mask = (self.np_rng.random(self.total_clauses) <= clause_drop_p).astype(np.int8)
        else:
            clause_drop_mask = np.zeros(self.total_clauses, dtype=np.int8)
        clause_drop_mask_gpu = ga.to_gpu(clause_drop_mask)

        # Persistent buffers for training (sparse range representation matching CPU)
        clause_positions = ga.empty((self.total_clauses, 4), dtype=np.int32)
        clause_feat_min = ga.empty((self.total_clauses, self.n_raw_patch_feats), dtype=np.int32)
        clause_feat_max = ga.empty((self.total_clauses, self.n_raw_patch_feats), dtype=np.int32)
        constrained_fids = ga.empty((self.total_clauses, self.n_raw_patch_feats), dtype=np.int32)
        n_constrained = ga.empty(self.total_clauses, dtype=np.int32)
        num_includes = ga.empty(self.total_clauses, dtype=np.uint32)
        is_clause_valid = ga.empty(self.total_clauses, dtype=np.int8)
        is_clause_synced = ga.to_gpu(np.zeros(self.total_clauses, dtype=np.int8))  # Start unsynced
        clause_outputs = ga.empty((self.total_clauses, self.n_patches), dtype=np.int8)
        selected_patch_ids = ga.empty(self.total_clauses, dtype=np.int32)
        votes = ga.to_gpu(np.zeros(self.args.n_classes, dtype=np.float32))
        prob = ga.empty(self.args.n_classes, dtype=np.float32)

        for i in tqdm(range(0, N, batch_size), desc="Fit batch", leave=False, dynamic_ncols=True):
            batch_end = min(i + batch_size, N)
            X_batch = ga.to_gpu(np.ascontiguousarray(X[i:batch_end], dtype=np.int32))
            tar_batch = ga.to_gpu(np.ascontiguousarray(targets[i:batch_end], dtype=np.float32))
            bs = batch_end - i

            for e in tqdm(range(bs), desc="Sample", leave=False, dynamic_ncols=True):
                self.k_pack_clauses.prepared_call(
                    *self.kconf_clauses,
                    self.ta_states.gpudata,
                    self.literal_offsets_gpu.gpudata,
                    clause_positions.gpudata,
                    clause_feat_min.gpudata,
                    clause_feat_max.gpudata,
                    constrained_fids.gpudata,
                    n_constrained.gpudata,
                    num_includes.gpudata,
                    is_clause_valid.gpudata,
                    is_clause_synced.gpudata,
                )

                self.k_eval_clauses.prepared_call(
                    *self.kconf_clause_patches,
                    X_batch.gpudata,
                    np.int32(e),
                    clause_drop_mask_gpu.gpudata,
                    self.feat_mins_gpu.gpudata,
                    clause_positions.gpudata,
                    clause_feat_min.gpudata,
                    clause_feat_max.gpudata,
                    constrained_fids.gpudata,
                    n_constrained.gpudata,
                    num_includes.gpudata,
                    is_clause_valid.gpudata,
                    clause_outputs.gpudata,
                )

                memset_d32_async(votes.gpudata, 0, self.args.n_classes)
                self.k_select_patch_and_count_votes.prepared_call(
                    *self.kconf_clauses,
                    self.rng.state,
                    clause_outputs.gpudata,
                    self.clause_weights.gpudata,
                    selected_patch_ids.gpudata,
                    self.patch_weights.gpudata,
                    votes.gpudata,
                )

                self.k_calc_update_prob.prepared_call(
                    *self.kconf_classes,
                    votes.gpudata,
                    tar_batch.gpudata,
                    np.int32(e),
                    prob.gpudata,
                )

                self.k_update_clauses.prepared_call(
                    *self.kconf_clauses,
                    self.rng.state,
                    selected_patch_ids.gpudata,
                    num_includes.gpudata,
                    clause_drop_mask_gpu.gpudata,
                    X_batch.gpudata,
                    tar_batch.gpudata,
                    np.int32(e),
                    prob.gpudata,
                    self.ta_states.gpudata,
                    self.clause_weights.gpudata,
                    self.feat_mins_gpu.gpudata,
                    self.literal_offsets_gpu.gpudata,
                    is_clause_synced.gpudata,
                )

            self.ctx.synchronize()

    def pack_clauses(self):
        clause_positions = ga.empty((self.total_clauses, 4), dtype=np.int32)
        clause_feat_min = ga.empty((self.total_clauses, self.n_raw_patch_feats), dtype=np.int32)
        clause_feat_max = ga.empty((self.total_clauses, self.n_raw_patch_feats), dtype=np.int32)
        constrained_fids = ga.empty((self.total_clauses, self.n_raw_patch_feats), dtype=np.int32)
        n_constrained = ga.empty(self.total_clauses, dtype=np.int32)
        num_includes = ga.empty(self.total_clauses, dtype=np.uint32)
        is_clause_valid = ga.empty(self.total_clauses, dtype=np.int8)
        is_clause_synced = ga.to_gpu(np.zeros(self.total_clauses, dtype=np.int8))  # Start unsynced

        self.k_pack_clauses.prepared_call(
            *self.kconf_clauses,
            self.ta_states.gpudata,
            self.literal_offsets_gpu.gpudata,
            clause_positions.gpudata,
            clause_feat_min.gpudata,
            clause_feat_max.gpudata,
            constrained_fids.gpudata,
            n_constrained.gpudata,
            num_includes.gpudata,
            is_clause_valid.gpudata,
            is_clause_synced.gpudata,
        )

        return {
            "clause_positions": clause_positions,
            "clause_feat_min": clause_feat_min,
            "clause_feat_max": clause_feat_max,
            "constrained_fids": constrained_fids,
            "n_constrained": n_constrained,
            "num_includes": num_includes,
            "is_clause_valid": is_clause_valid,
        }

    def infer(self, X: np.ndarray, batch_size: int):
        N = X.shape[0]
        if batch_size == -1:
            batch_size = N

        class_sums = np.zeros((N, self.args.n_classes), dtype=np.float32)
        bufs = self.pack_clauses()

        for i in tqdm(range(0, N, batch_size), desc="Infer batch", leave=False, dynamic_ncols=True):
            batch_end = min(i + batch_size, N)
            batch_X = ga.to_gpu(np.ascontiguousarray(X[i:batch_end], dtype=np.int32))
            bs = batch_end - i

            cs_batch = ga.to_gpu(np.zeros((bs, self.args.n_classes), dtype=np.float32))

            self.k_infer_batch.prepared_call(
                *self._kernel_config(bs * self.total_clauses),
                batch_X.gpudata,
                self.clause_weights.gpudata,
                cs_batch.gpudata,
                np.int32(bs),
                self.feat_mins_gpu.gpudata,
                bufs["clause_positions"].gpudata,
                bufs["clause_feat_min"].gpudata,
                bufs["clause_feat_max"].gpudata,
                bufs["constrained_fids"].gpudata,
                bufs["n_constrained"].gpudata,
                bufs["num_includes"].gpudata,
                bufs["is_clause_valid"].gpudata,
            )

            class_sums[i:batch_end] = cs_batch.get()

        return class_sums

    def transform_patchwise(self, X: np.ndarray, batch_size: int):
        N = X.shape[0]
        if batch_size == -1:
            batch_size = N

        patch_output = np.zeros((N, self.total_clauses, self.n_patches), dtype=np.int8)
        bufs = self.pack_clauses()

        for i in tqdm(range(0, N, batch_size), desc="Transform batch", leave=False, dynamic_ncols=True):
            batch_end = min(i + batch_size, N)
            batch_X = ga.to_gpu(np.ascontiguousarray(X[i:batch_end], dtype=np.int32))
            bs = batch_end - i
            po_batch = ga.to_gpu(np.zeros((bs, self.total_clauses, self.n_patches), dtype=np.int8))

            self.k_transform_patchwise.prepared_call(
                *self._kernel_config(bs * self.total_clauses * self.n_patches),
                batch_X.gpudata,
                po_batch.gpudata,
                np.int32(bs),
                self.feat_mins_gpu.gpudata,
                bufs["clause_positions"].gpudata,
                bufs["clause_feat_min"].gpudata,
                bufs["clause_feat_max"].gpudata,
                bufs["constrained_fids"].gpudata,
                bufs["n_constrained"].gpudata,
                bufs["num_includes"].gpudata,
                bufs["is_clause_valid"].gpudata,
            )

            patch_output[i:batch_end] = po_batch.get()

        return patch_output

    def get_weights(self):
        return self.clause_weights.get()

    def get_ta_states(self):
        return self.ta_states.get().reshape((self.n_clause_banks, self.args.n_clauses, self.n_literals))

    def get_patch_weights(self):
        if not self.args.track_patch_weights:
            warnings.warn("track_patch_weights is False, so no patch_weights were saved.")
            return self.patch_weights.get()
        return self.patch_weights.get().reshape(self.total_clauses, self.n_patches_y, self.n_patches_x)

    def get_clauses(self):
        # WARN: Needs testing. Probably wrong.
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

        # Transfer to CPU for processing
        clause_feat_min = bufs["clause_feat_min"].get()
        clause_feat_max = bufs["clause_feat_max"].get()
        clause_positions = bufs["clause_positions"].get()
        is_clause_valid = bufs["is_clause_valid"].get()

        feat_mins = self.args.feat_mins

        # Convert from shifted bounds to original feature space
        # clause_feat_min/max are in shifted space (value - feat_min)
        feature_bounds = np.zeros((self.total_clauses, self.n_raw_patch_feats, 2), dtype=np.int32)
        feature_bounds[:, :, 0] = clause_feat_min + feat_mins  # lower bounds
        feature_bounds[:, :, 1] = clause_feat_max + feat_mins  # upper bounds

        # Position bounds — already closed interval [min, max]
        position_bounds = None
        if self.args.position_literals:
            position_bounds = clause_positions.copy()

        return {
            "feature_bounds": feature_bounds,
            "position_bounds": position_bounds,
            "is_valid": is_clause_valid.astype(bool),
        }

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
