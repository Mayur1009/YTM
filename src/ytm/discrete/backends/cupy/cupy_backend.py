import os
import warnings

import numpy as np
import cupy as cp
from tqdm import tqdm
from ..base import BaseDevice, PackedClauses


class PackedClausesCUDA(PackedClauses):
    def to_cpu(self):
        self.clause_position_bounds = self.clause_position_bounds.get()
        self.clause_feat_bounds = self.clause_feat_bounds.get()
        self.bounded_feat_ids = self.bounded_feat_ids.get()
        self.n_bounded_feats = self.n_bounded_feats.get()
        self.clause_density = self.clause_density.get()
        self.is_clause_synced = self.is_clause_synced.get()


def read_file(path):
    with open(path, "r") as f:
        return f.read()


class CupyDevice(BaseDevice):
    def _build_header(self):
        header = f"""
#define TOTAL_CLAUSES {self.total_clauses}
#define THRESH {float(self.args.T)}
#define S {float(self.args.s)}
#define CLASSES {self.args.n_classes}
#define HEIGHT {self.args.dim[0]}
#define WIDTH {self.args.dim[1]}
#define DEPTH {self.args.dim[2]}
#define PATCH_HEIGHT {self.args.patch_dim[0]}
#define PATCH_WIDTH {self.args.patch_dim[1]}
#define STRIDE_Y {self.args.stride[0]}
#define STRIDE_X {self.args.stride[1]}
#define MAX_WEIGHT {float(self.args.max_weight)}
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

    def _init_kernels(self):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        header = self._build_header()
        common = header + "\n" + read_file(os.path.join(cur_dir, "common.cu")) + "\n"

        pack_mod = cp.RawModule(
            code=common + read_file(os.path.join(cur_dir, "pack_clauses.cu")),
            options=("--use_fast_math",),
        )
        self.k_pack_clauses = pack_mod.get_function("pack_clauses")

        eval_mod = cp.RawModule(
            code=common + read_file(os.path.join(cur_dir, "evaluate.cu")),
            options=("--use_fast_math",),
        )
        self.k_evaluate = eval_mod.get_function("evaluate")
        self.k_count_votes = eval_mod.get_function("count_votes")

        update_mod = cp.RawModule(
            code=common + read_file(os.path.join(cur_dir, "update.cu")),
            options=("--use_fast_math",),
        )
        self.k_calc_update_prob = update_mod.get_function("calc_update_prob")
        self.k_update_clauses = update_mod.get_function("update_clauses")

        infer_mod = cp.RawModule(
            code=common + read_file(os.path.join(cur_dir, "inference.cu")),
            options=("--use_fast_math",),
        )
        self.k_eval_clauses = infer_mod.get_function("infer_clauses")
        self.k_sum_votes = infer_mod.get_function("sum_votes")
        self.k_transform_patchwise = infer_mod.get_function("infer_clauses_patchwise")

        self.kconf_clauses = self._kernel_config(self.total_clauses * self.cuda_props["warp_size"])
        self.kconf_classes = self._kernel_config(self.args.n_classes * self.cuda_props["warp_size"])

    def _init_clauses(self):
        self.ta_states = cp.full(
            (self.total_clauses, self.n_literals),
            self.args.include_state - 1,
            dtype=np.uint32,
        )

    def _init_weights(self):
        n_neg_polarity = self.args.n_clauses // 2
        self.clause_weights = cp.ones((self.args.n_classes, self.args.n_clauses), dtype=np.float32)
        if self.args.coalesced:
            for i in range(self.args.n_classes):
                wt = np.ones((self.args.n_clauses,), dtype=np.float32)
                wt[n_neg_polarity:] *= -1.0
                self.clause_weights[i, :] = cp.asarray(self.np_rng.permutation(wt))
        else:
            self.clause_weights[:, n_neg_polarity:] *= -1.0

        if self.args.track_patch_weights:
            self.patch_weights = cp.zeros((self.total_clauses, self.n_patches), dtype=np.int32)
        else:
            self.patch_weights = cp.zeros((1, 1), dtype=np.int32)

    def _init_packed_clauses(self):
        clause_position_bounds = cp.empty((self.total_clauses, 4), dtype=np.int32)
        clause_feat_bounds = cp.empty((self.total_clauses, self.n_raw_patch_feats, 2), dtype=np.int32)
        bounded_feat_ids = cp.empty((self.total_clauses, self.n_raw_patch_feats), dtype=np.int32)
        n_bounded_feats = cp.empty(self.total_clauses, dtype=np.int32)
        clause_density = cp.empty(self.total_clauses, dtype=np.int32)
        is_clause_synced = cp.zeros(self.total_clauses, dtype=np.int8)

        self.packed_clauses_gpu = PackedClausesCUDA(
            clause_position_bounds=clause_position_bounds,
            clause_feat_bounds=clause_feat_bounds,
            bounded_feat_ids=bounded_feat_ids,
            n_bounded_feats=n_bounded_feats,
            clause_density=clause_density,
            is_clause_synced=is_clause_synced,
        )

    def _kernel_config(self, n) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        bs = min(self.args.block_size, self.cuda_props["max_threads_per_block"])
        if self.args.grid_size is None:
            gs = (n + bs - 1) // bs
        else:
            gs = self.args.grid_size
        return (gs, 1, 1), (bs, 1, 1)

    def dev_init(self):
        self.cuda_dev = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(self.cuda_dev.id)
        self.cuda_props = {
            "max_threads_per_block": props["maxThreadsPerBlock"],
            "multiprocessor_count": props["multiProcessorCount"],
            "warp_size": props["warpSize"],
        }

        self.seed = np.uint64(self.args.seed)
        self._init_clauses()
        self._init_weights()
        self._init_packed_clauses()
        self._init_kernels()
        self.feat_mins_gpu = cp.asarray(self.args.feat_mins, dtype=np.int32)
        self.feat_maxs_gpu = cp.asarray(self.args.feat_maxs, dtype=np.int32)
        self.literal_offsets_gpu = cp.asarray(self.literal_offsets.astype(np.int32))

    def fit_epoch(self, X: np.ndarray, targets: np.ndarray, clause_drop_p: float, batch_size: int):
        N = X.shape[0]
        if batch_size == -1:
            batch_size = N

        # Clause dropout mask (same for entire epoch)
        if clause_drop_p > 0.0:
            clause_drop_mask_gpu = cp.asarray(self.np_rng.random(self.total_clauses) <= clause_drop_p, dtype=cp.uint8)
        else:
            clause_drop_mask_gpu = cp.zeros(self.total_clauses, dtype=np.uint8)

        selected_patch_ids = cp.empty(self.total_clauses, dtype=np.int32)
        votes = cp.zeros(self.args.n_classes, dtype=np.float32)
        prob = cp.empty(self.args.n_classes, dtype=np.float32)

        for i in tqdm(range(0, N, batch_size), desc="Fit batch", leave=False, dynamic_ncols=True):
            batch_end = min(i + batch_size, N)
            X_batch = cp.asarray(X[i:batch_end], dtype=np.int32)
            tar_batch = cp.asarray(targets[i:batch_end], dtype=np.float32)
            bs = batch_end - i

            for e in tqdm(range(bs), desc="Sample", leave=False, dynamic_ncols=True):
                self.k_pack_clauses(
                    *self.kconf_clauses,
                    (
                        self.ta_states,
                        self.feat_mins_gpu,
                        self.feat_maxs_gpu,
                        self.literal_offsets_gpu,
                        self.packed_clauses_gpu.clause_position_bounds,
                        self.packed_clauses_gpu.clause_feat_bounds,
                        self.packed_clauses_gpu.bounded_feat_ids,
                        self.packed_clauses_gpu.n_bounded_feats,
                        self.packed_clauses_gpu.clause_density,
                        self.packed_clauses_gpu.is_clause_synced,
                    ),
                )
                self.k_evaluate(
                    *self.kconf_clauses,
                    (
                        X_batch,
                        np.int32(e),
                        clause_drop_mask_gpu,
                        self.packed_clauses_gpu.clause_position_bounds,
                        self.packed_clauses_gpu.clause_feat_bounds,
                        self.packed_clauses_gpu.bounded_feat_ids,
                        self.packed_clauses_gpu.n_bounded_feats,
                        self.packed_clauses_gpu.clause_density,
                        self.seed,
                        selected_patch_ids,
                        self.patch_weights,
                    ),
                )
                self.k_count_votes(
                    *self.kconf_classes,
                    (selected_patch_ids, self.clause_weights, votes),
                )
                self.k_calc_update_prob(
                    *self.kconf_classes,
                    (votes, tar_batch, np.int32(e), prob),
                )
                self.k_update_clauses(
                    *self.kconf_clauses,
                    (
                        self.seed,
                        selected_patch_ids,
                        self.packed_clauses_gpu.clause_density,
                        clause_drop_mask_gpu,
                        X_batch,
                        tar_batch,
                        np.int32(e),
                        prob,
                        self.ta_states,
                        self.clause_weights,
                        self.feat_mins_gpu,
                        self.literal_offsets_gpu,
                        self.packed_clauses_gpu.is_clause_synced,
                    ),
                )

    def pack_clauses(self, force_repack: bool = False):
        if force_repack:
            self.packed_clauses_gpu.is_clause_synced.fill(0)

        self.k_pack_clauses(
            *self.kconf_clauses,
            (
                self.ta_states,
                self.feat_mins_gpu,
                self.feat_maxs_gpu,
                self.literal_offsets_gpu,
                self.packed_clauses_gpu.clause_position_bounds,
                self.packed_clauses_gpu.clause_feat_bounds,
                self.packed_clauses_gpu.bounded_feat_ids,
                self.packed_clauses_gpu.n_bounded_feats,
                self.packed_clauses_gpu.clause_density,
                self.packed_clauses_gpu.is_clause_synced,
            ),
        )

    def infer(self, X: np.ndarray, batch_size: int):
        N = X.shape[0]
        if batch_size == -1:
            batch_size = N

        class_sums = cp.zeros((N, self.args.n_classes), dtype=np.float32)
        self.pack_clauses()

        for i in tqdm(range(0, N, batch_size), desc="Infer batch", leave=False, dynamic_ncols=True):
            batch_end = min(i + batch_size, N)
            bs = batch_end - i
            batch_X = cp.asarray(X[i:batch_end], dtype=np.int32)
            co_batch = cp.empty((bs, self.total_clauses), dtype=cp.int8)

            self.k_eval_clauses(
                *self._kernel_config(bs * self.total_clauses * self.cuda_props["warp_size"]),
                (
                    batch_X,
                    co_batch,
                    np.int32(bs),
                    self.packed_clauses_gpu.clause_position_bounds,
                    self.packed_clauses_gpu.clause_feat_bounds,
                    self.packed_clauses_gpu.bounded_feat_ids,
                    self.packed_clauses_gpu.n_bounded_feats,
                    self.packed_clauses_gpu.clause_density,
                ),
            )

            self.k_sum_votes(
                *self._kernel_config(bs * self.args.n_classes * self.cuda_props["warp_size"]),
                (
                    co_batch,
                    self.clause_weights,
                    class_sums[i:batch_end],
                    np.int32(bs),
                ),
            )

        return class_sums.get()

    def transform(self, X: np.ndarray, batch_size: int):
        N = X.shape[0]
        if batch_size == -1:
            batch_size = N

        clause_outputs = np.zeros((N, self.total_clauses), dtype=np.int8)
        self.pack_clauses()

        for i in tqdm(range(0, N, batch_size), desc="Transform batch", leave=False, dynamic_ncols=True):
            batch_end = min(i + batch_size, N)
            bs = batch_end - i
            batch_X = cp.asarray(X[i:batch_end], dtype=np.int32)
            co_batch = cp.empty((bs, self.total_clauses), dtype=cp.int8)

            self.k_eval_clauses(
                *self._kernel_config(bs * self.total_clauses * self.cuda_props["warp_size"]),
                (
                    batch_X,
                    co_batch,
                    np.int32(bs),
                    self.packed_clauses_gpu.clause_position_bounds,
                    self.packed_clauses_gpu.clause_feat_bounds,
                    self.packed_clauses_gpu.bounded_feat_ids,
                    self.packed_clauses_gpu.n_bounded_feats,
                    self.packed_clauses_gpu.clause_density,
                ),
            )
            clause_outputs[i:batch_end] = co_batch.get()

        return clause_outputs

    def transform_patchwise(self, X: np.ndarray, batch_size: int):
        N = X.shape[0]
        if batch_size == -1:
            batch_size = N

        patch_output = np.zeros((N, self.total_clauses, self.n_patches), dtype=np.int8)
        self.pack_clauses()

        for i in tqdm(range(0, N, batch_size), desc="Transform batch", leave=False, dynamic_ncols=True):
            batch_end = min(i + batch_size, N)
            batch_X = cp.asarray(X[i:batch_end], dtype=np.int32)
            bs = batch_end - i
            po_batch = cp.zeros((bs, self.total_clauses, self.n_patches), dtype=np.int8)

            self.k_transform_patchwise(
                *self._kernel_config(bs * self.total_clauses * self.n_patches),
                (
                    batch_X,
                    po_batch,
                    np.int32(bs),
                    self.packed_clauses_gpu.clause_position_bounds,
                    self.packed_clauses_gpu.clause_feat_bounds,
                    self.packed_clauses_gpu.bounded_feat_ids,
                    self.packed_clauses_gpu.n_bounded_feats,
                    self.packed_clauses_gpu.clause_density,
                ),
            )
            patch_output[i:batch_end] = po_batch.get()

        return patch_output.reshape((N, self.n_clause_banks, self.args.n_clauses, self.n_patches_y, self.n_patches_x))

    def get_weights(self):
        return self.clause_weights.get()

    def get_ta_states(self):
        return self.ta_states.get().reshape((self.n_clause_banks, self.args.n_clauses, self.n_literals))

    def get_patch_weights(self):
        if not self.args.track_patch_weights:
            warnings.warn("track_patch_weights is False, so no patch_weights were saved.")
        return self.patch_weights.get().reshape(
            self.n_clause_banks, self.args.n_clauses, self.n_patches_y, self.n_patches_x
        )

    def get_state_dict(self):
        return {
            "ta_states": self.ta_states.get(),
            "clause_weights": self.clause_weights.get(),
            "patch_weights": self.patch_weights.get(),
        }

    def load_state_dict(self, state_dict: dict):
        self.ta_states = cp.asarray(state_dict["ta_states"])
        self.clause_weights = cp.asarray(state_dict["clause_weights"])
        self.patch_weights = cp.asarray(state_dict["patch_weights"])
        self.packed_clauses_gpu.is_clause_synced.fill(0)
