import os
import warnings

import numpy as np
import cupy as cp
from tqdm import tqdm
from ..base import BaseDevice, PackedClauses
from cuda.pathfinder import find_nvidia_header_directory

curand_include = find_nvidia_header_directory("curand")


class PackedClausesCUDA(PackedClauses):
    def to_cpu(self):
        self.clause_position_bounds = self.clause_position_bounds.get()
        self.clause_feat_bounds = self.clause_feat_bounds.get()
        self.constrained_fids = self.constrained_fids.get()
        self.n_constrained = self.n_constrained.get()
        self.num_includes = self.num_includes.get()
        self.is_clause_valid = self.is_clause_valid.get()
        self.is_clause_synced = self.is_clause_synced.get()


def read_file(path):
    with open(path, "r") as f:
        return f.read()


class CupyDevice(BaseDevice):
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
#define WARP_SIZE {self.cuda_props["warp_size"]}
"""
        return header

    def _init_kernels(self):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        header = self._build_header()

        pack_mod = cp.RawModule(
            code=header + "\n" + read_file(os.path.join(cur_dir, "pack_clauses.cu")),
            options=("--use_fast_math",),
        )
        self.k_pack_clauses = pack_mod.get_function("pack_clauses")

        mod = cp.RawModule(
            code=header + "\n" + read_file(os.path.join(cur_dir, "kernel.cu")),
            options=("--use_fast_math", f"-I{curand_include}"),
        )
        self.k_eval_clauses = mod.get_function("eval_clauses")
        self.k_select_patch_and_count_votes = mod.get_function("select_patch_and_count_votes")
        self.k_calc_update_prob = mod.get_function("calc_update_prob")
        self.k_update_clauses = mod.get_function("update_clauses")
        self.k_infer_batch = mod.get_function("infer_batch")
        self.k_transform_patchwise = mod.get_function("transform_patchwise")

        self.kconf_clauses = self._kernel_config(self.total_clauses)
        self.kconf_clauses_warp = self._kernel_config(self.total_clauses * self.cuda_props["warp_size"])
        self.kconf_clause_patches = self._kernel_config(self.total_clauses * self.n_patches)
        self.kconf_classes = self._kernel_config(self.args.n_classes)

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

    def _kernel_config(self, n) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        bs = min(self.args.block_size, self.cuda_props["max_threads_per_block"])
        if self.args.grid_size is None:
            gs = (n + bs - 1) // bs
        else:
            gs = self.args.grid_size
        return (gs, 1, 1), (bs, 1, 1)

    def _init_rng(self):
        """Find size of curandState and init RNG state per clause."""
        mod = cp.RawModule(
            code=r"""
            #include <curand_kernel.h>

            extern "C" __global__ void get_curandstate_size(unsigned int* size) {
                *size = (unsigned int)sizeof(curandState);
            }

            extern "C" __global__ void setup_rng(const unsigned int* base_seeds, curandState* rng_state, int n) {
                int tid = blockDim.x * blockIdx.x + threadIdx.x;
                for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
                    curand_init(base_seeds[i], 0, 0, &rng_state[i]);
                }
            }
            """,
            options=(f"-I{curand_include}",),
        )
        curand_state_bytes = cp.zeros(1, dtype=np.uint32)
        mod.get_function("get_curandstate_size")((1,), (1,), (curand_state_bytes,))

        base_seeds = cp.asarray(self.np_rng.integers(1, 2**30, size=self.total_clauses, dtype=np.uint32))
        self.rng = cp.empty((self.total_clauses * int(curand_state_bytes[0])), dtype=np.uint8)
        mod.get_function("setup_rng")(
            *self._kernel_config(self.total_clauses), (base_seeds, self.rng, cp.int32(self.total_clauses))
        )

    def dev_init(self):
        self.cuda_dev = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(self.cuda_dev.id)
        self.cuda_props = {
            "max_threads_per_block": props["maxThreadsPerBlock"],
            "multiprocessor_count": props["multiProcessorCount"],
            "warp_size": props["warpSize"],
        }

        # Setup RNG states
        self._init_rng()
        self._init_clauses()
        self._init_weights()
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

        # Persistent buffers for training (sparse range representation matching CPU)
        clause_position_bounds = cp.empty((self.total_clauses, 4), dtype=np.int32)
        clause_feat_bounds = cp.empty((self.total_clauses, self.n_raw_patch_feats, 2), dtype=np.int32)
        constrained_fids = cp.empty((self.total_clauses, self.n_raw_patch_feats), dtype=np.int32)
        n_constrained = cp.empty(self.total_clauses, dtype=np.int32)
        num_includes = cp.empty(self.total_clauses, dtype=np.uint32)
        is_clause_valid = cp.empty(self.total_clauses, dtype=np.int8)
        is_clause_synced = cp.zeros(self.total_clauses, dtype=np.int8)
        clause_outputs = cp.empty((self.total_clauses, self.n_patches), dtype=np.int8)
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
                    *self.kconf_clauses_warp,
                    (
                        self.ta_states,
                        self.feat_mins_gpu,
                        self.feat_maxs_gpu,
                        self.literal_offsets_gpu,
                        clause_position_bounds,
                        clause_feat_bounds,
                        constrained_fids,
                        n_constrained,
                        num_includes,
                        is_clause_valid,
                        is_clause_synced,
                    ),
                )
                self.k_eval_clauses(
                    *self.kconf_clause_patches,
                    (
                        X_batch,
                        np.int32(e),
                        clause_drop_mask_gpu,
                        clause_position_bounds,
                        clause_feat_bounds,
                        constrained_fids,
                        n_constrained,
                        num_includes,
                        is_clause_valid,
                        clause_outputs,
                    ),
                )
                votes.fill(0)
                self.k_select_patch_and_count_votes(
                    *self.kconf_clauses,
                    (self.rng, clause_outputs, self.clause_weights, selected_patch_ids, self.patch_weights, votes),
                )
                self.k_calc_update_prob(
                    *self.kconf_classes,
                    (votes, tar_batch, np.int32(e), prob),
                )
                self.k_update_clauses(
                    *self.kconf_clauses,
                    (
                        self.rng,
                        selected_patch_ids,
                        num_includes,
                        clause_drop_mask_gpu,
                        X_batch,
                        tar_batch,
                        np.int32(e),
                        prob,
                        self.ta_states,
                        self.clause_weights,
                        self.feat_mins_gpu,
                        self.literal_offsets_gpu,
                        is_clause_synced,
                    ),
                )


    def pack_clauses(self):
        clause_position_bounds = cp.empty((self.total_clauses, 4), dtype=np.int32)
        clause_feat_bounds = cp.empty((self.total_clauses, self.n_raw_patch_feats, 2), dtype=np.int32)
        constrained_fids = cp.empty((self.total_clauses, self.n_raw_patch_feats), dtype=np.int32)
        n_constrained = cp.empty(self.total_clauses, dtype=np.int32)
        num_includes = cp.empty(self.total_clauses, dtype=np.uint32)
        is_clause_valid = cp.empty(self.total_clauses, dtype=np.int8)
        is_clause_synced = cp.zeros(self.total_clauses, dtype=np.int8)

        self.k_pack_clauses(
            *self.kconf_clauses_warp,
            (self.ta_states, self.feat_mins_gpu, self.feat_maxs_gpu, self.literal_offsets_gpu,
             clause_position_bounds, clause_feat_bounds, constrained_fids, n_constrained,
             num_includes, is_clause_valid, is_clause_synced),
        )

        return PackedClausesCUDA(
            clause_position_bounds=clause_position_bounds,
            clause_feat_bounds=clause_feat_bounds,
            constrained_fids=constrained_fids,
            n_constrained=n_constrained,
            num_includes=num_includes,
            is_clause_valid=is_clause_valid,
            is_clause_synced=is_clause_synced,
        )

    def infer(self, X: np.ndarray, batch_size: int):
        N = X.shape[0]
        if batch_size == -1:
            batch_size = N

        class_sums = np.zeros((N, self.args.n_classes), dtype=np.float32)
        bufs = self.pack_clauses()

        for i in tqdm(range(0, N, batch_size), desc="Infer batch", leave=False, dynamic_ncols=True):
            batch_end = min(i + batch_size, N)
            batch_X = cp.asarray(X[i:batch_end], dtype=np.int32)
            bs = batch_end - i

            cs_batch = cp.zeros((bs, self.args.n_classes), dtype=np.float32)

            self.k_infer_batch(
                *self._kernel_config(bs * self.total_clauses),
                (batch_X, self.clause_weights, cs_batch, np.int32(bs),
                 bufs.clause_position_bounds, bufs.clause_feat_bounds, bufs.constrained_fids,
                 bufs.n_constrained, bufs.num_includes, bufs.is_clause_valid),
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
            batch_X = cp.asarray(X[i:batch_end], dtype=np.int32)
            bs = batch_end - i
            po_batch = cp.zeros((bs, self.total_clauses, self.n_patches), dtype=np.int8)

            self.k_transform_patchwise(
                *self._kernel_config(bs * self.total_clauses * self.n_patches),
                (batch_X, po_batch, np.int32(bs), bufs.clause_position_bounds, bufs.clause_feat_bounds,
                 bufs.constrained_fids, bufs.n_constrained, bufs.num_includes, bufs.is_clause_valid),
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
