import numpy as np
import os
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

        # Training kernels (5-kernel approach)
        self.k_pack_clauses = mod.get_function("pack_clauses")
        self.k_pack_clauses.prepare("PPPPPPPPP")  # 9 args (+clause_dirty)

        self.k_eval_clauses = mod.get_function("eval_clauses")
        self.k_eval_clauses.prepare("PiPPPPPPPPPPP")  # 13 args (+lit_to_fid)

        self.k_select_patch_and_count_votes = mod.get_function("select_patch_and_count_votes")
        self.k_select_patch_and_count_votes.prepare("PPPPPP")  # 6 args

        self.k_calc_update_prob = mod.get_function("calc_update_prob")
        self.k_calc_update_prob.prepare("PPiP")  # 4 args

        self.k_update_clauses = mod.get_function("update_clauses")
        self.k_update_clauses.prepare("PPPPPPPPPPPPP")  # 13 args (+clause_dirty)

        # Inference kernels
        self.k_infer_batch = mod.get_function("infer_batch")
        self.k_infer_batch.prepare("PPPiPPPPPPPPP")  # 13 args (+lit_to_fid)

        self.k_transform_patchwise = mod.get_function("transform_patchwise")
        self.k_transform_patchwise.prepare("PPiPPPPPPPPP")  # 12 args (+lit_to_fid)

        # Kernel configs
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
        self.patch_weights = ga.to_gpu(np.zeros((self.total_clauses, self.n_patches), dtype=np.int32))

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
        self.feat_mins_gpu = ga.to_gpu(self.args.feat_mins.astype(np.int32))
        self.literal_offsets_gpu = ga.to_gpu(self.literal_offsets.astype(np.int32))

        # Precompute lit_to_fid lookup table: O(1) lookup instead of O(N_RAW_PATCH_FEATS) scan
        lit_to_fid = np.zeros(self.n_patch_feats, dtype=np.int32)
        for fid in range(self.n_raw_patch_feats):
            for lit in range(self.literal_offsets[fid], self.literal_offsets[fid + 1]):
                lit_to_fid[lit] = fid
        self.lit_to_fid_gpu = ga.to_gpu(lit_to_fid)

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

        # Persistent buffers for training
        clause_positions = ga.empty((self.total_clauses, 4), dtype=np.int32)
        included_lits_pos = ga.empty((self.total_clauses, self.n_patch_feats), dtype=np.int32)
        included_lits_neg = ga.empty((self.total_clauses, self.n_patch_feats), dtype=np.int32)
        n_lits_pos = ga.empty(self.total_clauses, dtype=np.int32)
        n_lits_neg = ga.empty(self.total_clauses, dtype=np.int32)
        num_includes = ga.empty(self.total_clauses, dtype=np.uint32)
        clause_outputs = ga.empty((self.total_clauses, self.n_patches), dtype=np.int8)
        selected_patch_ids = ga.empty(self.total_clauses, dtype=np.int32)
        votes = ga.zeros(self.args.n_classes, dtype=np.float32)
        prob = ga.empty(self.args.n_classes, dtype=np.float32)
        clause_dirty = ga.to_gpu(np.ones(self.total_clauses, dtype=np.int8))

        for i in tqdm(range(0, N, batch_size), desc="Fit batch", leave=False, dynamic_ncols=True):
            batch_end = min(i + batch_size, N)
            X_batch = ga.to_gpu(np.ascontiguousarray(X[i:batch_end], dtype=np.int32))
            tar_batch = ga.to_gpu(np.ascontiguousarray(targets[i:batch_end], dtype=np.int8))
            bs = batch_end - i

            for e in tqdm(range(bs), desc="Sample", leave=False, dynamic_ncols=True):
                # K1: Pack clauses - scan TA states into sparse representation (skip unchanged)
                self.k_pack_clauses.prepared_call(
                    *self.kconf_clauses,
                    self.ta_states.gpudata,
                    self.literal_offsets_gpu.gpudata,
                    clause_positions.gpudata,
                    included_lits_pos.gpudata,
                    included_lits_neg.gpudata,
                    n_lits_pos.gpudata,
                    n_lits_neg.gpudata,
                    num_includes.gpudata,
                    clause_dirty.gpudata,
                )

                # K2: Eval clauses - parallel over (clause, patch) pairs
                self.k_eval_clauses.prepared_call(
                    *self.kconf_clause_patches,
                    X_batch.gpudata,
                    np.int32(e),
                    clause_drop_mask_gpu.gpudata,
                    self.feat_mins_gpu.gpudata,
                    self.literal_offsets_gpu.gpudata,
                    self.lit_to_fid_gpu.gpudata,
                    clause_positions.gpudata,
                    included_lits_pos.gpudata,
                    included_lits_neg.gpudata,
                    n_lits_pos.gpudata,
                    n_lits_neg.gpudata,
                    num_includes.gpudata,
                    clause_outputs.gpudata,
                )

                # K3: Select patch and count votes
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

                # K4: Calculate update probability
                self.k_calc_update_prob.prepared_call(
                    *self.kconf_classes,
                    votes.gpudata,
                    tar_batch.gpudata,
                    np.int32(e),
                    prob.gpudata,
                )

                # K5: Update clauses (marks dirty clauses for re-packing)
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
                    clause_dirty.gpudata,
                )

            self.ctx.synchronize()

    def _pack_clauses_for_inference(self):
        """Precompute sparse representation for all clauses (called once before inference)."""
        clause_positions = ga.empty((self.total_clauses, 4), dtype=np.int32)
        included_lits_pos = ga.empty((self.total_clauses, self.n_patch_feats), dtype=np.int32)
        included_lits_neg = ga.empty((self.total_clauses, self.n_patch_feats), dtype=np.int32)
        n_lits_pos = ga.empty(self.total_clauses, dtype=np.int32)
        n_lits_neg = ga.empty(self.total_clauses, dtype=np.int32)
        num_includes = ga.empty(self.total_clauses, dtype=np.uint32)
        clause_dirty = ga.to_gpu(np.ones(self.total_clauses, dtype=np.int8))

        self.k_pack_clauses.prepared_call(
            *self.kconf_clauses,
            self.ta_states.gpudata,
            self.literal_offsets_gpu.gpudata,
            clause_positions.gpudata,
            included_lits_pos.gpudata,
            included_lits_neg.gpudata,
            n_lits_pos.gpudata,
            n_lits_neg.gpudata,
            num_includes.gpudata,
            clause_dirty.gpudata,
        )

        return {
            "clause_positions": clause_positions,
            "included_lits_pos": included_lits_pos,
            "included_lits_neg": included_lits_neg,
            "n_lits_pos": n_lits_pos,
            "n_lits_neg": n_lits_neg,
            "num_includes": num_includes,
        }

    def infer(self, X: np.ndarray, batch_size: int):
        N = X.shape[0]
        if batch_size == -1:
            batch_size = N

        class_sums = np.zeros((N, self.args.n_classes), dtype=np.float32)
        bufs = self._pack_clauses_for_inference()

        for i in tqdm(range(0, N, batch_size), desc="Infer batch", leave=False, dynamic_ncols=True):
            batch_end = min(i + batch_size, N)
            batch_X = ga.to_gpu(np.ascontiguousarray(X[i:batch_end], dtype=np.int32))
            bs = batch_end - i

            cs_batch = ga.zeros((bs, self.args.n_classes), dtype=np.float32)

            self.k_infer_batch.prepared_call(
                *self._kernel_config(bs * self.total_clauses),
                batch_X.gpudata,
                self.clause_weights.gpudata,
                cs_batch.gpudata,
                np.int32(bs),
                self.feat_mins_gpu.gpudata,
                self.literal_offsets_gpu.gpudata,
                self.lit_to_fid_gpu.gpudata,
                bufs["clause_positions"].gpudata,
                bufs["included_lits_pos"].gpudata,
                bufs["included_lits_neg"].gpudata,
                bufs["n_lits_pos"].gpudata,
                bufs["n_lits_neg"].gpudata,
                bufs["num_includes"].gpudata,
            )

            class_sums[i:batch_end] = cs_batch.get()

        return class_sums

    def transform_patchwise(self, X: np.ndarray, batch_size: int):
        N = X.shape[0]
        if batch_size == -1:
            batch_size = N

        patch_output = np.zeros((N, self.total_clauses, self.n_patches), dtype=np.int8)
        bufs = self._pack_clauses_for_inference()

        for i in tqdm(range(0, N, batch_size), desc="Transform batch", leave=False, dynamic_ncols=True):
            batch_end = min(i + batch_size, N)
            batch_X = ga.to_gpu(np.ascontiguousarray(X[i:batch_end], dtype=np.int32))
            bs = batch_end - i
            po_batch = ga.zeros((bs, self.total_clauses, self.n_patches), dtype=np.int8)

            self.k_transform_patchwise.prepared_call(
                *self._kernel_config(bs * self.total_clauses * self.n_patches),
                batch_X.gpudata,
                po_batch.gpudata,
                np.int32(bs),
                self.feat_mins_gpu.gpudata,
                self.literal_offsets_gpu.gpudata,
                self.lit_to_fid_gpu.gpudata,
                bufs["clause_positions"].gpudata,
                bufs["included_lits_pos"].gpudata,
                bufs["included_lits_neg"].gpudata,
                bufs["n_lits_pos"].gpudata,
                bufs["n_lits_neg"].gpudata,
                bufs["num_includes"].gpudata,
            )

            patch_output[i:batch_end] = po_batch.get()

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
