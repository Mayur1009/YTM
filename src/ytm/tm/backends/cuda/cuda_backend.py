import os
import warnings

import numpy as np
import pycuda.gpuarray as ga
from pycuda.compiler import SourceModule
from pycuda.curandom import XORWOWRandomNumberGenerator
from pycuda.driver import Context, device_attribute, memset_d32, memset_d32_async  # pyright: ignore # ty: ignore
from tqdm import tqdm

from .. import BaseDevice, FitBuffers


def read_file(path):
    with open(path, "r") as f:
        return f.read()


class CUDADevice(BaseDevice):
    def dev_init(self):
        import pycuda.autoprimaryctx  # noqa: F401

        self.ctx: Context = pycuda.autoprimaryctx.context

        self.rng = XORWOWRandomNumberGenerator(
            lambda count: ga.to_gpu(np.array([(self.args.seed + i) for i in range(1, count + 1)], dtype=np.int32))
        )
        self.np_rng = np.random.default_rng(self.args.seed)

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

    def _init_clauses(self):
        self.ta_states = ga.to_gpu(
            np.full(
                shape=(self.total_clauses, self.n_literals),
                fill_value=self.args.include_state - 1,
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
        if self.args.track_patch_weights:
            self.patch_weights = ga.to_gpu(np.zeros((self.total_clauses, self.n_patches), dtype=np.int32))
        else:
            self.patch_weights = ga.to_gpu(np.zeros((1, 1), dtype=np.int32))

    def _init_kernels(self):
        cur_dir = os.path.dirname(os.path.abspath(__file__))

        self.header = f"""
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

        mod_kernels = self._load_kernel(os.path.join(cur_dir, "kernels.cu"), self.header)

        self.kernel_encode = mod_kernels.get_function("encode")
        self.kernel_pack_clauses = mod_kernels.get_function("pack_clauses")
        self.kernel_eval_clauses = mod_kernels.get_function("eval_clauses")
        self.kernel_select_patch = mod_kernels.get_function("select_patch_and_count_votes")
        self.kernel_evidence_to_prob = mod_kernels.get_function("evidence_to_update_prob")
        self.kernel_clause_update = mod_kernels.get_function("update_clauses")
        self.kernel_clause_inference = mod_kernels.get_function("clause_inference")

        self.kernel_encode.prepare("PiP")
        self.kernel_pack_clauses.prepare("PPP")
        self.kernel_eval_clauses.prepare("PPPPiP")
        self.kernel_select_patch.prepare("PPPPPPP")
        self.kernel_evidence_to_prob.prepare("PPPiP")
        self.kernel_clause_update.prepare("PPPPPPPiPP")
        self.kernel_clause_inference.prepare("PPPPiP")

        self.kernel_pack_clauses_launch_config = self._kernel_config(self.total_clauses)
        self.kernel_eval_clauses_launch_config = self._kernel_config(self.total_clauses * self.n_patches)
        self.kernel_select_patch_launch_config = self._kernel_config(self.total_clauses)
        self.kernel_evidence_to_prob_launch_config = self._kernel_config(self.args.n_classes)
        self.kernel_clause_update_launch_config = self._kernel_config(self.total_clauses)

    def _load_kernel(self, kernel_file, header):
        kernel_code = read_file(kernel_file)
        return SourceModule(
            header + "\n" + kernel_code,
            options=["-O3", "--use_fast_math"],
            no_extern_c=True,
        )

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

    def encode(self, X: np.ndarray[tuple[int, int], np.dtype[np.int8]]):
        N = X.shape[0]
        X_gpu = ga.to_gpu(X.astype(np.int8))
        encoded_X_gpu = ga.to_gpu(np.zeros((N, self.n_patches, self.n_literal_chunks), dtype=np.uint32))

        self.kernel_encode.prepared_call(
            *self._kernel_config(N),
            X_gpu.gpudata,
            np.int32(N),
            encoded_X_gpu.gpudata,
        )
        self.ctx.synchronize()

        return encoded_X_gpu.get()

    def prepare_fit_buffers(
        self, encoded_X: np.ndarray, targets: np.ndarray, clause_drop_mask: np.ndarray
    ) -> FitBuffers:
        return FitBuffers(
            encoded_X=ga.to_gpu(encoded_X.astype(np.uint32)),
            targets=ga.to_gpu(targets.astype(np.float32)),
            packed_clauses=ga.empty((self.total_clauses, self.n_literal_chunks), dtype=np.uint32),
            n_includes=ga.empty((self.total_clauses,), dtype=np.uint32),
            clause_outputs=ga.empty((self.total_clauses * self.n_patches,), dtype=np.int8),
            selected_patch_ids=ga.empty((self.total_clauses,), dtype=np.int32),
            pos_votes=ga.empty((self.args.n_classes,), dtype=np.float32),
            neg_votes=ga.empty((self.args.n_classes,), dtype=np.float32),
            update_probs=ga.empty((self.args.n_classes,), dtype=np.float32),
            clause_drop_mask=ga.to_gpu(clause_drop_mask.astype(np.int8)),
        )

    def pack_clauses(self, packed_clauses: ga.GPUArray, n_includes: ga.GPUArray):
        memset_d32_async(packed_clauses.gpudata, 0, self.total_clauses * self.n_literal_chunks)
        self.kernel_pack_clauses.prepared_call(
            *self._kernel_config(self.total_clauses),
            self.ta_states.gpudata,
            packed_clauses.gpudata,
            n_includes.gpudata,
        )

    def eval_clauses(
        self,
        packed_clauses: ga.GPUArray,
        n_includes: ga.GPUArray,
        clause_drop_mask: ga.GPUArray,
        clause_outputs: ga.GPUArray,
        encoded_X: ga.GPUArray,
        e: int,
    ):
        self.kernel_eval_clauses.prepared_call(
            *self.kernel_eval_clauses_launch_config,
            packed_clauses.gpudata,
            n_includes.gpudata,
            clause_drop_mask.gpudata,
            encoded_X.gpudata,
            np.int32(e),
            clause_outputs.gpudata,
        )

    def select_patch_and_count_votes(
        self,
        clause_outputs: ga.GPUArray,
        selected_patch_ids: ga.GPUArray,
        pos_votes: ga.GPUArray,
        neg_votes: ga.GPUArray,
    ):
        memset_d32_async(pos_votes.gpudata, 0, self.args.n_classes)
        memset_d32_async(neg_votes.gpudata, 0, self.args.n_classes)
        self.kernel_select_patch.prepared_call(
            *self.kernel_select_patch_launch_config,
            self.rng.state,
            self.clause_weights.gpudata,
            clause_outputs.gpudata,
            self.patch_weights.gpudata,
            selected_patch_ids.gpudata,
            pos_votes.gpudata,
            neg_votes.gpudata,
        )

    def calc_update_prob(
        self,
        pos_votes: ga.GPUArray,
        neg_votes: ga.GPUArray,
        targets: ga.GPUArray,
        update_probs: ga.GPUArray,
        e: int,
    ):
        self.kernel_evidence_to_prob.prepared_call(
            *self.kernel_evidence_to_prob_launch_config,
            pos_votes.gpudata,
            neg_votes.gpudata,
            targets.gpudata,
            np.int32(e),
            update_probs.gpudata,
        )

    def update_clauses(
        self,
        n_includes: ga.GPUArray,
        selected_patch_ids: ga.GPUArray,
        clause_drop_mask: ga.GPUArray,
        update_probs: ga.GPUArray,
        encoded_X: ga.GPUArray,
        targets: ga.GPUArray,
        e: int,
    ):
        self.kernel_clause_update.prepared_call(
            *self.kernel_clause_update_launch_config,
            self.rng.state,
            selected_patch_ids.gpudata,
            n_includes.gpudata,
            clause_drop_mask.gpudata,
            encoded_X.gpudata,
            targets.gpudata,
            update_probs.gpudata,
            np.int32(e),
            self.ta_states.gpudata,
            self.clause_weights.gpudata,
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
            self.ctx.synchronize()

    def infer(self, encoded_X: np.ndarray, batch_size: int = -1):
        N = encoded_X.shape[0]
        if batch_size == -1:
            batch_size = N

        packed_clauses = ga.empty((self.total_clauses, self.n_literal_chunks), dtype=np.uint32)
        n_includes = ga.empty((self.total_clauses,), dtype=np.uint32)

        self.pack_clauses(packed_clauses, n_includes)
        self.ctx.synchronize()

        class_sums = np.zeros((N, self.args.n_classes), dtype=np.float32)
        for i in tqdm(range(0, N, batch_size), desc="Inference", leave=False, dynamic_ncols=True):
            X_batch = ga.to_gpu(encoded_X[i : i + batch_size])
            cs_batch = ga.to_gpu(np.zeros((X_batch.shape[0], self.args.n_classes), dtype=np.float32))

            self.kernel_clause_inference.prepared_call(
                *self._kernel_config(X_batch.shape[0] * self.total_clauses),
                packed_clauses.gpudata,
                self.clause_weights.gpudata,
                n_includes.gpudata,
                X_batch.gpudata,
                np.int32(X_batch.shape[0]),
                cs_batch.gpudata,
            )
            self.ctx.synchronize()

            class_sums[i : i + batch_size] = cs_batch.get()

        return class_sums

    def get_weights(self) -> np.ndarray[tuple[int, int], np.dtype[np.float32]]:
        return self.clause_weights.get()

    def get_ta_states(self) -> np.ndarray[tuple[int, int, int], np.dtype[np.uint32]]:
        n_clause_banks = 1 if self.args.coalesced else self.args.n_classes
        return self.ta_states.get().reshape((n_clause_banks, self.args.n_clauses, self.n_literals))

    def get_patch_weights(self) -> np.ndarray:
        if not self.args.track_patch_weights:
            warnings.warn("track_patch_weights is False, so no patch_weights were saved.")
            return self.patch_weights.get()
        return self.patch_weights.get().reshape(self.total_clauses, self.n_patches_y, self.n_patches_x)

    def transform_patchwise(
        self, encoded_X: np.ndarray[tuple[int, int, int], np.dtype[np.uint32]]
    ) -> np.ndarray[tuple[int, int, int, int], np.dtype[np.bool]]:
        N = encoded_X.shape[0]
        co_patchwise = np.zeros((N, self.total_clauses, self.n_patches), dtype=np.int8)

        X_gpu = ga.to_gpu(encoded_X.astype(np.uint32))
        packed_clauses = ga.empty((self.total_clauses, self.n_literal_chunks), dtype=np.uint32)
        n_includes = ga.empty((self.total_clauses,), dtype=np.uint32)
        clause_drop_mask = ga.to_gpu(np.zeros((self.total_clauses,), dtype=np.int8))
        clause_outputs = ga.empty((self.total_clauses * self.n_patches,), dtype=np.int8)
        self.pack_clauses(packed_clauses, n_includes)

        for i in tqdm(range(N), desc="Patchwise Transform", leave=False, dynamic_ncols=True):
            self.kernel_eval_clauses.prepared_call(
                *self.kernel_eval_clauses_launch_config,
                packed_clauses.gpudata,
                n_includes.gpudata,
                clause_drop_mask.gpudata,
                X_gpu.gpudata,
                np.int32(i),
                clause_outputs.gpudata,
            )
            self.ctx.synchronize()

            co_patchwise[i] = clause_outputs.get().reshape((self.total_clauses, self.n_patches))

        n_clause_banks = 1 if self.args.coalesced else self.args.n_classes
        return co_patchwise.astype(bool).reshape((N, n_clause_banks, self.args.n_clauses, self.n_patches))

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

    # =========================================================================
    # No-encoding methods (kernels2.cu)
    # =========================================================================

    def _init_kernels2(self):
        """Initialize kernels2.cu for no-encoding implementation."""
        self.cur_dir = os.path.dirname(os.path.abspath(__file__))
        mod_kernels2 = self._load_kernel(os.path.join(self.cur_dir, "kernels2.cu"), self.header)

        # Inference kernel
        self.kernel2_pack_clauses = mod_kernels2.get_function("pack_clauses")
        self.kernel2_infer_batch = mod_kernels2.get_function("infer_batch")

        # Fused 2-kernel training approach
        self.kernel2_eval_and_count = mod_kernels2.get_function("eval_and_count")
        self.kernel2_prob_and_update = mod_kernels2.get_function("prob_and_update")

        # Prepare kernel signatures
        self.kernel2_pack_clauses.prepare("PPPPPPP")
        self.kernel2_infer_batch.prepare("PPPPPPPPiP")
        self.kernel2_eval_and_count.prepare("PPiPPPPPPPPP")
        self.kernel2_prob_and_update.prepare("PPPPPPPiPPPP")

        # Launch configs
        self.kernel2_clauses_launch_config = self._kernel_config(self.total_clauses)

        # Allocate buffers for sparse clause representation
        n_feature_feats = self.args.patch_dim[0] * self.args.patch_dim[1] * self.args.dim[2]
        self.sparse_included_feats_pos = ga.empty((self.total_clauses, n_feature_feats), dtype=np.int32)
        self.sparse_included_feats_neg = ga.empty((self.total_clauses, n_feature_feats), dtype=np.int32)
        self.sparse_n_feats_pos = ga.empty((self.total_clauses,), dtype=np.int32)
        self.sparse_n_feats_neg = ga.empty((self.total_clauses,), dtype=np.int32)
        self.sparse_patch_ranges = ga.empty((self.total_clauses, 4), dtype=np.int32)
        self.sparse_num_includes = ga.empty((self.total_clauses,), dtype=np.uint32)

        self._kernels2_initialized = True

    def fit_epoch2(self, X: np.ndarray, targets: np.ndarray, clause_drop_p: float, batch_size: int = -1):
        """
        Memory-efficient training that computes patch matching on-the-fly.
        Does not pre-encode patches. Uses fused 2-kernel approach.

        Args:
            X: Raw input data of shape (N, HEIGHT, WIDTH, DEPTH) as int8
            targets: Target labels of shape (N, n_classes) as int8
            clause_drop_p: Probability of dropping a clause
            batch_size: Number of samples to process per batch. -1 means all at once.
        """
        # Lazy initialization of kernels2
        if not hasattr(self, "_kernels2_initialized"):
            self._init_kernels2()

        N = X.shape[0]
        if batch_size == -1:
            batch_size = N

        # Generate clause drop mask
        if clause_drop_p > 0.0:
            clause_drop_mask = (self.np_rng.random(self.total_clauses) <= clause_drop_p).astype(np.int8)
        else:
            clause_drop_mask = np.zeros(self.total_clauses, dtype=np.int8)

        clause_drop_mask_gpu = ga.to_gpu(clause_drop_mask)

        # Allocate GPU buffers for training
        selected_patch_ids = ga.empty((self.total_clauses,), dtype=np.int32)
        pos_votes = ga.empty((self.args.n_classes,), dtype=np.float32)
        neg_votes = ga.empty((self.args.n_classes,), dtype=np.float32)

        # Process in batches
        pbar = tqdm(range(0, N, batch_size), desc="Fitting (no-enc)", leave=False, dynamic_ncols=True)
        for batch_start in pbar:
            batch_end = min(batch_start + batch_size, N)
            batch_N = batch_end - batch_start

            # Upload entire batch to GPU once
            X_batch_gpu = ga.to_gpu(X[batch_start:batch_end].astype(np.int8).reshape(batch_N, -1))
            targets_batch_gpu = ga.to_gpu(targets[batch_start:batch_end].astype(np.float32))

            # Pre-compute skip mask to avoid np.all() in hot loop
            skip_mask = np.all(targets[batch_start:batch_end] == 0, axis=1)

            # Process each sample in the batch sequentially (training requires sequential updates)
            # Using fused 2-kernel approach: eval_and_count -> prob_and_update
            for e in range(batch_N):
                # Skip if all targets are zero
                if skip_mask[e]:
                    continue

                # Zero votes using async memset (will complete before kernel reads them)
                memset_d32_async(pos_votes.gpudata, 0, self.args.n_classes)
                memset_d32_async(neg_votes.gpudata, 0, self.args.n_classes)

                # Kernel 1: Evaluate clauses and count votes (fused)
                self.kernel2_eval_and_count.prepared_call(
                    *self.kernel2_clauses_launch_config,
                    self.rng.state,
                    X_batch_gpu.gpudata,
                    np.int32(e),
                    self.ta_states.gpudata,
                    clause_drop_mask_gpu.gpudata,
                    self.clause_weights.gpudata,
                    selected_patch_ids.gpudata,
                    self.sparse_patch_ranges.gpudata,
                    self.sparse_num_includes.gpudata,
                    self.patch_weights.gpudata,
                    pos_votes.gpudata,
                    neg_votes.gpudata,
                )

                # Kernel 2: Calculate probabilities and update clauses (fused)
                self.kernel2_prob_and_update.prepared_call(
                    *self.kernel2_clauses_launch_config,
                    self.rng.state,
                    selected_patch_ids.gpudata,
                    self.sparse_patch_ranges.gpudata,
                    self.sparse_num_includes.gpudata,
                    clause_drop_mask_gpu.gpudata,
                    X_batch_gpu.gpudata,
                    targets_batch_gpu.gpudata,
                    np.int32(e),
                    pos_votes.gpudata,
                    neg_votes.gpudata,
                    self.ta_states.gpudata,
                    self.clause_weights.gpudata,
                )

            # Sync once at end of batch
            self.ctx.synchronize()

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
        # Lazy initialization of kernels2
        if not hasattr(self, "_kernels2_initialized"):
            self._init_kernels2()

        N = X.shape[0]
        if batch_size == -1:
            batch_size = N

        # Pack clauses once before inference - sparse representation is reused for all samples
        self.kernel2_pack_clauses.prepared_call(
            *self.kernel2_clauses_launch_config,
            self.ta_states.gpudata,
            self.sparse_included_feats_pos.gpudata,
            self.sparse_included_feats_neg.gpudata,
            self.sparse_n_feats_pos.gpudata,
            self.sparse_n_feats_neg.gpudata,
            self.sparse_patch_ranges.gpudata,
            self.sparse_num_includes.gpudata,
        )
        self.ctx.synchronize()

        class_sums = np.zeros((N, self.args.n_classes), dtype=np.float32)

        for i in tqdm(range(0, N, batch_size), desc="Inference (no-enc)", leave=False, dynamic_ncols=True):
            batch_end = min(i + batch_size, N)
            batch_N = batch_end - i

            # Upload batch to GPU (flatten each sample)
            X_batch = ga.to_gpu(X[i:batch_end].astype(np.int8).reshape(batch_N, -1))
            cs_batch = ga.to_gpu(np.zeros((batch_N, self.args.n_classes), dtype=np.float32))

            self.kernel2_infer_batch.prepared_call(
                *self._kernel_config(batch_N * self.total_clauses),
                X_batch.gpudata,
                self.clause_weights.gpudata,
                self.sparse_included_feats_pos.gpudata,
                self.sparse_included_feats_neg.gpudata,
                self.sparse_n_feats_pos.gpudata,
                self.sparse_n_feats_neg.gpudata,
                self.sparse_patch_ranges.gpudata,
                self.sparse_num_includes.gpudata,
                np.int32(batch_N),
                cs_batch.gpudata,
            )
            self.ctx.synchronize()

            class_sums[i:batch_end] = cs_batch.get()

        return class_sums
