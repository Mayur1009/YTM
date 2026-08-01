import os

import numpy as np
import cupy as cp
from cupyx.scipy.special import expit as cp_expit, log_softmax as cp_log_softmax, softmax as cp_softmax
from ..base import BaseDevice, PackedClauses, tqdm_bar


def read_file(path):
    with open(path, "r") as f:
        return f.read()


class CUDADevice(BaseDevice):
    def _init_kernels(self):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        header = f"""
{self._build_header()}
#define WARPS_PER_CLAUSE {self.args.warps_per_clause}
"""
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

        update_mod = cp.RawModule(
            code=common + read_file(os.path.join(cur_dir, "update.cu")),
            options=("--use_fast_math",),
        )
        self.k_update_clauses = update_mod.get_function("update_clauses")
        self.k_update_weights = update_mod.get_function("update_weights")
        self.k_decide_feedback = update_mod.get_function("decide_feedback")

        infer_mod = cp.RawModule(
            code=common + read_file(os.path.join(cur_dir, "inference.cu")),
            options=("--use_fast_math",),
        )
        self.k_eval_clauses = infer_mod.get_function("infer_clauses")
        self.k_sum_votes = infer_mod.get_function("sum_votes")
        self.k_transform_patchwise = infer_mod.get_function("infer_clauses_patchwise")

        self.kconf_clauses = self._kernel_config(self.total_clauses * self.args.warps_per_clause * self.cuda_props["warp_size"])
        self.kconf_decide = self._kernel_config(self.total_clauses)

    def _kernel_config(self, n) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        bs = min(self.args.block_size, self.cuda_props["max_threads_per_block"])
        if self.args.grid_size is None:
            max_gs = self.cuda_props["multiprocessor_count"] * 32
            gs = min((n + bs - 1) // bs, max_gs)
        else:
            gs = self.args.grid_size
        return (gs, 1, 1), (bs, 1, 1)

    def dev_init(self):
        self.xp = cp
        self._softmax = cp_softmax
        self._expit = cp_expit
        self._log_softmax = cp_log_softmax
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
        self._init_frozen_clauses()
        self._init_kernels()
        self.feat_mins_gpu = cp.asarray(self.args.feat_mins, dtype=np.int32)
        self.feat_maxs_gpu = cp.asarray(self.args.feat_maxs, dtype=np.int32)
        self.literal_offsets_gpu = cp.asarray(self.literal_offsets.astype(np.int32))
        self.feedback_type = cp.zeros((self.total_clauses, self.args.n_classes), dtype=cp.uint8)
        self._init_act_fn()
        self._init_loss_fn()

    def _to_host(self, arr):
        return arr.get()

    def set_threads(self, n: int):
        raise RuntimeError("set_nthreads is only supported for CPU device")

    def fit_epoch(self, X: np.ndarray, Y: np.ndarray, clause_drop_p: float, batch_size: int, lr: float | None = None):
        N = X.shape[0]
        if batch_size == -1:
            batch_size = N

        _lr = lr if lr is not None else self.args.lr

        # Clause dropout mask (same for entire epoch)
        if clause_drop_p > 0.0:
            clause_drop_mask_gpu = cp.asarray(self.np_rng.random(self.total_clauses) <= clause_drop_p, dtype=cp.int8)
        else:
            clause_drop_mask_gpu = cp.zeros(self.total_clauses, dtype=np.int8)

        clause_drop_mask_gpu = cp.logical_or(clause_drop_mask_gpu, self.frozen_clauses.flatten()).astype(cp.int8)

        selected_patch_ids = cp.empty(self.total_clauses, dtype=np.int32)
        grad = cp.empty(self.args.n_classes, dtype=np.float32)
        loss_per_sample = cp.zeros(N, dtype=np.float32)
        running_loss = 0.0
        sample_count = 0

        for i in tqdm_bar(range(0, N, batch_size), desc="Fit batch"):
            batch_end = min(i + batch_size, N)
            X_batch = cp.asarray(X[i:batch_end], dtype=np.int32)
            Y_batch = cp.asarray(Y[i:batch_end], dtype=np.float32)
            bs = batch_end - i

            pbar = tqdm_bar(range(bs), desc="Sample")
            for e in pbar:
                self.k_pack_clauses(
                    *self.kconf_clauses,
                    (
                        self.ta_states,
                        self.feat_mins_gpu,
                        self.feat_maxs_gpu,
                        self.literal_offsets_gpu,
                        self.packed_clauses.clause_position_bounds,
                        self.packed_clauses.clause_feat_bounds,
                        self.packed_clauses.bounded_feat_ids,
                        self.packed_clauses.n_bounded_feats,
                        self.packed_clauses.clause_density,
                        self.packed_clauses.is_clause_synced,
                    ),
                )
                self.k_evaluate(
                    *self.kconf_clauses,
                    (
                        X_batch,
                        np.int32(e),
                        clause_drop_mask_gpu,
                        self.packed_clauses.clause_position_bounds,
                        self.packed_clauses.clause_feat_bounds,
                        self.packed_clauses.bounded_feat_ids,
                        self.packed_clauses.n_bounded_feats,
                        self.packed_clauses.clause_density,
                        self.seed,
                        selected_patch_ids,
                        self.patch_weights,
                    ),
                )

                mask = (selected_patch_ids >= 0).astype(cp.float32)
                if self.args.coalesced:
                    votes = self.clause_weights @ mask
                else:
                    votes = (self.clause_weights * mask.reshape(self.args.n_classes, -1)).sum(axis=-1)
                v = votes / self.args.n_clauses
                loss_per_sample[i + e] = self.loss_fn(v, Y_batch[e].astype(cp.float32), grad, **self.args.loss_fn_kwargs)

                running_loss += float(loss_per_sample[i + e])
                sample_count += 1
                pbar.set_postfix(loss=f"{running_loss / sample_count:.4f}")

                self.k_decide_feedback(
                    *self.kconf_decide,
                    (
                        self.seed,
                        np.int32(e),
                        grad,
                        self.clause_weights,
                        self.packed_clauses.clause_density,
                        selected_patch_ids,
                        clause_drop_mask_gpu,
                        np.float32(self.args.lambda_plus),
                        np.float32(self.args.lambda_minus),
                        self.feedback_type,
                        self.packed_clauses.is_clause_synced,
                    ),
                )
                self.k_update_clauses(
                    *self.kconf_clauses,
                    (
                        self.seed,
                        selected_patch_ids,
                        X_batch,
                        np.int32(e),
                        self.ta_states,
                        self.feat_mins_gpu,
                        self.literal_offsets_gpu,
                        self.feedback_type,
                    ),
                )
                self.k_update_weights(
                    *self._kernel_config(self.total_clauses),
                    (
                        selected_patch_ids,
                        clause_drop_mask_gpu,
                        grad,
                        np.float32(_lr),
                        self.clause_weights,
                    ),
                )
        return loss_per_sample.get()

    def pack_clauses(self, force_repack: bool = False):
        if force_repack:
            self.packed_clauses.is_clause_synced.fill(0)

        self.k_pack_clauses(
            *self.kconf_clauses,
            (
                self.ta_states,
                self.feat_mins_gpu,
                self.feat_maxs_gpu,
                self.literal_offsets_gpu,
                self.packed_clauses.clause_position_bounds,
                self.packed_clauses.clause_feat_bounds,
                self.packed_clauses.bounded_feat_ids,
                self.packed_clauses.n_bounded_feats,
                self.packed_clauses.clause_density,
                self.packed_clauses.is_clause_synced,
            ),
        )

    def infer(self, X: np.ndarray, batch_size: int):
        N = X.shape[0]
        if batch_size == -1:
            batch_size = N

        class_sums = cp.zeros((N, self.args.n_classes), dtype=np.float32)
        self.pack_clauses()

        for i in tqdm_bar(range(0, N, batch_size), desc="Infer batch"):
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
                    self.packed_clauses.clause_position_bounds,
                    self.packed_clauses.clause_feat_bounds,
                    self.packed_clauses.bounded_feat_ids,
                    self.packed_clauses.n_bounded_feats,
                    self.packed_clauses.clause_density,
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

        v = class_sums / self.args.n_clauses
        return self.act_fn(v).astype(cp.float32).get()

    def transform(self, X: np.ndarray, batch_size: int):
        N = X.shape[0]
        if batch_size == -1:
            batch_size = N

        clause_outputs = np.zeros((N, self.total_clauses), dtype=np.int8)
        self.pack_clauses()

        for i in tqdm_bar(range(0, N, batch_size), desc="Transform batch"):
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
                    self.packed_clauses.clause_position_bounds,
                    self.packed_clauses.clause_feat_bounds,
                    self.packed_clauses.bounded_feat_ids,
                    self.packed_clauses.n_bounded_feats,
                    self.packed_clauses.clause_density,
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

        for i in tqdm_bar(range(0, N, batch_size), desc="Transform batch"):
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
                    self.packed_clauses.clause_position_bounds,
                    self.packed_clauses.clause_feat_bounds,
                    self.packed_clauses.bounded_feat_ids,
                    self.packed_clauses.n_bounded_feats,
                    self.packed_clauses.clause_density,
                ),
            )
            patch_output[i:batch_end] = po_batch.get()

        return patch_output.reshape((N, self.n_clause_banks, self.args.n_clauses, self.n_patches_y, self.n_patches_x))

    def load_state_dict(self, state_dict: dict):
        self.ta_states = cp.asarray(state_dict["ta_states"])
        self.clause_weights = cp.asarray(state_dict["clause_weights"])
        self.patch_weights = cp.asarray(state_dict["patch_weights"])
        self.packed_clauses.is_clause_synced.fill(0)
