import os

import cupy as cp
import numpy as np

from ..base import BaseDevice, tqdm_bar


def read_file(path):
    with open(path, "r") as f:
        return f.read()


class CUDADevice(BaseDevice):
    def dev_init(self):
        self.xp = cp

        if self.args.device == "cuda":
            dev_id = 0
        else:
            dev_id = max(0, int(self.args.device[5:]))

        self.cuda_dev = cp.cuda.Device(dev_id)

        props = cp.cuda.runtime.getDeviceProperties(self.cuda_dev.id)
        self.cuda_props = {
            "max_threads_per_block": props["maxThreadsPerBlock"],
            "multiprocessor_count": props["multiProcessorCount"],
            "warp_size": props["warpSize"],
        }

        with self.cuda_dev:
            self._init_clauses()
            self._init_weights()
            self._init_bias()
            self._init_packed_clauses()
            self._init_frozen_clauses()
            self._init_kernels()
            self.feat_mins_gpu = cp.asarray(self.args.feat_mins, dtype=np.int32)
            self.feat_maxs_gpu = cp.asarray(self.args.feat_maxs, dtype=np.int32)
            self.literal_offsets_gpu = cp.asarray(self.literal_offsets.astype(np.int32))
            self._init_loss_fn()

    def _init_kernels(self):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        header = f"""
{self._build_header()}
#define WARPS_PER_CLAUSE {self.args.warps_per_clause}
"""
        common = header + "\n" + read_file(os.path.join(cur_dir, "common.cu")) + "\n"

        pack_mod = cp.RawModule(
            code=common + read_file(os.path.join(cur_dir, "pack_clauses.cu")),
            options=(),
        )
        self.k_pack_clauses = pack_mod.get_function("pack_clauses")

        eval_mod = cp.RawModule(
            code=common + read_file(os.path.join(cur_dir, "evaluate.cu")),
            options=(),
        )
        self.k_evaluate = eval_mod.get_function("evaluate")
        self.k_count_votes = eval_mod.get_function("count_votes")

        update_mod = cp.RawModule(
            code=common + read_file(os.path.join(cur_dir, "losses.cu")) + read_file(os.path.join(cur_dir, "update.cu")),
            options=(),
        )
        self.k_decide_feedback_grad = update_mod.get_function("decide_feedback_grad")
        self.k_decide_feedback_delta_l = update_mod.get_function("decide_feedback_delta_l")
        self.k_compute_votes_neg_ck = update_mod.get_function("compute_votes_neg_ck")
        self.k_compute_loss_neg_ck = update_mod.get_function("compute_loss_neg_ck")

        self.k_update_clauses = update_mod.get_function("update_clauses")
        self.k_update_weights = update_mod.get_function("update_weights")
        self.k_update_bias = update_mod.get_function("update_bias")

        infer_mod = cp.RawModule(
            code=common + read_file(os.path.join(cur_dir, "inference.cu")),
            options=(),
        )
        self.k_eval_clauses = infer_mod.get_function("infer_clauses")
        self.k_sum_votes = infer_mod.get_function("sum_votes")
        self.k_transform_patchwise = infer_mod.get_function("infer_clauses_patchwise")

        losses_mod = cp.RawModule(
            code=common + read_file(os.path.join(cur_dir, "losses.cu")),
            options=(),
        )
        self.k_votes_activation = losses_mod.get_function("votes_activation")
        self.k_votes_activation_batch = losses_mod.get_function("votes_activation_batch")
        self.k_loss_gradient = losses_mod.get_function("loss_gradient")

        self.kconf_serial = ((1, 1, 1), (1, 1, 1))
        self.kconf_clauses_warp = self._kernel_config(self.total_clauses * self.cuda_props["warp_size"])
        self.kconf_classes_warp = self._kernel_config(self.args.n_classes * self.cuda_props["warp_size"])

        self.kconf_clauses = self._kernel_config(self.total_clauses)
        self.kconf_clauses_classes = self._kernel_config(self.total_clauses * self.args.n_classes)
        self.kconf_classes = self._kernel_config(self.args.n_classes)

        self.kconf_votes_activation = self.kconf_serial if self.args.act_fn == "softmax" else self.kconf_classes
        self.kconf_delta_l_y_hat = self.kconf_clauses if self.args.act_fn == "softmax" else self.kconf_clauses_classes

        self.kconf_update_clauses = self._kernel_config(self.total_clauses * self.args.warps_per_clause * self.cuda_props["warp_size"])

    def _kernel_config(self, n) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        bs = min(self.args.block_size, self.cuda_props["max_threads_per_block"])
        if self.args.grid_size is None:
            max_gs = self.cuda_props["multiprocessor_count"] * 32
            gs = min((n + bs - 1) // bs, max_gs)
        else:
            gs = self.args.grid_size
        return (gs, 1, 1), (bs, 1, 1)

    def _to_host(self, arr):
        with self.cuda_dev:
            return arr.get()

    def set_threads(self, n: int):
        raise RuntimeError("set_nthreads is only supported for CPU device")

    # -- Public API -----------------------------------------------------

    def fit_epoch(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        clause_drop_p: float,
        batch_size: int,
        lr: float | None = None,
        lambda_: float | None = None,
    ):
        with self.cuda_dev:
            return self._fit_epoch_impl(X, Y, clause_drop_p, batch_size, lr, lambda_)

    def pack_clauses(self, force_repack: bool = False):
        with self.cuda_dev:
            return self._pack_clauses_impl(force_repack)

    def infer(self, X: np.ndarray, batch_size: int):
        with self.cuda_dev:
            return self._infer_impl(X, batch_size)

    def transform(self, X: np.ndarray, batch_size: int):
        with self.cuda_dev:
            return self._transform_impl(X, batch_size)

    def transform_patchwise(self, X: np.ndarray, batch_size: int):
        with self.cuda_dev:
            return self._transform_patchwise_impl(X, batch_size)

    def load_state_dict(self, state_dict):
        with self.cuda_dev:
            super().load_state_dict(state_dict)

    # -- Impls ------------------------------------------------------------

    def _fit_epoch_impl(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        clause_drop_p: float,
        batch_size: int,
        lr: float | None,
        lambda_: float | None,
    ):
        N = X.shape[0]
        if batch_size == -1:
            batch_size = N

        _lr = lr if lr is not None else self.args.lr
        _lambda = lambda_ if lambda_ is not None else self.args.lambda_

        # Clause dropout mask (same for entire epoch)
        if clause_drop_p > 0.0:
            clause_drop_mask_gpu = cp.asarray(self._rng.random(self.total_clauses) <= clause_drop_p, dtype=cp.int8)
        else:
            clause_drop_mask_gpu = cp.zeros(self.total_clauses, dtype=np.int8)

        clause_drop_mask_gpu = cp.logical_or(clause_drop_mask_gpu, self.frozen_clauses.flatten()).astype(cp.int8)

        selected_patch_ids = cp.empty(self.total_clauses, dtype=np.int32)
        votes = cp.empty(self.args.n_classes, dtype=np.float32)
        y_hat = cp.empty(self.args.n_classes, dtype=np.float32)
        grad = cp.empty(self.args.n_classes, dtype=np.float32)
        loss = cp.empty(1, dtype=np.float32)

        if self.args.fb_signal == "grad":
            feedback_type = cp.zeros((self.total_clauses, self.args.n_classes), dtype=cp.uint8)
        elif self.args.fb_signal == "delta_l":
            feedback_type = cp.zeros((self.total_clauses), dtype=cp.uint8)
            votes_neg_ck = cp.empty((self.total_clauses, self.args.n_classes), dtype=np.float32)
            y_hat_neg_ck = cp.empty((self.total_clauses, self.args.n_classes), dtype=np.float32)
            loss_neg_ck = cp.empty((self.total_clauses), dtype=np.float32)

        running_loss_val = cp.zeros(1, dtype=np.float32)

        pbar = tqdm_bar(None, desc="Fit", total=N)
        for i in range(0, N, batch_size):
            batch_end = min(i + batch_size, N)
            X_batch = cp.asarray(X[i:batch_end], dtype=np.int32)
            Y_batch = cp.asarray(Y[i:batch_end], dtype=np.float32)
            bs = batch_end - i

            for e in range(bs):
                _rng_key = np.uint64(self._rng.integers(1, 1 << 63, dtype=np.uint64))

                self.pack_clauses()
                self.k_evaluate(
                    *self.kconf_clauses_warp,
                    (
                        _rng_key,
                        X_batch,
                        np.int32(e),
                        clause_drop_mask_gpu,
                        self.packed_clauses.clause_position_bounds,
                        self.packed_clauses.clause_feat_bounds,
                        self.packed_clauses.bounded_feat_ids,
                        self.packed_clauses.n_bounded_feats,
                        self.packed_clauses.clause_density,
                        selected_patch_ids,
                        self.patch_weights,
                    ),
                )

                self.k_count_votes(
                    *self.kconf_classes_warp,
                    (selected_patch_ids, self.clause_weights, self.bias, votes),
                )

                self.k_votes_activation(
                    *self.kconf_votes_activation,
                    (votes, y_hat),
                )
                self.k_loss_gradient(
                    *self.kconf_serial,
                    (y_hat, Y_batch[e], self._loss_class_weights, grad, loss),
                )

                if self.args.fb_signal == "grad":
                    self.k_decide_feedback_grad(
                        *self.kconf_clauses_classes,
                        (
                            _rng_key,
                            grad,
                            self.clause_weights,
                            self.packed_clauses.clause_density,
                            selected_patch_ids,
                            clause_drop_mask_gpu,
                            np.float32(_lambda),
                            feedback_type,
                        ),
                    )
                elif self.args.fb_signal == "delta_l":
                    self.k_compute_votes_neg_ck(
                        *self.kconf_clauses_classes,
                        (votes, self.clause_weights, selected_patch_ids, votes_neg_ck),
                    )
                    self.k_votes_activation_batch(
                        *self.kconf_delta_l_y_hat,
                        (votes_neg_ck, np.int32(self.total_clauses), y_hat_neg_ck),
                    )
                    self.k_compute_loss_neg_ck(
                        *self.kconf_clauses,
                        (y_hat_neg_ck, Y_batch[e], self._loss_class_weights, loss_neg_ck),
                    )
                    self.k_decide_feedback_delta_l(
                        *self.kconf_clauses,
                        (
                            _rng_key,
                            loss,
                            loss_neg_ck,
                            self.packed_clauses.clause_density,
                            selected_patch_ids,
                            clause_drop_mask_gpu,
                            np.float32(_lambda),
                            feedback_type,
                        ),
                    )

                self.k_update_clauses(
                    *self.kconf_update_clauses,
                    (
                        _rng_key,
                        selected_patch_ids,
                        X_batch,
                        np.int32(e),
                        self.feat_mins_gpu,
                        self.literal_offsets_gpu,
                        feedback_type,
                        self.ta_states,
                        self.packed_clauses.is_clause_synced,
                    ),
                )

                self.k_update_weights(
                    *self.kconf_clauses,
                    (grad, np.float32(_lr), selected_patch_ids, clause_drop_mask_gpu, self.clause_weights),
                )

                if self.args.bias:
                    self.k_update_bias(
                        *self.kconf_classes,
                        (grad, np.float32(_lr), self.bias),
                    )

                running_loss_val += loss
                pbar.update(1)
        pbar.close()

        result = float(running_loss_val[0]) / N
        del clause_drop_mask_gpu, selected_patch_ids, votes, grad, y_hat, loss, running_loss_val, feedback_type
        if self.args.fb_signal == "delta_l":
            del votes_neg_ck, y_hat_neg_ck, loss_neg_ck
        del X_batch, Y_batch
        return result

    def _pack_clauses_impl(self, force_repack: bool = False):
        if force_repack:
            self.packed_clauses.is_clause_synced.fill(0)

        self.k_pack_clauses(
            *self.kconf_clauses_warp,
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

    def _infer_impl(self, X: np.ndarray, batch_size: int):
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
                    self.bias,
                    class_sums[i:batch_end],
                    np.int32(bs),
                ),
            )

        y_hat = cp.empty((N, self.args.n_classes), dtype=np.float32)
        act_kconf = self._kernel_config(N) if self.args.act_fn == "softmax" else self._kernel_config(N * self.args.n_classes)
        self.k_votes_activation_batch(
            *act_kconf,
            (class_sums, np.int32(N), y_hat),
        )
        result = y_hat.get()
        del class_sums, batch_X, co_batch, y_hat
        return result

    def _transform_impl(self, X: np.ndarray, batch_size: int):
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

        del batch_X, co_batch
        return clause_outputs

    def _transform_patchwise_impl(self, X: np.ndarray, batch_size: int):
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

        result = patch_output.reshape((N, self.n_clause_banks, self.args.n_clauses, self.n_patches_y, self.n_patches_x))
        del batch_X, po_batch
        return result
