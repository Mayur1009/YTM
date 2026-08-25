import os

import cupy as cp
import numpy as np

from ..base import BaseDevice, tqdm_bar
from .activations import sigmoid, softmax
from .losses import build_asl, build_ce, build_huber, build_mae, build_mse, build_sce, build_tversky


def read_file(path):
    with open(path, "r") as f:
        return f.read()


class CUDADevice(BaseDevice):
    def _init_kernels(self):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        header = f"""
{self._build_header()}
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
        self.k_count_votes = eval_mod.get_function("count_votes")

        update_mod = cp.RawModule(
            code=common + read_file(os.path.join(cur_dir, "update.cu")),
            options=("--use_fast_math",),
        )
        self.k_update_clauses = update_mod.get_function("update_clauses")
        self.k_decide_feedback = update_mod.get_function("decide_feedback_and_update_weights")
        self.k_update_bias = update_mod.get_function("update_bias")

        infer_mod = cp.RawModule(
            code=common + read_file(os.path.join(cur_dir, "inference.cu")),
            options=("--use_fast_math",),
        )
        self.k_eval_clauses = infer_mod.get_function("infer_clauses")
        self.k_sum_votes = infer_mod.get_function("sum_votes")
        self.k_transform_patchwise = infer_mod.get_function("infer_clauses_patchwise")

        self.kconf_clauses = self._kernel_config(self.total_clauses * self.cuda_props["warp_size"])
        self.kconf_decide = self._kernel_config(self.total_clauses)
        self.kconf_classes = self._kernel_config(self.args.n_classes * self.cuda_props["warp_size"])
        self.kconf_bias = self._kernel_config(self.args.n_classes if self.args.bias else 1)

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
        self._softmax = softmax
        self._expit = sigmoid

        self.cuda_dev = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(self.cuda_dev.id)
        self.cuda_props = {
            "max_threads_per_block": props["maxThreadsPerBlock"],
            "multiprocessor_count": props["multiProcessorCount"],
            "warp_size": props["warpSize"],
        }

        self._init_clauses()
        self._init_weights()
        self._init_bias()
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

    def _init_loss_fn(self):
        if isinstance(self.args.loss_fn, str):
            lw = cp.asarray(
                self.args.loss_fn_kwargs.get("class_weights", np.ones(self.args.n_classes)),
                dtype=np.float32,
            )

            if self.args.loss_fn == "ce":
                gamma = self.args.loss_fn_kwargs.get("gamma", 0.0)
                eps = self.args.loss_fn_kwargs.get("eps", 1e-7)
                _loss_fn, _grad_fn = build_ce(lw, gamma, eps, self.args.act_fn)
            elif self.args.loss_fn == "mse":
                _loss_fn, _grad_fn = build_mse(lw, self.dact_fn)
            elif self.args.loss_fn == "mae":
                _loss_fn, _grad_fn = build_mae(lw, self.dact_fn)
            elif self.args.loss_fn == "sce":
                alpha = self.args.loss_fn_kwargs.get("alpha", 1.0)
                beta = self.args.loss_fn_kwargs.get("beta", 1.0)
                eps = self.args.loss_fn_kwargs.get("eps", 1e-4)
                _loss_fn, _grad_fn = build_sce(lw, alpha, beta, eps, self.args.act_fn)
            elif self.args.loss_fn == "asl":
                gamma_pos = self.args.loss_fn_kwargs.get("gamma_pos", 0.0)
                gamma_neg = self.args.loss_fn_kwargs.get("gamma_neg", 4.0)
                clip = self.args.loss_fn_kwargs.get("clip", 0.05)
                eps = self.args.loss_fn_kwargs.get("eps", 1e-6)
                _loss_fn, _grad_fn = build_asl(gamma_pos, gamma_neg, clip, eps)
            elif self.args.loss_fn == "tversky":
                alpha = self.args.loss_fn_kwargs.get("alpha", 0.5)
                beta = self.args.loss_fn_kwargs.get("beta", 0.5)
                gamma = self.args.loss_fn_kwargs.get("gamma", 1.0)
                eps = self.args.loss_fn_kwargs.get("eps", 1e-6)
                _loss_fn, _grad_fn = build_tversky(lw, alpha, beta, gamma, eps, self.dact_fn)
            elif self.args.loss_fn == "huber":
                delta = self.args.loss_fn_kwargs.get("delta", 1.0)
                _loss_fn, _grad_fn = build_huber(lw, delta, self.dact_fn)
            else:
                raise NotImplementedError(f"loss_fn '{self.args.loss_fn}' not implemented")

            self.loss_fn = _loss_fn
            self.grad_fn = _grad_fn

        elif callable(self.args.loss_fn):
            self.loss_fn = self.args.loss_fn
        else:
            raise NotImplementedError(f"loss_fn '{self.args.loss_fn}' not implemented")

    def set_threads(self, n: int):
        raise RuntimeError("set_nthreads is only supported for CPU device")

    def fit_epoch(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        clause_drop_p: float,
        batch_size: int,
        rng_state: int,
        lr: float | None = None,
        loss_poll_rate: float = 0.1,
    ):
        N = X.shape[0]
        loss_poll_interval = max(1, int(N * loss_poll_rate))
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
        votes = cp.empty(self.args.n_classes, dtype=np.float32)
        grad = cp.empty(self.args.n_classes, dtype=np.float32)
        running_loss = 0.0
        n_polls = 0

        pbar = tqdm_bar(None, desc="Fit", total=N)
        for i in range(0, N, batch_size):
            batch_end = min(i + batch_size, N)
            X_batch = cp.asarray(X[i:batch_end], dtype=np.int32)
            Y_batch = cp.asarray(Y[i:batch_end], dtype=np.float32)
            bs = batch_end - i

            for e in range(bs):
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
                        np.int32(i + e),
                        clause_drop_mask_gpu,
                        self.packed_clauses.clause_position_bounds,
                        self.packed_clauses.clause_feat_bounds,
                        self.packed_clauses.bounded_feat_ids,
                        self.packed_clauses.n_bounded_feats,
                        self.packed_clauses.clause_density,
                        np.uint64(rng_state),
                        selected_patch_ids,
                        self.patch_weights,
                    ),
                )

                self.k_count_votes(
                    *self.kconf_classes,
                    (selected_patch_ids, self.clause_weights, self.bias, votes),
                )

                y_hat = self.act_fn(votes)
                self.grad_fn(Y_batch[e], y_hat, grad)

                self.k_decide_feedback(
                    *self.kconf_decide,
                    (
                        np.uint64(rng_state),
                        np.int32(i + e),
                        grad,
                        np.float32(_lr),
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
                self.k_update_bias(
                    *self.kconf_bias,
                    (grad, np.float32(_lr), self.bias),
                )
                self.k_update_clauses(
                    *self.kconf_clauses,
                    (
                        np.uint64(rng_state),
                        selected_patch_ids,
                        X_batch,
                        np.int32(e),
                        np.int32(i + e),
                        self.ta_states,
                        self.feat_mins_gpu,
                        self.literal_offsets_gpu,
                        self.feedback_type,
                    ),
                )

                if (i + e) % loss_poll_interval == 0 or (i + e) == N - 1:
                    running_loss += self.loss_fn(Y_batch[e], y_hat)
                    n_polls += 1
                    pbar.set_postfix(loss=f"{running_loss / n_polls:.4f}")

                pbar.update(1)
        pbar.close()

        return running_loss / n_polls

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
                    self.bias,
                    class_sums[i:batch_end],
                    np.int32(bs),
                ),
            )

        return self.act_fn(class_sums).astype(cp.float32).get()

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
