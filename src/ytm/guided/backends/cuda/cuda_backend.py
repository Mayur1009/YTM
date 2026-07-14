import os
import warnings

import numpy as np
import cupy as cp
from cupyx.scipy.special import expit as cp_expit, log_softmax as cp_log_softmax, softmax as cp_softmax
from ..base import BaseDevice, PackedClauses, tqdm_bar


class PackedClausesCUDA(PackedClauses):
    def get(self):
        return PackedClauses(
            clause_position_bounds=self.clause_position_bounds.get(),
            clause_feat_bounds=self.clause_feat_bounds.get(),
            bounded_feat_ids=self.bounded_feat_ids.get(),
            n_bounded_feats=self.n_bounded_feats.get(),
            clause_density=self.clause_density.get(),
            is_clause_synced=self.is_clause_synced.get(),
        )


def read_file(path):
    with open(path, "r") as f:
        return f.read()


class CUDADevice(BaseDevice):
    def _build_header(self):
        header = f"""
#define TOTAL_CLAUSES {self.total_clauses}
#define S {float(self.args.s)}f
#define CLASSES {self.args.n_classes}
#define Q {float(self.args.q)}f
#define HEIGHT {self.args.dim[0]}
#define WIDTH {self.args.dim[1]}
#define DEPTH {self.args.dim[2]}
#define PATCH_HEIGHT {self.args.patch_dim[0]}
#define PATCH_WIDTH {self.args.patch_dim[1]}
#define STRIDE_Y {self.args.stride[0]}
#define STRIDE_X {self.args.stride[1]}
#define MAX_WEIGHT {float(self.args.max_weight)}f
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
        self.k_update_clauses = update_mod.get_function("update_clauses")
        self.k_update_weights = update_mod.get_function("update_weights")

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
        self.clause_weights = cp.asarray(
            self.np_rng.uniform(-1.0, 1.0, size=(self.args.n_classes, self.args.n_clauses)),
            dtype=np.float32,
        )

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

        self.packed_clauses = PackedClausesCUDA(
            clause_position_bounds=clause_position_bounds,
            clause_feat_bounds=clause_feat_bounds,
            bounded_feat_ids=bounded_feat_ids,
            n_bounded_feats=n_bounded_feats,
            clause_density=clause_density,
            is_clause_synced=is_clause_synced,
        )

    def _init_frozen_clauses(self):
        self.frozen_clauses = cp.zeros((self.n_clause_banks, self.args.n_clauses), dtype=cp.int8)

    def _kernel_config(self, n) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        bs = min(self.args.block_size, self.cuda_props["max_threads_per_block"])
        if self.args.grid_size is None:
            gs = (n + bs - 1) // bs
        else:
            gs = self.args.grid_size
        return (gs, 1, 1), (bs, 1, 1)

    def _init_act_fn(self):
        if callable(self.args.act_fn):
            self.act_fn = self.args.act_fn
            self.dact_fn = lambda act: cp.ones_like(act)
        elif self.args.act_fn == "softmax":
            self.act_fn = lambda v: cp_softmax(v, axis=-1)
            self.dact_fn = lambda act: cp.ones_like(act)
        elif self.args.act_fn == "sigmoid":
            self.act_fn = cp_expit
            self.dact_fn = lambda act: act * (1.0 - act)
        elif self.args.act_fn == "identity":
            self.act_fn = lambda v: v
            self.dact_fn = lambda act: cp.ones_like(act)
        else:
            raise NotImplementedError(f"act_fn '{self.args.act_fn}' not implemented")

    def _init_loss_fn(self):
        if callable(self.args.loss_fn):
            _fn = self.args.loss_fn
            def _loss_fn(v, y, grad, **kwargs):
                return float(_fn(v, y, grad, **kwargs))
        else:
            lw = cp.asarray(
                self.args.loss_fn_kwargs.get("class_weights", np.ones(self.args.n_classes)),
                dtype=cp.float64,
            )
            if self.args.loss_fn == "ce":
                gamma = self.args.loss_fn_kwargs.get("gamma", 0.0)
                if self.args.act_fn == "softmax":
                    def _loss_fn(v, y, grad, **kwargs):
                        act = self.act_fn(v)
                        fw = (1.0 - float(cp.dot(y, act))) ** gamma
                        grad[:] = (lw * fw * (y - act)).astype(cp.float32)
                        return float(-cp.sum(lw * y * cp_log_softmax(v))) * fw
                else:
                    def _loss_fn(v, y, grad, **kwargs):
                        act = self.act_fn(v)
                        p_t = y * act + (1.0 - y) * (1.0 - act)
                        fw = (1.0 - p_t) ** gamma
                        grad[:] = (lw * fw * (y - act)).astype(cp.float32)
                        return float(-cp.sum(lw * fw * (y * cp.log(act + 1e-7) + (1 - y) * cp.log(1 - act + 1e-7))))
            elif self.args.loss_fn == "sce":
                alpha = self.args.loss_fn_kwargs.get("alpha", 1.0)
                beta  = self.args.loss_fn_kwargs.get("beta",  1.0)
                eps   = self.args.loss_fn_kwargs.get("eps",   1e-4)
                if self.args.act_fn == "softmax":
                    def _loss_fn(v, y, grad, **kwargs):
                        act = self.act_fn(v)
                        logy = cp.log(y + eps)
                        A = float(cp.dot(act, logy))
                        grad[:] = (lw * (alpha * (y - act) + beta * act * (logy - A))).astype(cp.float32)
                        ce  = float(-cp.sum(lw * y * cp_log_softmax(v)))
                        rce = float(-cp.sum(lw * act * logy))
                        return alpha * ce + beta * rce
                else:
                    def _loss_fn(v, y, grad, **kwargs):
                        act = self.act_fn(v)
                        bce_grad = y - act
                        rce_grad = cp.log((y + eps) / (1.0 - y + eps)) * act * (1.0 - act)
                        grad[:] = (lw * (alpha * bce_grad + beta * rce_grad)).astype(cp.float32)
                        bce = float(-cp.sum(lw * (y * cp.log(act + eps) + (1 - y) * cp.log(1 - act + eps))))
                        rce = float(-cp.sum(lw * (act * cp.log(y + eps) + (1 - act) * cp.log(1 - y + eps))))
                        return alpha * bce + beta * rce
            elif self.args.loss_fn == "mse":
                def _loss_fn(v, y, grad, **kwargs):
                    act = self.act_fn(v)
                    grad[:] = (2 * lw * (y - act) * self.dact_fn(act)).astype(cp.float32)
                    return float(cp.sum(lw * (y - act) ** 2))
            elif self.args.loss_fn == "mae":
                def _loss_fn(v, y, grad, **kwargs):
                    act = self.act_fn(v)
                    grad[:] = (lw * cp.sign(y - act) * self.dact_fn(act)).astype(cp.float32)
                    return float(cp.sum(lw * cp.abs(y - act)))
            elif self.args.loss_fn == "huber":
                delta = self.args.loss_fn_kwargs.get("delta", 1.0)
                def _loss_fn(v, y, grad, **kwargs):
                    act = self.act_fn(v)
                    r = y - act
                    grad[:] = (lw * cp.clip(r, -delta, delta) * self.dact_fn(act)).astype(cp.float32)
                    huber = cp.where(cp.abs(r) <= delta, 0.5 * r ** 2, delta * (cp.abs(r) - 0.5 * delta))
                    return float(cp.sum(lw * huber))
            elif self.args.loss_fn == "tversky":
                alpha = self.args.loss_fn_kwargs.get("alpha", 0.5)
                beta  = self.args.loss_fn_kwargs.get("beta",  0.5)
                gamma = self.args.loss_fn_kwargs.get("gamma", 1.0)
                eps   = self.args.loss_fn_kwargs.get("eps",   1e-6)
                if self.args.act_fn == "sigmoid":
                    def _loss_fn(v, y, grad, **kwargs):
                        act = self.act_fn(v)
                        tp = float(cp.dot(lw * y, act))
                        fp = float(cp.dot(lw * (1.0 - y), act))
                        fn = float(cp.dot(lw * y, 1.0 - act))
                        N = tp + eps
                        D = tp + alpha * fp + beta * fn + eps
                        T = N / D
                        fw = gamma * (1.0 - T) ** (gamma - 1.0)
                        coeff = alpha + y * (1.0 - alpha - beta)
                        grad[:] = (fw * lw * (y * D - N * coeff) / D ** 2 * act * (1.0 - act)).astype(cp.float32)
                        return float((1.0 - T) ** gamma)
                else:
                    def _loss_fn(v, y, grad, **kwargs):
                        act = self.act_fn(v)
                        tp = float(cp.dot(lw * y, act))
                        fp = float(cp.dot(lw * (1.0 - y), act))
                        fn = float(cp.dot(lw * y, 1.0 - act))
                        N = tp + eps
                        D = tp + alpha * fp + beta * fn + eps
                        T = N / D
                        fw = gamma * (1.0 - T) ** (gamma - 1.0)
                        coeff = alpha + y * (1.0 - alpha - beta)
                        grad[:] = (fw * lw * (y * D - N * coeff) / D ** 2 * self.dact_fn(act)).astype(cp.float32)
                        return float((1.0 - T) ** gamma)
            else:
                raise NotImplementedError(f"loss_fn '{self.args.loss_fn}' not implemented")
        self.loss_fn = _loss_fn

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
        self._init_frozen_clauses()
        self._init_kernels()
        self.feat_mins_gpu = cp.asarray(self.args.feat_mins, dtype=np.int32)
        self.feat_maxs_gpu = cp.asarray(self.args.feat_maxs, dtype=np.int32)
        self.literal_offsets_gpu = cp.asarray(self.literal_offsets.astype(np.int32))
        self._init_act_fn()
        self._init_loss_fn()

    def set_threads(self, n_threads: int):
        raise RuntimeError("set_nthreads is only supported for CPU device")
    def freeze_clauses(self, class_id: int, clause_ids: list[int] | np.ndarray):
        clause_ids = np.asarray(clause_ids, dtype=np.int32)
        self.frozen_clauses[class_id, clause_ids] = 1

    def unfreeze_clauses(self):
        self.frozen_clauses.fill(0)

    def fit_epoch(self, X: np.ndarray, encoded_Y: np.ndarray, clause_drop_p: float, batch_size: int, label_probs: np.ndarray, lr: float | None = None):
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
        votes = cp.zeros(self.args.n_classes, dtype=np.float32)
        grad = cp.empty(self.args.n_classes, dtype=np.float32)
        loss_per_sample = cp.zeros(N, dtype=np.float32)
        running_loss = 0.0
        sample_count = 0

        for i in tqdm_bar(range(0, N, batch_size), desc="Fit batch"):
            batch_end = min(i + batch_size, N)
            X_batch = cp.asarray(X[i:batch_end], dtype=np.int32)
            encoded_Y_batch = cp.asarray(encoded_Y[i:batch_end], dtype=np.float32)
            label_probs_batch = cp.asarray(label_probs[i:batch_end], dtype=np.float32)
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
                self.k_count_votes(
                    *self.kconf_classes,
                    (selected_patch_ids, self.clause_weights, votes),
                )
                v = (votes / self.args.n_clauses).astype(cp.float64)
                y = (encoded_Y_batch[e] > 0).astype(cp.float64)
                loss_per_sample[i + e] = self.loss_fn(v, y, grad, **self.args.loss_fn_kwargs)
                running_loss += float(loss_per_sample[i + e])
                sample_count += 1
                pbar.set_postfix(loss=f"{running_loss / sample_count:.4f}")
                self.k_update_clauses(
                    *self.kconf_clauses,
                    (
                        self.seed,
                        selected_patch_ids,
                        self.packed_clauses.clause_density,
                        clause_drop_mask_gpu,
                        X_batch,
                        encoded_Y_batch,
                        np.int32(e),
                        grad,
                        label_probs_batch,
                        self.ta_states,
                        self.clause_weights,
                        self.feat_mins_gpu,
                        self.literal_offsets_gpu,
                        self.packed_clauses.is_clause_synced,
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
        self.packed_clauses.is_clause_synced.fill(0)
