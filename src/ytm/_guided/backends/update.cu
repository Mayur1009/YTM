#ifdef IS_NEOVIM_CLANGD_ENV
#include "../../_core/backends/common.h"
#include "../../_core/backends/cuda.h"
#include "../../_core/backends/feedback.cu"
#include "../../_core/backends/feedback.h"
#include "../../_core/backends/rng.h"
#include "act.h"

#define FB_NONE 0
#define FB_T1A 1
#define FB_T1B 2
#define FB_T2 3

#define FB_SIGNAL_GRAD 0
#define FB_SIGNAL_DELTA_L 1
#define FB_SIGNAL FB_SIGNAL_DELTA_L

INLINE_FN void compute_loss(const float* RESTRICT y, const float* RESTRICT y_hat, float* RESTRICT grad,
                            float* RESTRICT loss) {}
#endif

// compute_act and compute_loss fold over CLASSES sequentially, so one thread owns a whole sample.
extern "C" __global__ void votes_activation(const float* votes, float* y_hat) {
    if (threadIdx.x == 0 && blockIdx.x == 0)
        compute_act(votes, y_hat);
}

extern "C" __global__ void votes_activation_batch(const float* votes, int n_samples, float* y_hat) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int e = tid; e < n_samples; e += stride)
        compute_act(&votes[(ull)e * CLASSES], &y_hat[(ull)e * CLASSES]);
}

extern "C" __global__ void loss_gradient(const float* y_hat, const float* y, float* grad, float* loss) {
    if (threadIdx.x == 0 && blockIdx.x == 0)
        compute_loss(y, y_hat, grad, loss);
}

extern "C" __global__ void compute_votes_neg_ck(const float* votes, const float* clause_weights,
                                                const int* selected_patch_ids, float* votes_neg_ck) {
    ull tid = threadIdx.x + blockIdx.x * blockDim.x;
    ull stride = (ull)blockDim.x * gridDim.x;

    for (ull clause_class = tid; clause_class < (ull)TOTAL_CLAUSES * (ull)CLASSES; clause_class += stride) {
        ull clause_id = clause_class / (ull)CLASSES;
        ull class_id = clause_class % (ull)CLASSES;

#if COALESCED == 0
        if (class_id != clause_id / (ull)CLAUSES_PER_CLASS) {
            votes_neg_ck[clause_class] = votes[class_id];
            continue;
        }
#endif
        int ck = (selected_patch_ids[clause_id] >= 0) ? 1 : 0;
        ull rel_clause = clause_id % (ull)CLAUSES_PER_CLASS;
        float w = clause_weights[class_id * (ull)CLAUSES_PER_CLASS + rel_clause];
        votes_neg_ck[clause_class] = votes[class_id] - (ck ? w : -w);
    }
}

extern "C" __global__ void compute_loss_neg_ck(const float* y_hat_neg_ck, const float* y, float* loss_neg_ck) {
    ull tid = threadIdx.x + blockIdx.x * blockDim.x;
    ull stride = (ull)blockDim.x * gridDim.x;

    for (ull clause_id = tid; clause_id < (ull)TOTAL_CLAUSES; clause_id += stride) {
        float loss = 0.0f;
        compute_loss(y, &y_hat_neg_ck[clause_id * (ull)CLASSES], nullptr, &loss);
        loss_neg_ck[clause_id] = loss;
    }
}

extern "C" __global__ void decide_feedback_grad(const ull seed, const float* grad, const float* clause_weights,
                                                const int* clause_density, const int* selected_patch_ids,
                                                const int8_t* clause_drop_mask, const float lambda_,
                                                uint8_t* feedback_type) {
    ull tid = threadIdx.x + blockIdx.x * blockDim.x;
    ull stride = (ull)blockDim.x * gridDim.x;

    for (ull clause = tid; clause < (ull)TOTAL_CLAUSES; clause += stride) {
        ull rel_clause = clause % (ull)CLAUSES_PER_CLASS;
        int ck = (selected_patch_ids[clause] >= 0) ? 1 : 0;
        bool has_space = (clause_density[clause] <= (int)MAX_INCLUDED_LITERALS);
        bool dropped = (clause_drop_mask[clause] == 1);

        ull rng_k = rng_hash(seed, clause, 0xFEEDFACEULL);
        uint rng_counter = 0;

        ull class_id;
        LOOP_CLASS_ID(class_id, clause) {
            uint8_t fb = FB_NONE;

            if (!dropped) {
                float w = clause_weights[class_id * (ull)CLAUSES_PER_CLASS + rel_clause];
                float g = grad[class_id];

                int polarity = (w >= 0) - (w < 0);
                int sign_grad = (g >= 0) - (g < 0);
                int dir = polarity * sign_grad;
                float update_prob = 1.0f - expf(-lambda_ * fabsf(g));

                if (rand_uniform(rng_k, &rng_counter) <= update_prob) {
                    bool t1a = (ck == 1 && dir > 0 && has_space);
                    bool t1b = ((ck == 0 && dir > 0) || (ck == 1 && dir > 0 && !has_space));
                    bool t2 = (ck == 1 && dir < 0);
                    fb = t1a ? FB_T1A : (t1b ? FB_T1B : (t2 ? FB_T2 : FB_NONE));
                }
            }
            feedback_type[rel_clause * (ull)CLASSES + class_id] = fb;
        }
    }
}

__device__ inline uint8_t select_fb_delta_l(const ull seed, ull rng_id, float uprob, float delta_L, int ck,
                                            bool has_space) {
    ull rng_k = rng_hash(seed, rng_id, 0xFEEDFACEULL);
    uint rng_counter = 0;

    uint8_t fb;
    if (rand_uniform(rng_k, &rng_counter) > uprob) {
        fb = FB_NONE;
    } else {
        // T1a -> ck = 1, and deltaL < 0, meaning turning ck=0 increased the loss
        // T1b -> ck = 0, and deltaL > 0, meaning turning ck=1 decreased the loss
        // T2 -> ck = 1, and deltaL > 0, meaning turning ck=0 decreased the loss
        // None -> ck = 0, and deltaL < 0, meaning turning ck = 1 increased the loss.
        bool t1a = (ck == 1 && delta_L < 0 && has_space);
        bool t1b = ((ck == 0 && delta_L > 0) || (ck == 1 && delta_L < 0 && !has_space));
        bool t2 = (ck == 1 && delta_L > 0);
        fb = t1a ? FB_T1A : (t1b ? FB_T1B : (t2 ? FB_T2 : FB_NONE));
    }
    return fb;
}

extern "C" __global__ void decide_feedback_delta_l(const ull seed, const float* loss, const float* loss_neg_ck,
                                                   const int* clause_density, const int* selected_patch_ids,
                                                   const int8_t* clause_drop_mask, const float lambda_,
                                                   uint8_t* feedback_type) {
    ull tid = threadIdx.x + blockIdx.x * blockDim.x;
    ull stride = (ull)blockDim.x * gridDim.x;

    for (ull clause_id = tid; clause_id < (ull)TOTAL_CLAUSES; clause_id += stride) {
        if (clause_drop_mask[clause_id] == 1) {
            feedback_type[clause_id] = FB_NONE;
            continue;
        }

        int ck = (selected_patch_ids[clause_id] >= 0) ? 1 : 0;
        bool has_space = (clause_density[clause_id] <= (int)MAX_INCLUDED_LITERALS);
        float delta_L = *loss - loss_neg_ck[clause_id];
        float update_prob = 1.0f - expf(-lambda_ * fabsf(delta_L));
        feedback_type[clause_id] = select_fb_delta_l(seed, clause_id, update_prob, delta_L, ck, has_space);
    }
}

// A whole warp owns one clause, so `fb` is warp uniform and `apply_feedback` can stride its lanes
// over the literals.
extern "C" __global__ void update_clauses(const ull seed, const int* selected_patch_ids, const int* X, const int e,
                                          const int* feat_mins, const int* literal_offsets,
                                          const uint8_t* feedback_type, TA_STATE_T* global_ta_states,
                                          int8_t* is_clause_synced) {
    auto warp = cg::tiled_partition<WARP_SIZE>(cg::this_thread_block());
    auto grid = cg::this_grid();
    int lane = warp.thread_rank();
    ull warp_id = grid.thread_rank() / warp.size();
    ull total_warps = grid.size() / warp.size();

    const int* Xe = &X[(ull)e * HEIGHT * WIDTH * DEPTH];

    for (ull clause = warp_id; clause < (ull)TOTAL_CLAUSES; clause += total_warps) {
        TA_STATE_T* ta_states = &global_ta_states[clause * (ull)N_LITERALS];

        int patch_id = selected_patch_ids[clause];
        int patch_idx_y = (patch_id >= 0) ? patch_id / N_PATCHES_X : -1;
        int patch_idx_x = (patch_id >= 0) ? patch_id % N_PATCHES_X : -1;

        ull rng_k = rng_hash(seed, clause * (ull)WARP_SIZE + (ull)lane, 0xCAFEBABEULL);
        uint rng_counter = 0;

#if FB_SIGNAL == FB_SIGNAL_GRAD
        ull rel_clause = clause % (ull)CLAUSES_PER_CLASS;
        ull class_id;
        LOOP_CLASS_ID(class_id, clause) {
            uint8_t fb = feedback_type[rel_clause * (ull)CLASSES + class_id];
            if (fb == FB_NONE)
                continue;
            apply_feedback(rng_k, &rng_counter, fb, Xe, patch_idx_y, patch_idx_x, feat_mins, literal_offsets,
                           ta_states, lane);
            if (lane == 0)
                is_clause_synced[clause] = 0;
        }
#else
        uint8_t fb = feedback_type[clause];
        if (fb != FB_NONE) {
            apply_feedback(rng_k, &rng_counter, fb, Xe, patch_idx_y, patch_idx_x, feat_mins, literal_offsets,
                           ta_states, lane);
            if (lane == 0)
                is_clause_synced[clause] = 0;
        }
#endif
    }
}

extern "C" __global__ void update_weights(const float* grad, const float lr, const int* selected_patch_ids,
                                          const int8_t* clause_drop_mask, float* clause_weights) {
    ull tid = threadIdx.x + blockIdx.x * blockDim.x;
    ull stride = (ull)blockDim.x * gridDim.x;

    for (ull clause = tid; clause < (ull)TOTAL_CLAUSES; clause += stride) {
        if (clause_drop_mask[clause] == 1 || selected_patch_ids[clause] < 0)
            continue;

        ull rel_clause = clause % (ull)CLAUSES_PER_CLASS;
        ull class_id;
        LOOP_CLASS_ID(class_id, clause) {
            ull idx = class_id * (ull)CLAUSES_PER_CLASS + rel_clause;
            float new_w = clause_weights[idx] + lr * grad[class_id];
#if ALLOW_POLARITY_CHANGE
            clause_weights[idx] = clip(new_w, -MAX_WEIGHT, MAX_WEIGHT);
#else
            clause_weights[idx] =
                (clause_weights[idx] >= 0) ? clip(new_w, 0.0001f, MAX_WEIGHT) : clip(new_w, -MAX_WEIGHT, -0.0001f);
#endif
        }
    }
}
