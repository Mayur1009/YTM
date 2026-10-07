#ifdef IS_NEOVIM_CLANGD_ENV
#include "../../_core/backends/common.h"
#include "../../_core/backends/cuda.h"
#include "../../_core/backends/feedback.h"
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
    GRID_STRIDE_LOOP(e, n_samples)
        compute_act(&votes[(ull)e * CLASSES], &y_hat[(ull)e * CLASSES]);
}

extern "C" __global__ void loss_gradient(const float* y_hat, const float* Y, const int* e_dev, float* grad, float* loss) {
    const float* y = &Y[(ull)(*e_dev) * CLASSES];
    if (threadIdx.x == 0 && blockIdx.x == 0)
        compute_loss(y, y_hat, grad, loss);
}

extern "C" __global__ void compute_votes_neg_ck(const float* votes, const float* clause_weights,
                                                const int8_t* clause_output, float* votes_neg_ck) {
    GRID_STRIDE_LOOP(clause_class, (ull)TOTAL_CLAUSES * (ull)CLASSES) {
        ull clause_id = clause_class / (ull)CLASSES;
        ull class_id = clause_class % (ull)CLASSES;

#if COALESCED == 0
        if (class_id != clause_id / (ull)CLAUSES_PER_CLASS) {
            votes_neg_ck[clause_class] = votes[class_id];
            continue;
        }
#endif
        int ck = clause_output[clause_id];
        float w = clause_weights[weight_offset(class_id, clause_id)];
        votes_neg_ck[clause_class] = votes[class_id] - (ck ? w : -w);
    }
}

extern "C" __global__ void compute_loss_neg_ck(const float* y_hat_neg_ck, const float* Y, const int* e_dev,
                                               float* loss_neg_ck) {
    const float* y = &Y[(ull)(*e_dev) * CLASSES];
    GRID_STRIDE_LOOP(clause_id, (ull)TOTAL_CLAUSES) {
        float loss = 0.0f;
        compute_loss(y, &y_hat_neg_ck[clause_id * (ull)CLASSES], nullptr, &loss);
        loss_neg_ck[clause_id] = loss;
    }
}

extern "C" __global__ void decide_feedback_grad(const ull* keys, const int* e_dev, const float* grad, const float* clause_weights,
                                                const NLITS_T* clause_len, const int8_t* clause_output,
                                                const int8_t* clause_drop_mask, const float lambda_,
                                                uint8_t* feedback_type, uint* t1b_n, uint* t1b_cid,
                                                NCLASSES_T* t1b_cnt, uint* t1a_t2_n, uint* t1a_t2_cid,
                                                NCLASSES_T* t1a_cnt, NCLASSES_T* t2_cnt) {
    const int e = *e_dev;
    const ull seed = keys[e];
    GRID_STRIDE_LOOP(clause, (ull)TOTAL_CLAUSES) {
        int ck = clause_output[clause];
        bool has_space = (clause_len[clause] <= (NLITS_T)MAX_INCLUDED_LITERALS);
        bool dropped = (clause_drop_mask[clause] == 1);
        NCLASSES_T n1a = 0, n1b = 0, n2 = 0;

        ull rng_k = rng_hash(seed, clause, 0xFEEDFACEULL);
        uint rng_counter = 0;

        ull class_id;
        LOOP_CLASS_ID(class_id, clause) {
            uint8_t fb = FB_NONE;

            if (!dropped) {
                float w = clause_weights[weight_offset(class_id, clause)];
                float g = grad[class_id];

                int polarity = (w >= 0) - (w < 0);
                int sign_grad = (g >= 0) - (g < 0);
                int dir = polarity * sign_grad;
                float update_prob = 1.0f - expf(-lambda_ * fabsf(g));

                if (rand_uniform(rng_k, &rng_counter) < update_prob) {
                    bool t1a = (ck == 1 && dir > 0 && has_space);
                    bool t1b = ((ck == 0 && dir > 0) || (ck == 1 && dir > 0 && !has_space));
                    bool t2 = (ck == 1 && dir < 0);
                    fb = t1a ? FB_T1A : (t1b ? FB_T1B : (t2 ? FB_T2 : FB_NONE));
                }
            }
            feedback_type[fbtype_offset(class_id, clause)] = fb;
            n1a += (fb == FB_T1A);
            n1b += (fb == FB_T1B);
            n2 += (fb == FB_T2);
        }
        fb_lists_append(clause, n1a, n1b, n2, t1b_n, t1b_cid, t1b_cnt, t1a_t2_n, t1a_t2_cid, t1a_cnt, t2_cnt);
    }
}

__device__ inline uint8_t select_fb_delta_l(const ull seed, ull rng_id, float uprob, float delta_L, int ck,
                                            bool has_space) {
    ull rng_k = rng_hash(seed, rng_id, 0xFEEDFACEULL);
    uint rng_counter = 0;

    uint8_t fb;
    if (rand_uniform(rng_k, &rng_counter) >= uprob) {
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

extern "C" __global__ void decide_feedback_delta_l(const ull* keys, const int* e_dev, const float* loss, const float* loss_neg_ck,
                                                   const NLITS_T* clause_len, const int8_t* clause_output,
                                                   const int8_t* clause_drop_mask, const float lambda_,
                                                   uint8_t* feedback_type, uint* t1b_n, uint* t1b_cid,
                                                   NCLASSES_T* t1b_cnt, uint* t1a_t2_n, uint* t1a_t2_cid,
                                                   NCLASSES_T* t1a_cnt, NCLASSES_T* t2_cnt) {
    const int e = *e_dev;
    const ull seed = keys[e];
    GRID_STRIDE_LOOP(clause_id, (ull)TOTAL_CLAUSES) {
        if (clause_drop_mask[clause_id] == 1) {
            feedback_type[clause_id] = FB_NONE;
            continue;
        }

        int ck = clause_output[clause_id];
        bool has_space = (clause_len[clause_id] <= (NLITS_T)MAX_INCLUDED_LITERALS);
        float delta_L = *loss - loss_neg_ck[clause_id];
        float update_prob = 1.0f - expf(-lambda_ * fabsf(delta_L));
        uint8_t fb = select_fb_delta_l(seed, clause_id, update_prob, delta_L, ck, has_space);
        feedback_type[clause_id] = fb;
        fb_lists_append(clause_id, fb == FB_T1A, fb == FB_T1B, fb == FB_T2, t1b_n, t1b_cid, t1b_cnt, t1a_t2_n,
                        t1a_t2_cid, t1a_cnt, t2_cnt);
    }
}

extern "C" __global__ void update_weights(const float* grad, const float lr, const int8_t* clause_output,
                                          const int8_t* clause_drop_mask, float* clause_weights) {
    GRID_STRIDE_LOOP(clause, (ull)TOTAL_CLAUSES) {
        if (clause_drop_mask[clause] == 1 || !clause_output[clause])
            continue;

        ull class_id;
        LOOP_CLASS_ID(class_id, clause) {
            ull idx = weight_offset(class_id, clause);
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
