#ifdef IS_NEOVIM_CLANGD_ENV
#include "../../_core/backends/common.h"
#include "../../_core/backends/cpu.h"
#include "../../_core/backends/feedback.c"
#include "../../_core/backends/rng.h"
#include "act.h"

#define FB_NONE 0
#define FB_T1A 1
#define FB_T1B 2
#define FB_T2 3

#define FB_SIGNAL_GRAD 0
#define FB_SIGNAL_DELTA_L 1
#define FB_SIGNAL FB_SIGNAL_DELTA_L

INLINE_FN void compute_loss(const float* restrict y, const float* restrict y_hat, float* restrict grad,
                            float* restrict loss) {}
#endif

void votes_activation(const float* restrict votes, float* restrict y_hat) { compute_act(votes, y_hat); }

void votes_activation_batch(const float* restrict votes, int n_samples, float* restrict y_hat) {
#pragma omp parallel for schedule(static)
    for (int e = 0; e < n_samples; e++)
        compute_act(&votes[(ull)e * CLASSES], &y_hat[(ull)e * CLASSES]);
}

void loss_gradient(const float* restrict y_hat, const float* restrict y, float* restrict grad, float* restrict loss) {
    compute_loss(y, y_hat, grad, loss);
}

void compute_votes_neg_ck(const float* restrict votes, const float* restrict clause_weights,
                          const int8_t* restrict clause_output, float* restrict votes_neg_ck) {
#pragma omp parallel for schedule(dynamic)
    for (ull clause_class = 0; clause_class < (ull)TOTAL_CLAUSES * (ull)CLASSES; clause_class++) {
        ull clause_id = clause_class / (ull)CLASSES;
        ull class_id = clause_class % (ull)CLASSES;

#if COALESCED == 0
        if (class_id != clause_id / (ull)CLAUSES_PER_CLASS) {
            votes_neg_ck[clause_class] = votes[class_id];
            continue;
        }
#endif
        int ck = clause_output[clause_id];
        ull rel_clause = clause_id % (ull)CLAUSES_PER_CLASS;
        float w = clause_weights[class_id * (ull)CLAUSES_PER_CLASS + rel_clause];
        votes_neg_ck[clause_class] = votes[class_id] - (ck ? w : -w);
    }
}

void compute_loss_neg_ck(const float* restrict y_hat_neg_ck, const float* restrict y, float* restrict loss_neg_ck) {
#pragma omp parallel for schedule(dynamic)
    for (ull clause_id = 0; clause_id < (ull)TOTAL_CLAUSES; clause_id++) {
        float loss = 0.0f;
        compute_loss(y, &y_hat_neg_ck[clause_id * (ull)CLASSES], NULL, &loss);
        loss_neg_ck[clause_id] = loss;
    }
}

void decide_feedback_grad(const ull seed, const float* restrict grad, const float* restrict clause_weights,
                          const NLITS_T* restrict clause_len, const int8_t* restrict clause_output,
                          const int8_t* restrict clause_drop_mask, const float lambda_,
                          uint8_t* restrict feedback_type) {
#pragma omp parallel for schedule(dynamic)
    for (ull clause = 0; clause < (ull)TOTAL_CLAUSES; clause++) {
        ull rel_clause = clause % (ull)CLAUSES_PER_CLASS;
        int ck = clause_output[clause];
        bool has_space = (clause_len[clause] <= (NLITS_T)MAX_INCLUDED_LITERALS);
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

static inline uint8_t select_fb_delta_l(const ull seed, ull rng_id, float uprob, float delta_L, int ck,
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

void decide_feedback_delta_l(const ull seed, const float* restrict loss, const float* restrict loss_neg_ck,
                             const NLITS_T* restrict clause_len, const int8_t* restrict clause_output,
                             const int8_t* restrict clause_drop_mask, const float lambda_,
                             uint8_t* restrict feedback_type) {
#pragma omp parallel for schedule(dynamic)
    for (ull clause_id = 0; clause_id < (ull)TOTAL_CLAUSES; clause_id++) {
        if (clause_drop_mask[clause_id] == 1) {
            feedback_type[clause_id] = FB_NONE;
            continue;
        }

        int ck = clause_output[clause_id];
        bool has_space = (clause_len[clause_id] <= (NLITS_T)MAX_INCLUDED_LITERALS);
        float delta_L = *loss - loss_neg_ck[clause_id];
        float update_prob = 1.0f - expf(-lambda_ * fabsf(delta_L));
        feedback_type[clause_id] = select_fb_delta_l(seed, clause_id, update_prob, delta_L, ck, has_space);
    }
}

void update_clauses(const ull seed, const int8_t* restrict clause_output, const NPATCHES_T* restrict selected_patch_ids,
                    const FBOUND_T* restrict X, const int e, const NLITS_T* restrict literal_offsets,
                    const uint8_t* restrict feedback_type, TA_STATE_T* restrict global_ta_states,
                    int8_t* restrict is_clause_synced) {
    const FBOUND_T* Xe = &X[(ull)e * HEIGHT * WIDTH * DEPTH];

#pragma omp parallel for schedule(dynamic)
    for (ull clause = 0; clause < (ull)TOTAL_CLAUSES; clause++) {
        ull rel_clause = clause % (ull)CLAUSES_PER_CLASS;
        TA_STATE_T* ta_states = &global_ta_states[clause * (ull)N_LITERALS];

#if (N_PATCHES > 1)
        int patch_id = clause_output[clause] ? (int)selected_patch_ids[clause] : 0;
        int patch_idx_y = patch_id / N_PATCHES_X;
        int patch_idx_x = patch_id % N_PATCHES_X;
#else
        int patch_idx_y = 0, patch_idx_x = 0;
#endif

        ull rng_k = rng_hash(seed, clause, 0xCAFEBABEULL);
        uint rng_counter = 0;

#if FB_SIGNAL == FB_SIGNAL_GRAD
        ull class_id;
        LOOP_CLASS_ID(class_id, clause) {
            uint8_t fb = feedback_type[rel_clause * (ull)CLASSES + class_id];
            if (fb == FB_NONE)
                continue;
            apply_feedback(rng_k, &rng_counter, fb, Xe, patch_idx_y, patch_idx_x, literal_offsets, ta_states);
            is_clause_synced[clause] = 0;
        }
#else
        uint8_t fb = feedback_type[clause];
        if (fb != FB_NONE) {
            apply_feedback(rng_k, &rng_counter, fb, Xe, patch_idx_y, patch_idx_x, literal_offsets, ta_states);
            is_clause_synced[clause] = 0;
        }
#endif
    }
}

void update_weights(const float* restrict grad, const float lr, const int8_t* restrict clause_output,
                    const int8_t* restrict clause_drop_mask, float* restrict clause_weights) {
#pragma omp parallel for schedule(dynamic)
    for (ull clause = 0; clause < (ull)TOTAL_CLAUSES; clause++) {
        if (clause_drop_mask[clause] == 1 || !clause_output[clause])
            continue;

        ull rel_clause = clause % (ull)CLAUSES_PER_CLASS;
        ull class_id;
        LOOP_CLASS_ID(class_id, clause) {
            ull idx = class_id * (ull)CLAUSES_PER_CLASS + rel_clause;
            float new_w = clause_weights[idx] + lr * grad[class_id];
#if ALLOW_POLARITY_CHANGE
            clause_weights[idx] = clip(new_w, -MAX_WEIGHT, MAX_WEIGHT);
#else
            clause_weights[idx] = (clause_weights[idx] >= 0) ? clip(new_w, 0.0001f, MAX_WEIGHT) : clip(new_w, -MAX_WEIGHT, -0.0001f);
#endif
        }
    }
}
