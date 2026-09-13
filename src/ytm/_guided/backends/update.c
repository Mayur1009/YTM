#ifdef IS_NEOVIM_CLANGD_ENV
#include "../../_core/backends/common.h"
#include "../../_core/backends/cpu.h"
#include "../../_core/backends/feedback.c"
#include "../../_core/backends/rng.h"
#include "act_loss.h"

#define FB_NONE 0
#define FB_T1A 1
#define FB_T1B 2
#define FB_T2 3

#define FB_SIGNAL_GRAD 0
#define FB_SIGNAL_DELTA_L 1
#define FB_SIGNAL FB_SIGNAL_DELTA_L
#endif

void votes_activation(const float* votes, float* y_hat) {
#if ACT_FN == ACT_SOFTMAX
    _softmax(votes, y_hat);
#elif ACT_FN == ACT_SIGMOID
    for (int c = 0; c < CLASSES; c++)
        y_hat[c] = _sigmoid(votes[c]);
#else
    for (int c = 0; c < CLASSES; c++)
        y_hat[c] = _identity(votes[c]);
#endif
}

void votes_activation_batch(const float* votes, int n_samples, float* y_hat) {
#if ACT_FN == ACT_SOFTMAX
#pragma omp parallel for schedule(static)
    for (int e = 0; e < n_samples; e++)
        _softmax(&votes[(ull)e * CLASSES], &y_hat[(ull)e * CLASSES]);
#else
    ull total = (ull)n_samples * (ull)CLASSES;
#pragma omp parallel for schedule(static)
    for (ull idx = 0; idx < total; idx++)
#if ACT_FN == ACT_SIGMOID
        y_hat[idx] = _sigmoid(votes[idx]);
#else
        y_hat[idx] = _identity(votes[idx]);
#endif
#endif
}

void loss_gradient(const float* y_hat, const float* y, const float* class_weights, float* grad, float* loss) {
    loss_gradient_impl(y_hat, y, class_weights, grad, loss);
}

void compute_votes_neg_ck(const float* votes, const float* clause_weights, const int* selected_patch_ids,
                          float* votes_neg_ck) {
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
        int ck = (selected_patch_ids[clause_id] >= 0) ? 1 : 0;
        ull rel_clause = clause_id % (ull)CLAUSES_PER_CLASS;
        float w = clause_weights[class_id * (ull)CLAUSES_PER_CLASS + rel_clause];
        votes_neg_ck[clause_class] = votes[class_id] - (ck ? w : -w);
    }
}

void compute_loss_neg_ck(const float* y_hat_neg_ck, const float* y, const float* loss_class_weights,
                         float* loss_neg_ck) {
#pragma omp parallel for schedule(dynamic)
    for (ull clause_id = 0; clause_id < (ull)TOTAL_CLAUSES; clause_id++) {
        float loss = 0.0f;
        loss_gradient_impl(&y_hat_neg_ck[clause_id * (ull)CLASSES], y, loss_class_weights, NULL, &loss);
        loss_neg_ck[clause_id] = loss;
    }
}

void decide_feedback_grad(const ull seed, const float* grad, const float* clause_weights, const int* clause_density,
                          const int* selected_patch_ids, const int8_t* clause_drop_mask, const float lambda_,
                          uint8_t* feedback_type) {
#pragma omp parallel for schedule(dynamic)
    for (ull clause_class = 0; clause_class < (ull)TOTAL_CLAUSES * (ull)CLASSES; clause_class++) {
        ull clause_id = clause_class / (ull)CLASSES;
        ull class_id = clause_class % (ull)CLASSES;

#if COALESCED == 0
        if (class_id != clause_id / (ull)CLAUSES_PER_CLASS)
            continue;
#endif

        if (clause_drop_mask[clause_id] == 1) {
            feedback_type[clause_class] = FB_NONE;
            continue;
        }

        int ck = (selected_patch_ids[clause_id] >= 0) ? 1 : 0;
        bool has_space = (clause_density[clause_id] <= (int)MAX_INCLUDED_LITERALS);
        ull rel_clause = clause_id % (ull)CLAUSES_PER_CLASS;
        float w = clause_weights[class_id * (ull)CLAUSES_PER_CLASS + rel_clause];
        float g = grad[class_id];

        int polarity = (w >= 0) - (w < 0);
        int sign_grad = (g >= 0) - (g < 0);
        int dir = polarity * sign_grad;
        float update_prob = 1.0f - expf(-lambda_ * fabsf(g));

        ull rng_k = rng_hash(seed, clause_class, 0xFEEDFACEULL);
        uint rng_counter = 0;

        uint8_t fb;
        if (rand_uniform(rng_k, &rng_counter) > update_prob) {
            fb = FB_NONE;
        } else {
            bool t1a = (ck == 1 && dir > 0 && has_space);
            bool t1b = ((ck == 0 && dir > 0) || (ck == 1 && dir > 0 && !has_space));
            bool t2 = (ck == 1 && dir < 0);
            fb = t1a ? FB_T1A : (t1b ? FB_T1B : (t2 ? FB_T2 : FB_NONE));
        }
        feedback_type[clause_class] = fb;
    }
}

static inline uint8_t select_fb_delta_l(const ull seed, ull rng_id, float uprob, float delta_L, int ck, bool has_space) {
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

void decide_feedback_delta_l(const ull seed, const float* loss, const float* loss_neg_ck, const int* clause_density,
                             const int* selected_patch_ids, const int8_t* clause_drop_mask, const float lambda_,
                             uint8_t* feedback_type) {
#pragma omp parallel for schedule(dynamic)
    for (ull clause_id = 0; clause_id < (ull)TOTAL_CLAUSES; clause_id++) {
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

void update_clauses(const ull seed, const int* selected_patch_ids, const int* X, const int e, const int* feat_mins,
                    const int* literal_offsets, const uint8_t* feedback_type, uint* global_ta_states,
                    int8_t* is_clause_synced) {
    const int* Xe = &X[(ull)e * HEIGHT * WIDTH * DEPTH];

#pragma omp parallel for schedule(dynamic)
    for (ull clause = 0; clause < (ull)TOTAL_CLAUSES; clause++) {
        uint* ta_states = &global_ta_states[clause * (ull)N_LITERALS];

        int patch_id = selected_patch_ids[clause];
        int patch_idx_y = (patch_id >= 0) ? patch_id / N_PATCHES_X : -1;
        int patch_idx_x = (patch_id >= 0) ? patch_id % N_PATCHES_X : -1;

        ull rng_k = rng_hash(seed, clause, 0xCAFEBABEULL);
        uint rng_counter = 0;

#if FB_SIGNAL == FB_SIGNAL_GRAD
        ull class_id;
        LOOP_CLASS_ID(class_id, clause) {
            uint8_t fb = feedback_type[clause * (ull)CLASSES + class_id];
            if (fb == FB_NONE)
                continue;
            apply_feedback(rng_k, &rng_counter, fb, Xe, patch_idx_y, patch_idx_x, feat_mins, literal_offsets, ta_states);
            is_clause_synced[clause] = 0;
        }
#else
        uint8_t fb = feedback_type[clause];
        if (fb != FB_NONE) {
            apply_feedback(rng_k, &rng_counter, fb, Xe, patch_idx_y, patch_idx_x, feat_mins, literal_offsets, ta_states);
            is_clause_synced[clause] = 0;
        }
#endif
    }
}

void update_weights(const float* grad, const float lr, const int* selected_patch_ids, const int8_t* clause_drop_mask,
                    float* clause_weights) {
#pragma omp parallel for schedule(dynamic)
    for (ull clause = 0; clause < (ull)TOTAL_CLAUSES; clause++) {
        if (clause_drop_mask[clause] == 1 || selected_patch_ids[clause] < 0)
            continue;

        ull rel_clause = clause % (ull)CLAUSES_PER_CLASS;
        ull class_id;
        LOOP_CLASS_ID(class_id, clause) {
            ull idx = class_id * (ull)CLAUSES_PER_CLASS + rel_clause;
            clause_weights[idx] = clip(clause_weights[idx] + lr * grad[class_id], -MAX_WEIGHT, MAX_WEIGHT);
        }
    }
}

void update_bias(const float* grad, const float lr, float* bias) {
#if BIAS
#pragma omp parallel for schedule(static)
    for (int c = 0; c < CLASSES; c++)
        bias[c] += lr * grad[c];
#endif
}
