#ifdef IS_NEOVIM_CLANGD_ENV
#include "common.c"
#include "losses.c"
#define FB_SIGNAL 0
#endif

#include <math.h>

#define FB_NONE 0
#define FB_T1A 1
#define FB_T1B 2
#define FB_T2 3

#define FB_SIGNAL_GRAD 0
#define FB_SIGNAL_DELTAL 1

static inline int geom_sample(ull rng_key, uint* rng_counter, float p) {
    float u = rand_uniform(rng_key, rng_counter);
    double u_clamp = clip(u, 1e-7f, 1.0f - 1e-7f);
    double log_u = log1p(-u_clamp);
    double log_p = log1p(-p);
    int sample = (int)(log_u / log_p) + 1;
    return sample;
}

static inline void dec_literals(ull rng_key, uint* rng_counter, int start, int end, int offset, uint* ta_state) {
    if (S > 1.0f) {
        int li = start + geom_sample(rng_key, rng_counter, S_INV) - 1;
        while (li < end) {
            if (ta_state[li + offset] > 0)
                ta_state[li + offset] -= 1;
            li += geom_sample(rng_key, rng_counter, S_INV);
        }
    } else {
        for (int li = start; li < end; ++li)
            if (ta_state[li + offset] > 0)
                ta_state[li + offset] -= 1;
    }
}

static inline void t2_inc_literals(int start, int end, int offset, uint* ta_state) {
    for (int li = start; li < end; ++li)
        if (ta_state[li + offset] < MAX_TA_STATE)
            ta_state[li + offset] += 1;
}

static inline void t1a_inc_literals(ull rng_key, uint* rng_counter, int start, int end, int offset, uint* ta_state) {
#if BOOST_TP_FB
    for (int li = start; li < end; ++li)
        if (ta_state[li + offset] < MAX_TA_STATE)
            ta_state[li + offset] += 1;
#else
    if (S > 1.0f) {
        int li = start + geom_sample(rng_key, rng_counter, 1 - S_INV) - 1;
        while (li < end) {
            if (ta_state[li + offset] < MAX_TA_STATE)
                ta_state[li + offset] += 1;
            li += geom_sample(rng_key, rng_counter, 1 - S_INV);
        }
    }
#endif
}

static inline void type1a_fb(ull rng_key, uint* rng_counter, const int* X, int patch_idx_y, int patch_idx_x,
                             const int* feat_mins, const int* literal_offsets, uint* ta_states) {
#if POSITION_LITERALS
    t1a_inc_literals(rng_key, rng_counter, 0, patch_idx_y, 0, ta_states);
    t1a_inc_literals(rng_key, rng_counter, N_POSITION_FEATS_Y, N_POSITION_FEATS_Y + patch_idx_x, 0, ta_states);

    dec_literals(rng_key, rng_counter, patch_idx_y, N_POSITION_FEATS_Y, 0, ta_states);
    dec_literals(rng_key, rng_counter, N_POSITION_FEATS_Y + patch_idx_x, N_POSITION_FEATS, 0, ta_states);
#if NEGATED_LITERALS
    t1a_inc_literals(rng_key, rng_counter, patch_idx_y, N_POSITION_FEATS_Y, N_LITERALS / 2, ta_states);
    t1a_inc_literals(rng_key, rng_counter, N_POSITION_FEATS_Y + patch_idx_x, N_POSITION_FEATS, N_LITERALS / 2,
                     ta_states);

    dec_literals(rng_key, rng_counter, 0, patch_idx_y, N_LITERALS / 2, ta_states);
    dec_literals(rng_key, rng_counter, N_POSITION_FEATS_Y, N_POSITION_FEATS_Y + patch_idx_x, N_LITERALS / 2,
                ta_states);
#endif
#endif

    for (int fid = 0; fid < N_RAW_PATCH_FEATS; fid++) {
        int lit_start = N_POSITION_FEATS + literal_offsets[fid];
        int lit_end = N_POSITION_FEATS + literal_offsets[fid + 1];
        int shifted_val = get_feature_value(X, patch_idx_y, patch_idx_x, fid) - feat_mins[fid];

        t1a_inc_literals(rng_key, rng_counter, lit_start, lit_start + shifted_val, 0, ta_states);
        dec_literals(rng_key, rng_counter, lit_start + shifted_val, lit_end, 0, ta_states);
#if NEGATED_LITERALS
        t1a_inc_literals(rng_key, rng_counter, lit_start + shifted_val, lit_end, N_LITERALS / 2, ta_states);
        dec_literals(rng_key, rng_counter, lit_start, lit_start + shifted_val, N_LITERALS / 2, ta_states);
#endif
    }
}

static inline void type1b_fb(ull rng_key, uint* rng_counter, uint* ta_state) {
    dec_literals(rng_key, rng_counter, 0, N_LITERALS, 0, ta_state);
}

static inline void type2_fb(const int* X, int patch_idx_y, int patch_idx_x, const int* feat_mins,
                            const int* literal_offsets, uint* ta_state) {
#if POSITION_LITERALS
    t2_inc_literals(patch_idx_y, N_POSITION_FEATS_Y, 0, ta_state);
    t2_inc_literals(N_POSITION_FEATS_Y + patch_idx_x, N_POSITION_FEATS, 0, ta_state);

#if NEGATED_LITERALS
    t2_inc_literals(0, patch_idx_y, N_LITERALS / 2, ta_state);
    t2_inc_literals(N_POSITION_FEATS_Y, N_POSITION_FEATS_Y + patch_idx_x, N_LITERALS / 2, ta_state);
#endif
#endif

    for (int fid = 0; fid < N_RAW_PATCH_FEATS; fid++) {
        int lit_start = N_POSITION_FEATS + literal_offsets[fid];
        int lit_end = N_POSITION_FEATS + literal_offsets[fid + 1];
        int shifted_val = get_feature_value(X, patch_idx_y, patch_idx_x, fid) - feat_mins[fid];

        t2_inc_literals(lit_start + shifted_val, lit_end, 0, ta_state);

#if NEGATED_LITERALS
        t2_inc_literals(lit_start, lit_start + shifted_val, N_LITERALS / 2, ta_state);
#endif
    }
}

static inline void apply_feedback(ull rng_key, uint* rng_counter, uint8_t fb_type, const int* Xe, int patch_idx_y,
                                  int patch_idx_x, const int* feat_mins, const int* literal_offsets,
                                  uint* ta_states) {
    if (fb_type == FB_T1A) {
#if TYPE1A_FB
        type1a_fb(rng_key, rng_counter, Xe, patch_idx_y, patch_idx_x, feat_mins, literal_offsets, ta_states);
#endif
    } else if (fb_type == FB_T1B) {
#if TYPE1B_FB
        type1b_fb(rng_key, rng_counter, ta_states);
#endif
    } else if (fb_type == FB_T2) {
#if TYPE2_FB
        type2_fb(Xe, patch_idx_y, patch_idx_x, feat_mins, literal_offsets, ta_states);
#endif
    }
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

static inline uint8_t select_fb_delta_l(const ull seed, ull rng_id, float uprob, float delta_L, int ck,
                                        bool has_space) {
    ull rng_k = rng_hash(seed, rng_id, 0xFEEDFACEULL, 0xD00D1E00ULL);
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

        ull rng_k = rng_hash(seed, clause_class, 0xFEEDFACEULL, 0xD00D1E00ULL);
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

void update_clauses(const ull seed, const int* selected_patch_ids, const int* X, const int e, const int* feat_mins,
                    const int* literal_offsets, const uint8_t* feedback_type, uint* global_ta_states,
                    int8_t* is_clause_synced) {
    const int* Xe = &X[(ull)e * HEIGHT * WIDTH * DEPTH];

#pragma omp parallel for schedule(dynamic)
    for (ull clause = 0; clause < (ull)TOTAL_CLAUSES; clause++) {
        ull rng_k = rng_hash(seed, clause, 0xC0DEBA5EULL, 0xCAFEBABEULL);
        uint rng_counter = 0;

        uint* ta_states = &global_ta_states[clause * (ull)N_LITERALS];
        int patch_id = selected_patch_ids[clause];
        int clause_output = (patch_id >= 0) ? 1 : 0;

        int patch_idx_y = -1, patch_idx_x = -1;
        if (clause_output) {
            patch_idx_y = patch_id / N_PATCHES_X;
            patch_idx_x = patch_id % N_PATCHES_X;
        }

#if FB_SIGNAL == FB_SIGNAL_GRAD
        ull class_id;
        LOOP_CLASS_ID(class_id, clause) {
            uint8_t fb = feedback_type[clause * (ull)CLASSES + class_id];
            apply_feedback(rng_k, &rng_counter, fb, Xe, patch_idx_y, patch_idx_x, feat_mins, literal_offsets,
                           ta_states);
            if (fb != FB_NONE)
                is_clause_synced[clause] = 0;
        }
#else
        uint8_t fb = feedback_type[clause];
        apply_feedback(rng_k, &rng_counter, fb, Xe, patch_idx_y, patch_idx_x, feat_mins, literal_offsets, ta_states);
        if (fb != FB_NONE)
            is_clause_synced[clause] = 0;
#endif
    }
}
