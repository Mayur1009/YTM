#ifdef IS_NEOVIM_CLANGD_ENV
#include "common.c"
#endif

#include <math.h>

static inline int geometric_sample(ull rng_key, uint* rng_counter, float p) {
    float u = rand_uniform(rng_key, rng_counter);
    float p_clamp = (p > 1e-9f) ? p : 1e-9f;
    return (int)(logf(1.0f - u + 1e-9f) / logf(1.0f - p_clamp)) + 1;
}

static inline void literal_dec_with_p(ull rng_key, uint* rng_counter, uint* ta_state, int start, int end, int offset,
                                      float p) {
    int li = start + geometric_sample(rng_key, rng_counter, p) - 1;
    while (li < end) {
        if (ta_state[li + offset] > 0)
            ta_state[li + offset] -= 1;
        li += geometric_sample(rng_key, rng_counter, p);
    }
}

static inline void literal_inc(uint* ta_state, int start, int end, int offset, uint max_val) {
    for (int li = start; li < end; ++li)
        ta_state[li + offset] += (ta_state[li + offset] < max_val);
}

static inline void literal_inc_maybe_p(ull rng_key, uint* rng_counter, uint* ta_state, int start, int end, int offset,
                                       float p) {
#if BOOST_TP_FB
    literal_inc(ta_state, start, end, offset, MAX_TA_STATE);
#else
    int li = start + geometric_sample(rng_key, rng_counter, p) - 1;
    while (li < end) {
        if (ta_state[li + offset] < MAX_TA_STATE)
            ta_state[li + offset] += 1;
        li += geometric_sample(rng_key, rng_counter, p);
    }
#endif
}

static inline void type1a_fb(ull rng_key, uint* rng_counter, uint* ta_states, const int* X, int patch_idx_y,
                             int patch_idx_x, const int* feat_mins, const int* literal_offsets) {
#if POSITION_LITERALS
    literal_inc_maybe_p(rng_key, rng_counter, ta_states, 0, patch_idx_y, 0, 1.0f - S_INV);
    literal_dec_with_p(rng_key, rng_counter, ta_states, patch_idx_y, N_POSITION_FEATS_Y, 0, S_INV);

    literal_inc_maybe_p(rng_key, rng_counter, ta_states, N_POSITION_FEATS_Y, N_POSITION_FEATS_Y + patch_idx_x, 0,
                        1.0f - S_INV);
    literal_dec_with_p(rng_key, rng_counter, ta_states, N_POSITION_FEATS_Y + patch_idx_x, N_POSITION_FEATS, 0, S_INV);

#if NEGATED_LITERALS
    literal_dec_with_p(rng_key, rng_counter, ta_states, 0, patch_idx_y, N_LITERALS / 2, S_INV);
    literal_inc_maybe_p(rng_key, rng_counter, ta_states, patch_idx_y, N_POSITION_FEATS_Y, N_LITERALS / 2, 1.0f - S_INV);

    literal_dec_with_p(rng_key, rng_counter, ta_states, N_POSITION_FEATS_Y, N_POSITION_FEATS_Y + patch_idx_x,
                       N_LITERALS / 2, S_INV);
    literal_inc_maybe_p(rng_key, rng_counter, ta_states, N_POSITION_FEATS_Y + patch_idx_x, N_POSITION_FEATS,
                        N_LITERALS / 2, 1.0f - S_INV);
#endif
#endif

    for (int fid = 0; fid < N_RAW_PATCH_FEATS; fid++) {
        int lit_start = N_POSITION_FEATS + literal_offsets[fid];
        int lit_end = N_POSITION_FEATS + literal_offsets[fid + 1];
        int shifted_val = get_feature_value(X, patch_idx_y, patch_idx_x, fid) - feat_mins[fid];

        literal_inc_maybe_p(rng_key, rng_counter, ta_states, lit_start, lit_start + shifted_val, 0, 1.0f - S_INV);
        literal_dec_with_p(rng_key, rng_counter, ta_states, lit_start + shifted_val, lit_end, 0, S_INV);

#if NEGATED_LITERALS
        literal_dec_with_p(rng_key, rng_counter, ta_states, lit_start, lit_start + shifted_val, N_LITERALS / 2, S_INV);
        literal_inc_maybe_p(rng_key, rng_counter, ta_states, lit_start + shifted_val, lit_end, N_LITERALS / 2,
                            1.0f - S_INV);
#endif
    }
}

static inline void type1b_fb(ull rng_key, uint* rng_counter, uint* ta_state) {
    int li = geometric_sample(rng_key, rng_counter, S_INV) - 1;
    while (li < N_LITERALS) {
        if (ta_state[li] > 0)
            ta_state[li] -= 1;
        li += geometric_sample(rng_key, rng_counter, S_INV);
    }
}

static inline void type2_fb(uint* ta_state, const int* X, int patch_idx_y, int patch_idx_x, const int* feat_mins,
                            const int* literal_offsets) {
#if POSITION_LITERALS
    literal_inc(ta_state, patch_idx_y, N_POSITION_FEATS_Y, 0, INCLUDE_STATE);
    literal_inc(ta_state, N_POSITION_FEATS_Y + patch_idx_x, N_POSITION_FEATS, 0, INCLUDE_STATE);

#if NEGATED_LITERALS
    literal_inc(ta_state, 0, patch_idx_y, N_LITERALS / 2, INCLUDE_STATE);
    literal_inc(ta_state, N_POSITION_FEATS_Y, N_POSITION_FEATS_Y + patch_idx_x, N_LITERALS / 2, INCLUDE_STATE);
#endif
#endif

    for (int fid = 0; fid < N_RAW_PATCH_FEATS; fid++) {
        int lit_start = N_POSITION_FEATS + literal_offsets[fid];
        int lit_end = N_POSITION_FEATS + literal_offsets[fid + 1];
        int shifted_val = get_feature_value(X, patch_idx_y, patch_idx_x, fid) - feat_mins[fid];

        literal_inc(ta_state, lit_start + shifted_val, lit_end, 0, INCLUDE_STATE);

#if NEGATED_LITERALS
        literal_inc(ta_state, lit_start, lit_start + shifted_val, N_LITERALS / 2, INCLUDE_STATE);
#endif
    }
}

static inline void update_clause_class(ull class_id, ull clause, ull rel_clause, int clause_output, int patch_idx_y,
                                       int patch_idx_x, int clause_density, uint* ta_states, float* clause_weights,
                                       const int* Xe, const float* targets_e, const float* prob, const int* feat_mins,
                                       const int* literal_offsets, int8_t* is_clause_synced, ull rng_k,
                                       uint* rng_counter) {
    if ((targets_e[class_id] < 0.0f && rand_uniform(rng_k, rng_counter) > (Q / fmaxf(1.0f, (CLASSES - 1)))) ||
        prob[class_id] == 0.0f || rand_uniform(rng_k, rng_counter) > fabsf(prob[class_id]))
        return;

    int target = (prob[class_id] > 0.0f) ? 1 : -1;
    is_clause_synced[clause] = 0;

    float* weight = &clause_weights[class_id * (ull)CLAUSES_PER_CLASS + rel_clause];
    int sign = (*weight >= 0) - (*weight < 0);
    bool has_space = (clause_density <= (int)MAX_INCLUDED_LITERALS);
    bool t1 = (target * sign) > 0;

    if (t1 && clause_output && has_space) {
#if TYPE1A_FB
#if WEIGHTED
        if (fabsf(*weight) < MAX_WEIGHT)
            (*weight) += sign * 1.0f;
#endif
        type1a_fb(rng_k, rng_counter, ta_states, Xe, patch_idx_y, patch_idx_x, feat_mins, literal_offsets);
#endif
    } else if (t1 && !(clause_output && has_space)) {
#if TYPE1B_FB
        type1b_fb(rng_k, rng_counter, ta_states);
#endif
    } else if ((target * sign) < 0 && clause_output) {
#if TYPE2_FB
#if WEIGHTED
        if (fabsf(*weight) < MAX_WEIGHT)
            (*weight) -= sign * 1.0f;
#if ALLOW_POLARITY_CHANGE == 0
        if (sign == 1 && *weight < 0)
            *weight = 1;
        if (sign == -1 && *weight >= 0)
            *weight = -1;
#endif
#endif
#if NEGATIVE_CLAUSES == 0
        if (*weight < 1)
            *weight = 1;
#endif
        type2_fb(ta_states, Xe, patch_idx_y, patch_idx_x, feat_mins, literal_offsets);
#endif
    }
}

static inline float uprob_fun(float y, float y_hat) { return (y - y_hat) / (T_MAX - T_MIN); }

void calc_update_prob(const float* votes, const float* targets, const int e, float* prob) {
#pragma omp parallel for
    for (ull class_id = 0; class_id < (ull)CLASSES; class_id++) {
        float target = targets[(ull)e * CLASSES + class_id];
        float v = clip(votes[class_id], T_MIN, T_MAX);
        prob[class_id] = uprob_fun(target, v);
    }
}

void update_clauses(const ull seed, const int* selected_patch_ids, const int* clause_density,
                    const int8_t* clause_drop_mask, const int* X, const float* targets, const int e, const float* prob,
                    uint* global_ta_states, float* clause_weights, const int* feat_mins, const int* literal_offsets,
                    int8_t* is_clause_synced) {
    const int* Xe = &X[(ull)e * HEIGHT * WIDTH * DEPTH];
    const float* targets_e = &targets[(ull)e * CLASSES];

#pragma omp parallel for schedule(dynamic)
    for (ull clause = 0; clause < (ull)TOTAL_CLAUSES; clause++) {
        if (clause_drop_mask[clause] == 1)
            continue;

        ull rng_k = rng_hash(seed, clause, (ull)e, 0xCAFEBABEULL);
        uint rng_counter = 0;

        uint* ta_states = &global_ta_states[clause * (ull)N_LITERALS];
        int patch_id = selected_patch_ids[clause];
        int clause_output = (patch_id >= 0) ? 1 : 0;

        int patch_idx_y = -1, patch_idx_x = -1;
        if (clause_output) {
            patch_idx_y = patch_id / N_PATCHES_X;
            patch_idx_x = patch_id % N_PATCHES_X;
        }

        int cd = clause_density[clause];
        ull rel_clause = clause % (ull)CLAUSES_PER_CLASS;

        ull class_id;
        LOOP_CLASS_ID(class_id, clause) {
            update_clause_class(class_id, clause, rel_clause, clause_output, patch_idx_y, patch_idx_x, cd, ta_states,
                                clause_weights, Xe, targets_e, prob, feat_mins, literal_offsets, is_clause_synced,
                                rng_k, &rng_counter);
        }
    }
}
