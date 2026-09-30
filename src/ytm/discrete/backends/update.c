#ifdef IS_NEOVIM_CLANGD_ENV
#include "../../_core/backends/common.h"
#include "../../_core/backends/cpu.h"
#include "../../_core/backends/feedback.h"
#include "../../_core/backends/rng.h"

#define T_MIN -100.0f
#define T_MAX 100.0f
#define FB_NONE 0
#define FB_T1A 1
#define FB_T1B 2
#define FB_T2 3
#endif

void decide_feedback(const ull seed, const int8_t* restrict clause_output_arr, const NLITS_T* restrict clause_len,
                     const int8_t* restrict clause_drop_mask, const float* restrict prob,
                     const float* restrict label_probs, const int e, const float* restrict clause_weights,
                     uint8_t* restrict feedback_type, uint* restrict fb_count, uint* restrict fb_ids) {
    const float* label_probs_e = &label_probs[(ull)e * CLASSES];

#pragma omp parallel for schedule(dynamic) num_threads(ytm_n_threads)
    for (ull clause = 0; clause < (ull)TOTAL_CLAUSES; clause++) {
        int clause_output = clause_output_arr[clause];
        bool has_space = (clause_len[clause] <= (NLITS_T)MAX_INCLUDED_LITERALS);
        bool dropped = (clause_drop_mask[clause] == 1);
        bool any_fb = false;

        ull rng_k = rng_hash(seed, clause, 0xFEEDFACEULL);
        uint rng_counter = 0;

        ull class_id;
        LOOP_CLASS_ID(class_id, clause) {
            uint8_t fb = FB_NONE;

            if (!dropped) {
                int target = (prob[class_id] > 0.0f) - (prob[class_id] < 0.0f);

                if (!(rand_uniform(rng_k, &rng_counter) >= label_probs_e[class_id] || target == 0 ||
                      rand_uniform(rng_k, &rng_counter) >= fabsf(prob[class_id]))) {
                    float weight = clause_weights[weight_offset(class_id, clause)];
                    int sign = (weight >= 0) - (weight < 0);

                    if ((target * sign) > 0)
                        fb = (clause_output && has_space) ? FB_T1A : FB_T1B;
                    else if (clause_output)
                        fb = FB_T2;
                }
            }
            feedback_type[fbtype_offset(class_id, clause)] = fb;
            any_fb |= (fb != FB_NONE);
        }
        if (any_fb)
            fb_list_append(fb_count, fb_ids, clause);
    }
}

void update_clauses(const ull seed, const int8_t* restrict clause_output, const NPATCHES_T* restrict selected_patch_ids,
                    const FBOUND_T* restrict X, const int e, const NLITS_T* restrict literal_offsets,
                    const uint8_t* restrict feedback_type, TA_STATE_T* restrict global_ta_states,
                    int8_t* restrict is_clause_synced) {
    const FBOUND_T* Xe = &X[sample_offset(e)];

#pragma omp parallel for schedule(dynamic) num_threads(ytm_n_threads)
    for (ull clause = 0; clause < (ull)TOTAL_CLAUSES; clause++) {
        TA_STATE_T* ta_states = &global_ta_states[ta_offset(clause, 0)];

#if (N_PATCHES > 1)
        int patch_id = clause_output[clause] ? (int)selected_patch_ids[clause] : 0;
        int patch_idx_y = patch_id / N_PATCHES_X;
        int patch_idx_x = patch_id % N_PATCHES_X;
#else
        int patch_idx_y = 0, patch_idx_x = 0;
#endif

        ull rng_k = rng_hash(seed, clause, 0xCAFEBABEULL);
        uint rng_counter = 0;

        ull class_id;
        LOOP_CLASS_ID(class_id, clause) {
            uint8_t fb = feedback_type[fbtype_offset(class_id, clause)];
            if (fb == FB_NONE)
                continue;

            apply_feedback(rng_k, &rng_counter, fb, Xe, patch_idx_y, patch_idx_x, literal_offsets, ta_states, 0);
            is_clause_synced[clause] = 0;
        }
    }
}

void update_weights(const uint8_t* restrict feedback_type, float* restrict clause_weights) {
#if WEIGHTED
#pragma omp parallel for schedule(dynamic) num_threads(ytm_n_threads)
    for (ull clause = 0; clause < (ull)TOTAL_CLAUSES; clause++) {

        ull class_id;
        LOOP_CLASS_ID(class_id, clause) {
            uint8_t fb = feedback_type[fbtype_offset(class_id, clause)];
            if (fb == FB_T1B)
                continue;

            float* weight = &clause_weights[weight_offset(class_id, clause)];
            int sign = (*weight >= 0) - (*weight < 0);

            if (fb == FB_T1A) {
#if TYPE1A_FB
                float nw = *weight + sign * 1.0f;
                if (fabsf(nw) < MAX_WEIGHT)
                    *weight = nw;
#endif
            }

            else if (fb == FB_T2) {
#if TYPE2_FB
                float nw = *weight - sign * 1.0f;

#if NEGATIVE_CLAUSES == 0
                *weight = clip(nw, 1, MAX_WEIGHT);
#else

                if (fabsf(nw) < 1.0f) {
#if ALLOW_POLARITY_CHANGE
                    nw = -sign * 1.0f;
#else
                    nw = sign * 1.0f;
#endif
                }
                *weight = nw;
#endif

#endif
            }
        }
    }
#else
#endif
}

void calc_update_prob(const float* restrict votes, const float* restrict encoded_Y, const int e, float* restrict prob) {
    const float* encoded_Y_e = &encoded_Y[(ull)e * CLASSES];

#pragma omp parallel for schedule(static) num_threads(ytm_n_threads)
    for (int class_id = 0; class_id < CLASSES; class_id++) {
        float v = clip(votes[class_id], T_MIN, T_MAX);
        prob[class_id] = (encoded_Y_e[class_id] - v) / (T_MAX - T_MIN);
    }
}
