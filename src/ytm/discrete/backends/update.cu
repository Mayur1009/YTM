#ifdef IS_NEOVIM_CLANGD_ENV
#include "../../_core/backends/common.h"
#include "../../_core/backends/cuda.h"
#include "../../_core/backends/feedback.h"
#include "../../_core/backends/feedback.h"
#include "../../_core/backends/rng.h"

#define T_MIN -100.0f
#define T_MAX 100.0f
#define FB_NONE 0
#define FB_T1A 1
#define FB_T1B 2
#define FB_T2 3
#endif

extern "C" __global__ void calc_update_prob(const float* votes, const float* encoded_Y, const int e, float* prob) {
    const float* encoded_Y_e = &encoded_Y[(ull)e * CLASSES];
    GRID_STRIDE_LOOP(class_id, CLASSES) {
        float v = clip(votes[class_id], T_MIN, T_MAX);
        prob[class_id] = (encoded_Y_e[class_id] - v) / (T_MAX - T_MIN);
    }
}

extern "C" __global__ void decide_feedback(const ull seed, const int8_t* clause_output_arr, const NLITS_T* clause_len,
                                           const int8_t* clause_drop_mask, const float* prob, const float* label_probs,
                                           const int e, const float* clause_weights, uint8_t* feedback_type,
                                           uint* fb_count, uint* fb_ids) {
    const float* label_probs_e = &label_probs[(ull)e * CLASSES];
    GRID_STRIDE_LOOP(clause, (ull)TOTAL_CLAUSES) {
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

// A whole warp owns one clause, so `fb` is warp uniform and `apply_feedback` can stride its lanes
// over the literals.
extern "C" __global__ void update_clauses(const ull seed, const int8_t* clause_output, const NPATCHES_T* selected_patch_ids,
                                          const FBOUND_T* X, const int e, const NLITS_T* literal_offsets,
                                          const uint8_t* feedback_type, TA_STATE_T* global_ta_states,
                                          int8_t* is_clause_synced) {
    auto [warp, lane, warp_id, total_warps] = warp_grid();

    const FBOUND_T* Xe = &X[sample_offset(e)];

    WARP_STRIDE_LOOP(clause, (ull)TOTAL_CLAUSES) {
        TA_STATE_T* ta_states = &global_ta_states[ta_offset(clause, 0)];

#if (N_PATCHES > 1)
        int patch_id = clause_output[clause] ? (int)selected_patch_ids[clause] : 0;
        int patch_idx_y = patch_id / N_PATCHES_X;
        int patch_idx_x = patch_id % N_PATCHES_X;
#else
        int patch_idx_y = 0, patch_idx_x = 0;
#endif

        ull rng_k = rng_hash(seed, clause * (ull)WARP_SIZE + (ull)lane, 0xCAFEBABEULL);
        uint rng_counter = 0;

        ull class_id;
        LOOP_CLASS_ID(class_id, clause) {
            uint8_t fb = feedback_type[fbtype_offset(class_id, clause)];
            if (fb == FB_NONE)
                continue;

            apply_feedback(rng_k, &rng_counter, fb, Xe, patch_idx_y, patch_idx_x, literal_offsets, ta_states, lane);
            if (lane == 0)
                is_clause_synced[clause] = 0;
        }
    }
}

extern "C" __global__ void update_weights(const uint8_t* feedback_type, float* clause_weights) {
#if WEIGHTED
    GRID_STRIDE_LOOP(clause, (ull)TOTAL_CLAUSES) {

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
                *weight = clip(nw, 1.0f, MAX_WEIGHT);
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
#endif
}
