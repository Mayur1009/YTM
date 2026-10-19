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
                                           uint* t1b_n, uint* t1b_cid, NCLASSES_T* t1b_cnt, uint* t1a_t2_n,
                                           uint* t1a_t2_cid, NCLASSES_T* t1a_cnt, NCLASSES_T* t2_cnt) {
    const float* label_probs_e = &label_probs[(ull)e * CLASSES];
    GRID_STRIDE_LOOP(clause, (ull)TOTAL_CLAUSES) {
        int clause_output = clause_output_arr[clause];
        bool has_space = (clause_len[clause] <= (NLITS_T)MAX_INCLUDED_LITERALS);
        bool dropped = (clause_drop_mask[clause] == 1);
        NCLASSES_T n1a = 0, n1b = 0, n2 = 0;

        ull rng_k = rng_hash(seed, clause, 0xFEEDFACEULL);
        uint rng_counter = 0;

        ull class_id;
        LOOP_CLASS_ID(class_id, clause) {
            uint8_t fb = FB_NONE;

            if (!dropped) {
                int target = (prob[class_id] > 0.0f) - (prob[class_id] < 0.0f);

                if (bernoulli(rng_k, &rng_counter, label_probs_e[class_id]) && target != 0 &&
                    bernoulli(rng_k, &rng_counter, fabsf(prob[class_id]))) {
                    float weight = clause_weights[weight_offset(class_id, clause)];
                    int sign = (weight >= 0) - (weight < 0);

                    if ((target * sign) > 0)
                        fb = (clause_output && has_space) ? FB_T1A : FB_T1B;
                    else if (clause_output)
                        fb = FB_T2;
                }
            }
            feedback_type[fbtype_offset(class_id, clause)] = fb;
            n1a += (fb == FB_T1A);
            n1b += (fb == FB_T1B);
            n2 += (fb == FB_T2);
        }
        append_fb_lists(clause, n1a, n1b, n2, t1b_n, t1b_cid, t1b_cnt, t1a_t2_n, t1a_t2_cid, t1a_cnt, t2_cnt);
    }
}

extern "C" __global__ void update_weights(const uint8_t* feedback_type, const uint* t1a_t2_n, const uint* t1a_t2_cid,
                                          float* clause_weights) {
    const uint n = *t1a_t2_n;

    GRID_STRIDE_LOOP(i, n) {
        ull clause = t1a_t2_cid[i];

        ull class_id;
        LOOP_CLASS_ID(class_id, clause) {
            uint8_t fb = feedback_type[fbtype_offset(class_id, clause)];
            if (fb == FB_T1B)
                continue;

            float* weight = &clause_weights[weight_offset(class_id, clause)];
            int sign = (*weight >= 0) - (*weight < 0);

            if (fb == FB_T1A && TYPE1A_FB) {
                float nw = *weight + sign * 1.0f;
                if (fabsf(nw) < MAX_WEIGHT)
                    *weight = nw;
            } else if (fb == FB_T2 && TYPE2_FB) {
                float nw = *weight - sign * 1.0f;
                if (!NEGATIVE_CLAUSES) {
                    *weight = clip(nw, 1.0f, MAX_WEIGHT);
                } else {
                    if (fabsf(nw) < 1.0f)
                        nw = ALLOW_POLARITY_CHANGE ? -sign * 1.0f : sign * 1.0f;
                    *weight = nw;
                }
            }
        }
    }
}
