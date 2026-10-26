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

extern "C" __global__ void sample_classes(const ull* __restrict__ rng_keys, const float* __restrict__ label_probs, const int e,
                                          uint8_t* __restrict__ class_active) {
    const ull seed = rng_keys[e];
    const float* label_probs_e = &label_probs[(ull)e * CLASSES];
    GRID_STRIDE_LOOP(class_id, CLASSES) {
        uint rng_counter = 0;
        class_active[class_id] =
            bernoulli(rng_hash(seed, class_id, 0xC1A55C1AULL), &rng_counter, label_probs_e[class_id]);
    }
}

extern "C" __global__ void calc_update_prob(const float* __restrict__ votes, const float* __restrict__ encoded_Y,
                                            const int e, float* __restrict__ prob) {
    const float* encoded_Y_e = &encoded_Y[(ull)e * CLASSES];
    GRID_STRIDE_LOOP(class_id, CLASSES) {
        float v = clip(votes[class_id], T_MIN, T_MAX);
        prob[class_id] = (encoded_Y_e[class_id] - v) / (T_MAX - T_MIN);
    }
}

extern "C" __global__ void
decide_feedback(const ull* __restrict__ rng_keys, const int8_t* __restrict__ clause_output_arr, const NLITS_T* __restrict__ clause_len,
                const int8_t* __restrict__ clause_drop_mask, const float* __restrict__ prob,
                const float* __restrict__ label_probs, const uint8_t* __restrict__ class_active, const int e,
                const float* __restrict__ clause_weights, uint8_t* __restrict__ feedback_type, uint* __restrict__ t1b_n,
                uint* __restrict__ t1b_cid, NCLASSES_T* __restrict__ t1b_cnt, uint* __restrict__ t1a_t2_n,
                uint* __restrict__ t1a_t2_cid, NCLASSES_T* __restrict__ t1a_cnt, NCLASSES_T* __restrict__ t2_cnt) {
    const ull seed = rng_keys[e];
    const float* label_probs_e = &label_probs[(ull)e * CLASSES];
    GRID_STRIDE_LOOP(clause, (ull)TOTAL_CLAUSES) {
        int clause_output = clause_output_arr[clause];
        bool has_space = (clause_len[clause] < (NLITS_T)MAX_INCLUDED_LITERALS);
        bool dropped = (clause_drop_mask[clause] == 1);
        NCLASSES_T n1a = 0, n1b = 0, n2 = 0;

        ull rng_k = rng_hash(seed, clause, 0xFEEDFACEULL);
        uint rng_counter = 0;

        ull class_id;
        LOOP_CLASS_ID(class_id, clause) {
            uint8_t fb = FB_NONE;

            if (!dropped) {
                int target = (prob[class_id] > 0.0f) - (prob[class_id] < 0.0f);

                bool sampled, update;
                if (COALESCED) {
                    int2 d = bernoulli2(rng_k, &rng_counter, label_probs_e[class_id], fabsf(prob[class_id]));
                    sampled = d.x;
                    update = d.y;
                } else {
                    sampled = class_active[class_id];
                    update = sampled && target != 0 && bernoulli(rng_k, &rng_counter, fabsf(prob[class_id]));
                }
                if (sampled && target != 0 && update) {
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

extern "C" __global__ void update_weights(const uint8_t* __restrict__ feedback_type, const uint* __restrict__ t1a_t2_n,
                                          const uint* __restrict__ t1a_t2_cid, float* __restrict__ clause_weights) {
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
