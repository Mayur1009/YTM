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
                     const float* restrict label_probs, const uint8_t* restrict class_active, const int e,
                     const float* restrict clause_weights, uint8_t* restrict feedback_type, uint* restrict t1b_n,
                     uint* restrict t1b_cid, NCLASSES_T* restrict t1b_cnt, uint* restrict t1a_t2_n,
                     uint* restrict t1a_t2_cid, NCLASSES_T* restrict t1a_cnt, NCLASSES_T* restrict t2_cnt) {
    const float* label_probs_e = &label_probs[(ull)e * CLASSES];

#pragma omp parallel for schedule(static) num_threads(ytm_n_threads)
    for (ull clause = 0; clause < (ull)TOTAL_CLAUSES; clause++) {
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

void update_weights(const uint8_t* restrict feedback_type, const uint* restrict t1a_t2_n,
                    const uint* restrict t1a_t2_cid,
                    float* restrict clause_weights) {
    const uint n = *t1a_t2_n;
    if (n == 0)
        return;

#pragma omp parallel for schedule(static) num_threads(ytm_n_threads)
    for (ull i = 0; i < n; i++) {
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

void sample_classes(const ull seed, const float* restrict label_probs, const int e, uint8_t* restrict class_active) {
    const float* label_probs_e = &label_probs[(ull)e * CLASSES];
    for (int class_id = 0; class_id < CLASSES; class_id++) {
        uint rng_counter = 0;
        class_active[class_id] =
            bernoulli(rng_hash(seed, class_id, 0xC1A55C1AULL), &rng_counter, label_probs_e[class_id]);
    }
}

void calc_update_prob(const float* restrict votes, const float* restrict encoded_Y, const int e, float* restrict prob) {
    const float* encoded_Y_e = &encoded_Y[(ull)e * CLASSES];

#pragma omp parallel for schedule(static) num_threads(ytm_n_threads)
    for (int class_id = 0; class_id < CLASSES; class_id++) {
        float v = clip(votes[class_id], T_MIN, T_MAX);
        prob[class_id] = (encoded_Y_e[class_id] - v) / (T_MAX - T_MIN);
    }
}
