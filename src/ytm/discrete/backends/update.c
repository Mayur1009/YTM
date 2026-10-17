#ifdef IS_NEOVIM_CLANGD_ENV
#include "../../_core/backends/common.h"
#include "../../_core/backends/feedback.h"

#define T_MIN -100.0f
#define T_MAX 100.0f
#endif

/*
 * Update probability of each class for the current sample.
 *
 * prob = (target - clip(votes, T_MIN, T_MAX)) / (T_MAX - T_MIN), in [-1, 1]. The sign is the direction (raise or
 * lower the class sum). For calssification this is the same as the labels for the sample,
 * (+ve -> label 1, -ve -> label 0).
 *
 * Arguments:
 *   votes      class sums of the current sample
 *   encoded_Y  target class sums of the batch, T_MAX for positive and T_MIN for negative labels
 *   e          index of the current sample
 *   prob       output, update probability per class
 */
void calc_update_prob(const float* restrict votes, const float* restrict encoded_Y, const int e, float* restrict prob) {
    const float* encoded_Y_e = &encoded_Y[(ull)e * CLASSES];

#pragma omp parallel for schedule(static) num_threads(ytm_n_threads)
    for (int class_id = 0; class_id < CLASSES; class_id++) {
        float v = clip(votes[class_id], T_MIN, T_MAX);
        prob[class_id] = (encoded_Y_e[class_id] - v) / (T_MAX - T_MIN);
    }
}

/*
 * Decide each clause's feedback for the current sample and put it on the feedback lists.
 * The type of feedback depends on the label and the polarity of the clause.
 *
 * Arguments:
 *   seed, e            RNG key and index of the current sample
 *   clause_output_arr  clause outputs of the current sample
 *   clause_len         number of included literals per clause
 *   clause_drop_mask   1 for clauses dropped this epoch
 *   prob               update probability per class, from calc_update_prob
 *   label_probs        per sample, per class chance to take part (label sampling)
 *   clause_weights     clause weights, their sign is the clause polarity per class
 *   feedback_type      output, feedback type per class and clause
 *   t1b_*, t1a_t2_*    output, the feedback lists
 */
void decide_feedback(const ull seed, const int8_t* restrict clause_output_arr, const NLITS_T* restrict clause_len,
                     const int8_t* restrict clause_drop_mask, const float* restrict prob,
                     const float* restrict label_probs, const int e, const float* restrict clause_weights,
                     uint8_t* restrict feedback_type, uint* restrict t1b_n, uint* restrict t1b_cid,
                     NCLASSES_T* restrict t1b_cnt, uint* restrict t1a_t2_n, uint* restrict t1a_t2_cid,
                     NCLASSES_T* restrict t1a_cnt, NCLASSES_T* restrict t2_cnt) {
    const float* label_probs_e = &label_probs[(ull)e * CLASSES];

#pragma omp parallel for schedule(dynamic) num_threads(ytm_n_threads)
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

                if (target != 0 && rand_uniform(rng_k, &rng_counter) < label_probs_e[class_id] &&
                    rand_uniform(rng_k, &rng_counter) < fabsf(prob[class_id])) {
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
        // One list entry per clause, with how many classes gave each type.
        fb_lists_append(clause, n1a, n1b, n2, t1b_n, t1b_cid, t1b_cnt, t1a_t2_n, t1a_t2_cid, t1a_cnt, t2_cnt);
    }
}

/*
 * Update the clause weights from the per class feedback of the current sample.
 *
 * Only T1a and T2 change weights, thus only T1a/T2 lists are needed. T1a moves the weight away from 0. T2 moves it
 * toward 0. |weight| >= +1. When ALLOW_POLARITY_CHANGE = True, the weight directly changes from -1 to +1 or +1 to -1
 * when flipping the polarity.
 *
 * Arguments:
 *   feedback_type   feedback type per class and clause, from decide_feedback
 *   t1a_t2_n        T1a/T2 list length
 *   t1a_t2_cid      T1a/T2 list clause ids
 *   clause_weights  clause weights per class, updated in place
 */
void update_weights(const uint8_t* restrict feedback_type, const uint* restrict t1a_t2_n,
                    const uint* restrict t1a_t2_cid, float* restrict clause_weights) {
    const uint n = *t1a_t2_n;
    if (n == 0)
        return;

#pragma omp parallel for schedule(dynamic) num_threads(ytm_n_threads)
    for (ull i = 0; i < n; i++) {
        ull clause = t1a_t2_cid[i];

        ull class_id;
        LOOP_CLASS_ID(class_id, clause) {
            uint8_t fb = feedback_type[fbtype_offset(class_id, clause)];
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
                    // Reached 0: flip polarity or keep the clause at its smallest weight.
                    if (fabsf(nw) < 1.0f)
                        nw = ALLOW_POLARITY_CHANGE ? -sign * 1.0f : sign * 1.0f;
                    *weight = nw;
                }
            }
        }
    }
}
