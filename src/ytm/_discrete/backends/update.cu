#ifdef IS_NEOVIM_CLANGD_ENV
#include "../../_core/backends/common.h"
#include "../../_core/backends/cuda.h"
#include "../../_core/backends/feedback.cu"
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
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int class_id = tid; class_id < CLASSES; class_id += stride) {
        float v = clip(votes[class_id], T_MIN, T_MAX);
        prob[class_id] = (encoded_Y_e[class_id] - v) / (T_MAX - T_MIN);
    }
}

extern "C" __global__ void decide_feedback(const ull seed, const int* selected_patch_ids, const int* clause_density,
                                           const int8_t* clause_drop_mask, const float* prob, const float* label_probs,
                                           const int e, const float* clause_weights, uint8_t* feedback_type) {
    const float* label_probs_e = &label_probs[(ull)e * CLASSES];
    ull tid = threadIdx.x + blockIdx.x * blockDim.x;
    ull stride = (ull)blockDim.x * gridDim.x;

    for (ull clause = tid; clause < (ull)TOTAL_CLAUSES; clause += stride) {
        ull rel_clause = clause % (ull)CLAUSES_PER_CLASS;
        int clause_output = (selected_patch_ids[clause] >= 0) ? 1 : 0;
        bool has_space = (clause_density[clause] <= (int)MAX_INCLUDED_LITERALS);
        bool dropped = (clause_drop_mask[clause] == 1);

        ull rng_k = rng_hash(seed, clause, 0xFEEDFACEULL);
        uint rng_counter = 0;

        ull class_id;
        LOOP_CLASS_ID(class_id, clause) {
            uint8_t fb = FB_NONE;

            if (!dropped) {
                int target = (prob[class_id] > 0.0f) - (prob[class_id] < 0.0f);

                if (!(rand_uniform(rng_k, &rng_counter) > label_probs_e[class_id] || target == 0 ||
                      rand_uniform(rng_k, &rng_counter) > fabsf(prob[class_id]))) {
                    float weight = clause_weights[class_id * (ull)CLAUSES_PER_CLASS + rel_clause];
                    int sign = (weight >= 0) - (weight < 0);

                    if ((target * sign) > 0)
                        fb = (clause_output && has_space) ? FB_T1A : FB_T1B;
                    else if (clause_output)
                        fb = FB_T2;
                }
            }
            feedback_type[rel_clause * (ull)CLASSES + class_id] = fb;
        }
    }
}

// A whole warp owns one clause, so `fb` is warp uniform and `apply_feedback` can stride its lanes
// over the literals.
extern "C" __global__ void update_clauses(const ull seed, const int* selected_patch_ids, const int* X, const int e,
                                          const int* feat_mins, const int* literal_offsets,
                                          const uint8_t* feedback_type, TA_STATE_T* global_ta_states,
                                          int8_t* is_clause_synced) {
    auto warp = cg::tiled_partition<WARP_SIZE>(cg::this_thread_block());
    auto grid = cg::this_grid();
    int lane = warp.thread_rank();
    ull warp_id = grid.thread_rank() / warp.size();
    ull total_warps = grid.size() / warp.size();

    const int* Xe = &X[(ull)e * HEIGHT * WIDTH * DEPTH];

    for (ull clause = warp_id; clause < (ull)TOTAL_CLAUSES; clause += total_warps) {
        ull rel_clause = clause % (ull)CLAUSES_PER_CLASS;
        TA_STATE_T* ta_states = &global_ta_states[clause * (ull)N_LITERALS];

        int patch_id = selected_patch_ids[clause];
        int patch_idx_y = (patch_id >= 0) ? patch_id / N_PATCHES_X : -1;
        int patch_idx_x = (patch_id >= 0) ? patch_id % N_PATCHES_X : -1;

        ull rng_k = rng_hash(seed, clause * (ull)WARP_SIZE + (ull)lane, 0xCAFEBABEULL);
        uint rng_counter = 0;

        ull class_id;
        LOOP_CLASS_ID(class_id, clause) {
            uint8_t fb = feedback_type[rel_clause * (ull)CLASSES + class_id];
            if (fb == FB_NONE)
                continue;

            apply_feedback(rng_k, &rng_counter, fb, Xe, patch_idx_y, patch_idx_x, feat_mins, literal_offsets,
                           ta_states, lane);
            if (lane == 0)
                is_clause_synced[clause] = 0;
        }
    }
}

extern "C" __global__ void update_weights(const uint8_t* feedback_type, float* clause_weights) {
#if WEIGHTED
    ull tid = threadIdx.x + blockIdx.x * blockDim.x;
    ull stride = (ull)blockDim.x * gridDim.x;

    for (ull clause = tid; clause < (ull)TOTAL_CLAUSES; clause += stride) {
        ull rel_clause = clause % (ull)CLAUSES_PER_CLASS;

        ull class_id;
        LOOP_CLASS_ID(class_id, clause) {
            uint8_t fb = feedback_type[rel_clause * (ull)CLASSES + class_id];
            if (fb == FB_T1B)
                continue;

            float* weight = &clause_weights[class_id * (ull)CLAUSES_PER_CLASS + rel_clause];
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
