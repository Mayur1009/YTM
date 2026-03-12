#ifdef IS_NEOVIM_CLANGD_ENV
#define USE_OMP 1
#define TOTAL_CLAUSES 1000
#define THRESH 100
#define S 10.0
#define CLASSES 10
#define HEIGHT 28
#define WIDTH 28
#define DEPTH 1
#define PATCH_HEIGHT 10
#define PATCH_WIDTH 10
#define STRIDE_Y 1
#define STRIDE_X 1
#define NEGATED_LITERALS 1
#define POSITION_LITERALS 1
#define COALESCED 1
#define WEIGHTED 1
#define MAX_WEIGHT 10.0f
#define NEGATIVE_CLAUSES 1
#define ALLOW_POLARITY_CHANGE 1
#define MAX_INCLUDED_LITERALS 100
#define INCLUDE_STATE 128
#define MAX_TA_STATE 255
#define TYPE1A_FB 1
#define TYPE1B_FB 1
#define TYPE2_FB 1
#define N_RAW_PATCH_FEATS 100
#define N_PATCH_FEATS 100
#define N_POSITION_FEATS 36
#define N_PATCHES_Y 19
#define N_PATCHES_X 19
#define N_PATCHES 361
#define N_LITERALS 272
#endif

#define N_POSITION_FEATS_Y (N_PATCHES_Y - 1)
#define N_POSITION_FEATS_X (N_PATCHES_X - 1)
#define INT_SIZE 32
#define S_INV (1.0f / S)

#if COALESCED == 0
#define CLAUSES_PER_CLASS (TOTAL_CLAUSES / CLASSES)
#define LOOP_CLASS_ID(class_id, clause) class_id = (ull)clause / (CLAUSES_PER_CLASS);
#else
#define CLAUSES_PER_CLASS TOTAL_CLAUSES
#define LOOP_CLASS_ID(class_id, clause) for (class_id = 0; class_id < CLASSES; ++class_id)
#endif

#define CLIP(val, min, max) ((val < min) ? min : ((val > max) ? max : val))

#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if USE_OMP
#include <omp.h>
#define OMP_PARALLEL_FOR _Pragma("omp parallel for schedule(static)")
#define OMP_ATOMIC _Pragma("omp atomic")
#define OMP_SIMD _Pragma("omp simd")
#define GET_THREAD_ID omp_get_thread_num()
void set_num_threads(int num_threads) { omp_set_num_threads(num_threads); }
#else
#define OMP_PARALLEL_FOR
#define OMP_ATOMIC
#define OMP_SIMD
#define GET_THREAD_ID 0
#endif

typedef unsigned long long ull;
typedef unsigned int uint;

#define UINT_MAX_INV (1.0f / UINT_MAX)

static inline void max(int* a, int b) { *a = (*a > b) ? *a : b; }
static inline void min(int* a, int b) { *a = (*a < b) ? *a : b; }

// Rng
static inline float xorshift32(uint* state) {
    uint x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return (float)x * UINT_MAX_INV;
}

// Sample from geometric distribution with probability p
// Returns the number of trials until first success (1-indexed)
static inline int geometric_sample(uint* rng, float p) {
    float u = xorshift32(rng);
    if (u >= 1.0f)
        u = 0.9999999f;
    return (int)(logf(1.0f - u) / logf(1.0f - p)) + 1;
}

// Literal decrement with prob.
static inline void literal_dec_with_p(uint* restrict rng, uint* restrict ta_state, int start, int end, int offset,
                                      float p) {
    int li = start + geometric_sample(rng, p) - 1;
    while (li < end) {
        if (ta_state[li + offset] > 0)
            ta_state[li + offset] -= 1;
        li += geometric_sample(rng, p);
    }
}

// Literal increment
static inline void literal_inc(uint* restrict ta_state, int start, int end, int offset, uint max_val) {
    OMP_SIMD
    for (int li = start; li < end; ++li) {
        ta_state[li + offset] += (ta_state[li + offset] < max_val);
    }
}

// Update prob function - v is the clipped class sum and y it the target threshold.
static inline float uprob_fun(float v, float y) {
    float prob = (y - v) / (2 * y);
    return prob;
}

// Get feature value from X at a given patch position
static inline int32_t get_feature_value(const int32_t* X, int patch_idx_y, int patch_idx_x, int fid) {
    ull rel_y = fid / (PATCH_WIDTH * DEPTH);
    ull rel_x = (fid / DEPTH) % PATCH_WIDTH;
    ull z = fid % DEPTH;
    ull abs_y = patch_idx_y * STRIDE_Y + rel_y;
    ull abs_x = patch_idx_x * STRIDE_X + rel_x;
    return X[abs_y * (WIDTH * DEPTH) + abs_x * DEPTH + z];
}

// ============================================================================
// Feedback functions
// ============================================================================

// Type 1a feedback - reinforce matching literals
static inline void type1a_fb(uint* restrict rng, uint* restrict ta_state, float* restrict weight, const int32_t* X,
                             int patch_idx_y, int patch_idx_x, const int sign, const int* feat_mins,
                             const int* literal_offsets) {
#if TYPE1A_FB
#if WEIGHTED
    if (fabs(*weight) < MAX_WEIGHT)
        (*weight) += sign * 1.0f;
#endif

#if POSITION_LITERALS
    // Position Y literals: [0, patch_idx_y) have value 1, [patch_idx_y, N_POSITION_FEATS_Y) have value 0
    literal_inc(ta_state, 0, patch_idx_y, 0, MAX_TA_STATE);
    literal_dec_with_p(rng, ta_state, patch_idx_y, N_POSITION_FEATS_Y, 0, S_INV);

    // Position X literals: [0, patch_idx_x) have value 1, [patch_idx_x, N_POSITION_FEATS_X) have value 0
    literal_inc(ta_state, N_POSITION_FEATS_Y, N_POSITION_FEATS_Y + patch_idx_x, 0, MAX_TA_STATE);
    literal_dec_with_p(rng, ta_state, N_POSITION_FEATS_Y + patch_idx_x, N_POSITION_FEATS, 0, S_INV);

#if NEGATED_LITERALS
    // Negated position Y: [0, patch_idx_y) have value 0, [patch_idx_y, N_POSITION_FEATS_Y) have value 1
    literal_dec_with_p(rng, ta_state, 0, patch_idx_y, N_LITERALS / 2, S_INV);
    literal_inc(ta_state, patch_idx_y, N_POSITION_FEATS_Y, N_LITERALS / 2, MAX_TA_STATE);

    // Negated position X: [0, patch_idx_x) have value 0, [patch_idx_x, N_POSITION_FEATS_X) have value 1
    literal_dec_with_p(rng, ta_state, N_POSITION_FEATS_Y, N_POSITION_FEATS_Y + patch_idx_x, N_LITERALS / 2, S_INV);
    literal_inc(ta_state, N_POSITION_FEATS_Y + patch_idx_x, N_POSITION_FEATS, N_LITERALS / 2, MAX_TA_STATE);
#endif
#endif

    // Feature literals with thermometer encoding
    for (int fid = 0; fid < N_RAW_PATCH_FEATS; ++fid) {
        int lit_start = N_POSITION_FEATS + literal_offsets[fid];
        int lit_end = N_POSITION_FEATS + literal_offsets[fid + 1];

        int32_t val = get_feature_value(X, patch_idx_y, patch_idx_x, fid);
        int shifted_val = val - feat_mins[fid];

        // Positive literals: bits [0, shifted_val) are 1, [shifted_val, n_bits) are 0
        literal_inc(ta_state, lit_start, lit_start + shifted_val, 0, MAX_TA_STATE);
        literal_dec_with_p(rng, ta_state, lit_start + shifted_val, lit_end, 0, S_INV);

#if NEGATED_LITERALS
        // Negated: bits [0, shifted_val) are 0, [shifted_val, n_bits) are 1
        literal_dec_with_p(rng, ta_state, lit_start, lit_start + shifted_val, N_LITERALS / 2, S_INV);
        literal_inc(ta_state, lit_start + shifted_val, lit_end, N_LITERALS / 2, MAX_TA_STATE);
#endif
    }
#endif
}

// Type 1b feedback - decrement all literals with 1 / s
static inline void type1b_fb(uint* restrict rng, uint* restrict ta_state, const int sign) {
#if TYPE1B_FB
    literal_dec_with_p(rng, ta_state, 0, N_LITERALS, 0, S_INV);
#endif
}

// Type 2 feedback - increment excluded literals in the clause.
static inline void type2_fb(uint* restrict ta_state, float* restrict weight, const int32_t* X, int patch_idx_y,
                            int patch_idx_x, const int sign, const int* feat_mins, const int* literal_offsets) {
#if TYPE2_FB
#if WEIGHTED
    if (fabs(*weight) < MAX_WEIGHT)
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

#if POSITION_LITERALS
    // Position Y literals: [patch_idx_y, N_POSITION_FEATS_Y) have value 0
    literal_inc(ta_state, patch_idx_y, N_POSITION_FEATS_Y, 0, INCLUDE_STATE);

    // Position X literals: [patch_idx_x, N_POSITION_FEATS_X) have value 0
    literal_inc(ta_state, N_POSITION_FEATS_Y + patch_idx_x, N_POSITION_FEATS, 0, INCLUDE_STATE);

#if NEGATED_LITERALS
    // Negated position Y: [0, patch_idx_y) have value 0
    literal_inc(ta_state, 0, patch_idx_y, N_LITERALS / 2, INCLUDE_STATE);

    // Negated position X: [0, patch_idx_x) have value 0
    literal_inc(ta_state, N_POSITION_FEATS_Y, N_POSITION_FEATS_Y + patch_idx_x, N_LITERALS / 2, INCLUDE_STATE);
#endif
#endif

    // Feature literals with thermometer encoding
    for (int fid = 0; fid < N_RAW_PATCH_FEATS; ++fid) {
        int lit_start = N_POSITION_FEATS + literal_offsets[fid];
        int lit_end = N_POSITION_FEATS + literal_offsets[fid + 1];

        int32_t val = get_feature_value(X, patch_idx_y, patch_idx_x, fid);
        int shifted_val = val - feat_mins[fid];

        // Positive literals: increment where value is 0, i.e., [shifted_val, n_bits)
        literal_inc(ta_state, lit_start + shifted_val, lit_end, 0, INCLUDE_STATE);

#if NEGATED_LITERALS
        // Negated: increment where value is 0, i.e., [0, shifted_val)
        literal_inc(ta_state, lit_start, lit_start + shifted_val, N_LITERALS / 2, INCLUDE_STATE);
#endif
    }
#endif
}

// ============================================================================
// Clause evaluation and update
// ============================================================================

void pack_clauses(const uint* restrict global_ta_states, const int* restrict literal_offsets,
                  int* restrict clause_positions, int* restrict included_lits_pos, int* restrict included_lits_neg,
                  int* restrict n_lits_pos, int* restrict n_lits_neg, uint* restrict num_includes,
                  int8_t* restrict clause_dirty) {
    OMP_PARALLEL_FOR
    for (ull clause = 0; clause < TOTAL_CLAUSES; clause++) {
        if (!clause_dirty[clause])
            continue;

        const uint* ta_state = &global_ta_states[clause * N_LITERALS];
        int* pos = &clause_positions[clause * 4];
        int* lits_pos = &included_lits_pos[clause * N_PATCH_FEATS];
        int* lits_neg = &included_lits_neg[clause * N_PATCH_FEATS];

        // Initialize position bounds
        pos[0] = 0;           // min_row
        pos[1] = N_PATCHES_Y; // max_row
        pos[2] = 0;           // min_col
        pos[3] = N_PATCHES_X; // max_col

        int local_n_pos = 0;
        int local_n_neg = 0;
        uint total_includes = 0;

#if POSITION_LITERALS
        // Scan Y position literals
        for (int lit = 0; lit < N_POSITION_FEATS_Y; ++lit) {
            if (ta_state[lit] >= INCLUDE_STATE) {
                max(&pos[0], lit + 1);
                total_includes++;
            }
#if NEGATED_LITERALS
            if (ta_state[lit + N_LITERALS / 2] >= INCLUDE_STATE) {
                min(&pos[1], lit + 1);
                total_includes++;
            }
#endif
        }

        // Scan X position literals
        for (int lit = N_POSITION_FEATS_Y; lit < N_POSITION_FEATS; ++lit) {
            int x_lit = lit - N_POSITION_FEATS_Y;
            if (ta_state[lit] >= INCLUDE_STATE) {
                max(&pos[2], x_lit + 1);
                total_includes++;
            }
#if NEGATED_LITERALS
            if (ta_state[lit + N_LITERALS / 2] >= INCLUDE_STATE) {
                min(&pos[3], x_lit + 1);
                total_includes++;
            }
#endif
        }
#endif

        // Scan feature literals - store indices
        for (int lit_idx = 0; lit_idx < N_PATCH_FEATS; ++lit_idx) {
            int lit = N_POSITION_FEATS + lit_idx;
            if (ta_state[lit] >= INCLUDE_STATE) {
                lits_pos[local_n_pos++] = lit_idx;
                total_includes++;
            }
#if NEGATED_LITERALS
            if (ta_state[lit + N_LITERALS / 2] >= INCLUDE_STATE) {
                lits_neg[local_n_neg++] = lit_idx;
                total_includes++;
            }
#endif
        }

        n_lits_pos[clause] = local_n_pos;
        n_lits_neg[clause] = local_n_neg;
        num_includes[clause] = total_includes;

        clause_dirty[clause] = 0;
    }
}

void eval_clauses(uint* restrict rng, const int32_t* restrict X, const int8_t* restrict clause_drop_mask,
                  int* restrict selected_patch_ids, const uint* restrict num_includes, const int* restrict feat_mins,
                  const int* restrict literal_offsets, const int* restrict clause_positions,
                  const int* restrict included_lits_pos, const int* restrict n_lits_pos, const int* included_lits_neg,
                  const int* restrict n_lits_neg, const int* restrict lit_to_fid) {
    OMP_PARALLEL_FOR
    for (ull clause = 0; clause < TOTAL_CLAUSES; clause++) {
        // Skip dropped clauses
        if (clause_drop_mask[clause] == 1) {
            selected_patch_ids[clause] = -1;
            continue;
        }

        const int* pos = &clause_positions[clause * 4];

        if (pos[0] >= pos[1] || pos[2] >= pos[3]) {
            // Clause has contradiction
            selected_patch_ids[clause] = -1;
            continue;
        }

        if (num_includes[clause] == 0) {
            // Empty clause: randomly select a patch
            selected_patch_ids[clause] = (int)(xorshift32(&rng[GET_THREAD_ID]) * N_PATCHES);
            continue;
        }

        int selected_patch = -1;
        int active_patch_count = 0;

        for (int patch_idx_y = pos[0]; patch_idx_y < pos[1]; patch_idx_y++) {
            for (int patch_idx_x = pos[2]; patch_idx_x < pos[3]; patch_idx_x++) {
                const int* lits_pos = &included_lits_pos[clause * N_PATCH_FEATS];
                const int clause_n_lits_pos = n_lits_pos[clause];
                bool matches = true;

                for (int i = 0; matches && i < clause_n_lits_pos; ++i) {
                    int lit_idx = lits_pos[i];
                    int fid = lit_to_fid[lit_idx];
                    int bit = lit_idx - literal_offsets[fid];
                    int shifted_val = get_feature_value(X, patch_idx_y, patch_idx_x, fid) - feat_mins[fid];
                    if (shifted_val < bit + 1)
                        matches = false;
                }
#if NEGATED_LITERALS
                const int* lits_neg = &included_lits_neg[clause * N_PATCH_FEATS];
                const int clause_n_lits_neg = n_lits_neg[clause];
                for (int i = 0; matches && i < clause_n_lits_neg; ++i) {
                    int lit_idx = lits_neg[i];
                    int fid = lit_to_fid[lit_idx];
                    int bit = lit_idx - literal_offsets[fid];
                    int shifted_val = get_feature_value(X, patch_idx_y, patch_idx_x, fid) - feat_mins[fid];
                    if (shifted_val > bit)
                        matches = false;
                }
#endif

                if (matches) {
                    active_patch_count++;
                    if (xorshift32(&rng[GET_THREAD_ID]) < 1.0f / active_patch_count) {
                        selected_patch = patch_idx_y * N_PATCHES_X + patch_idx_x;
                    }
                }
            }
        }

        selected_patch_ids[clause] = selected_patch;
    }
}

void update_clauses(uint* restrict rng, const int* restrict selected_patch_ids,
                    const uint* restrict clause_num_includes, uint* restrict global_ta_states,
                    float* restrict clause_weights, const int8_t* restrict clause_drop_mask, const int32_t* restrict X,
                    const int8_t* restrict targets, const float* restrict prob, const int* restrict feat_mins,
                    const int* restrict literal_offsets, int8_t* restrict clause_dirty) {
    for (ull clause = 0; clause < TOTAL_CLAUSES; clause++) {
        // Skip dropped clauses
        if (clause_drop_mask[clause] == 1)
            continue;

        uint* ta_state = &global_ta_states[clause * N_LITERALS];
        int local_clause_output = selected_patch_ids[clause] > -1 ? 1 : 0;

        // Get patch coordinates if clause was active
        int patch_idx_y = -1, patch_idx_x = -1;
        if (local_clause_output) {
            patch_idx_y = selected_patch_ids[clause] / N_PATCHES_X;
            patch_idx_x = selected_patch_ids[clause] % N_PATCHES_X;
        }

        uint num_includes = clause_num_includes[clause];

        ull class_id, rel_clause = clause % CLAUSES_PER_CLASS;
        LOOP_CLASS_ID(class_id, clause) {
            int local_target = targets[class_id];
            if (local_target == 0)
                continue;

            float* local_weight = &clause_weights[class_id * CLAUSES_PER_CLASS + rel_clause];
            int sign = (*local_weight >= 0) - (*local_weight < 0);

            float update_prob = prob[class_id];
            bool should_update = (xorshift32(&rng[GET_THREAD_ID]) <= update_prob);
            bool clause_has_space = (num_includes <= (uint)MAX_INCLUDED_LITERALS);
            bool t1 = (local_target * sign) > 0;

            // Type 1a feedback - TP - clause is active with correct polarity and has space
            if (should_update && t1 && local_clause_output && clause_has_space) {
                type1a_fb(&rng[GET_THREAD_ID], ta_state, local_weight, X, patch_idx_y, patch_idx_x, sign, feat_mins,
                          literal_offsets);
                clause_dirty[clause] = 1;
            }

            // Type 1b feedback - FN - clause is inactive or overflowing, but should have been active
            if (should_update && t1 && !(local_clause_output && clause_has_space)) {
                type1b_fb(&rng[GET_THREAD_ID], ta_state, sign);
                clause_dirty[clause] = 1;
            }

            // Type 2 feedback - FP - clause is active but has wrong polarity
            if (should_update && (local_target * sign) < 0 && local_clause_output) {
                type2_fb(ta_state, local_weight, X, patch_idx_y, patch_idx_x, sign, feat_mins, literal_offsets);
                clause_dirty[clause] = 1;
            }
        }
    }
}

void fit_sample(uint* restrict rng, uint* restrict global_ta_states, float* restrict clause_weights,
                int* restrict patch_weights, const int8_t* restrict clause_drop_mask, const int32_t* restrict X,
                const int8_t* restrict targets, int e, const int* restrict feat_mins,
                const int* restrict literal_offsets, const int* restrict lit_to_fid, int* restrict clause_positions,
                int* restrict included_lits_pos, int* restrict included_lits_neg, int* restrict n_lits_pos,
                int* restrict n_lits_neg, uint* restrict num_includes, int8_t* restrict clause_dirty,
                int* restrict selected_patch_ids, float* restrict votes, float* restrict prob) {

    const int32_t* Xe = &X[(ull)e * HEIGHT * WIDTH * DEPTH];
    const int8_t* targets_sample = &targets[(ull)e * CLASSES];

    pack_clauses(global_ta_states, literal_offsets, clause_positions, included_lits_pos, included_lits_neg, n_lits_pos,
                 n_lits_neg, num_includes, clause_dirty);

    eval_clauses(rng, Xe, clause_drop_mask, selected_patch_ids, num_includes, feat_mins, literal_offsets,
                 clause_positions, included_lits_pos, n_lits_pos, included_lits_neg, n_lits_neg, lit_to_fid);

    memset(votes, 0, sizeof(float) * CLASSES);
    for (ull clause = 0; clause < TOTAL_CLAUSES; clause++) {
        if (selected_patch_ids[clause] != -1) {
            ull class_id, rel_clause = clause % CLAUSES_PER_CLASS;
            LOOP_CLASS_ID(class_id, clause) {
                OMP_ATOMIC
                votes[class_id] += clause_weights[class_id * CLAUSES_PER_CLASS + rel_clause];
            }
            patch_weights[clause * N_PATCHES + selected_patch_ids[clause]]++;
        }
    }

    OMP_PARALLEL_FOR
    for (ull class_id = 0; class_id < CLASSES; class_id++) {
        int local_target = targets_sample[class_id];
        if (local_target == 0) {
            prob[class_id] = 0.0f;
            continue;
        }

        float y = (float)THRESH * (float)local_target;
        float class_sum = (float)CLIP(votes[class_id], -THRESH, THRESH);
        prob[class_id] = uprob_fun(class_sum, y);
    }

    update_clauses(rng, selected_patch_ids, num_includes, global_ta_states, clause_weights, clause_drop_mask, Xe,
                   targets_sample, prob, feat_mins, literal_offsets, clause_dirty);
}

void infer_sample(const int32_t* restrict X, const float* restrict clause_weights, float* restrict class_sums, int e,
                  const int* restrict feat_mins, const int* restrict clause_positions, const int* included_lits_pos,
                  const int* included_lits_neg, const int* n_lits_pos, const int* n_lits_neg,
                  const uint* restrict num_includes, const int* restrict lit_to_fid,
                  const int* restrict literal_offsets) {

    float* class_sums_e = &class_sums[(ull)e * CLASSES];
    const int32_t* Xe = &X[(ull)e * HEIGHT * WIDTH * DEPTH];
    memset(class_sums_e, 0, sizeof(float) * CLASSES);

    OMP_PARALLEL_FOR
    for (ull clause = 0; clause < TOTAL_CLAUSES; clause++) {
        if (num_includes[clause] == 0) {
            // Skip empty clauses during inference
            continue;
        }
        const int* pos = &clause_positions[clause * 4];

        // Check if clause is valid (position bounds)
        if (pos[0] >= pos[1] || pos[2] >= pos[3])
            continue;

        const int* lits_pos = &included_lits_pos[clause * N_PATCH_FEATS];
        const int* lits_neg = &included_lits_neg[clause * N_PATCH_FEATS];
        const int clause_n_lits_pos = n_lits_pos[clause];
        const int clause_n_lits_neg = n_lits_neg[clause];

        bool matched = false;

        // Early exit on first matching patch
        for (int py = pos[0]; py < pos[1] && !matched; ++py) {
            for (int px = pos[2]; px < pos[3] && !matched; ++px) {
                bool match = true;

                // Check positive literals
                for (int i = 0; i < clause_n_lits_pos && match; ++i) {
                    int lit_idx = lits_pos[i];
                    int fid = lit_to_fid[lit_idx];
                    int bit = lit_idx - literal_offsets[fid];
                    int shifted_val = get_feature_value(Xe, py, px, fid) - feat_mins[fid];
                    if (shifted_val < bit + 1)
                        match = false;
                }
#if NEGATED_LITERALS
                // Check negated literals
                for (int i = 0; i < clause_n_lits_neg && match; ++i) {
                    int lit_idx = lits_neg[i];
                    int fid = lit_to_fid[lit_idx];
                    int bit = lit_idx - literal_offsets[fid];
                    int shifted_val = get_feature_value(Xe, py, px, fid) - feat_mins[fid];
                    if (shifted_val > bit)
                        match = false;
                }
#endif

                if (match)
                    matched = true;
            }
        }

        if (matched) {
            ull class_id, rel_clause = clause % CLAUSES_PER_CLASS;
            LOOP_CLASS_ID(class_id, clause) {
                OMP_ATOMIC
                class_sums_e[class_id] += clause_weights[class_id * CLAUSES_PER_CLASS + rel_clause];
            }
        }
    }
}

void eval_sample_patchwise(const int32_t* restrict X, const int* restrict feat_mins,
                           const int* restrict literal_offsets, const int* restrict lit_to_fid,
                           const int* restrict clause_positions, const int* restrict included_lits_pos,
                           const int* included_lits_neg, const int* restrict n_lits_pos, const int* n_lits_neg,
                           const uint* restrict num_includes, int8_t* restrict co_patchwise, int e) {

    int8_t* co_patchwise_e = &co_patchwise[(ull)e * TOTAL_CLAUSES * N_PATCHES];

    OMP_PARALLEL_FOR
    for (ull clause = 0; clause < TOTAL_CLAUSES; clause++) {
        if (num_includes[clause] == 0) {
            // Empty clauses match all patches
            memset(&co_patchwise_e[clause * N_PATCHES], 1, sizeof(int8_t) * N_PATCHES);
            continue;
        }

        const int* pos = &clause_positions[clause * 4];
        if (pos[0] >= pos[1] || pos[2] >= pos[3]) {
            // Clause has contradiction, it doesn not match anything, so let the patchwise output be 0
            continue;
        }

        const int* lits_pos = &included_lits_pos[clause * N_PATCH_FEATS];
        const int* lits_neg = &included_lits_neg[clause * N_PATCH_FEATS];
        const int clause_n_lits_pos = n_lits_pos[clause];
        const int clause_n_lits_neg = n_lits_neg[clause];

        for (int py = pos[0]; py < pos[1]; ++py) {
            for (int px = pos[2]; px < pos[3]; ++px) {
                bool match = true;

                // Check positive literals
                for (int i = 0; i < clause_n_lits_pos && match; ++i) {
                    int lit_idx = lits_pos[i];
                    int fid = lit_to_fid[lit_idx];
                    int bit = lit_idx - literal_offsets[fid];
                    int shifted_val = get_feature_value(X, py, px, fid) - feat_mins[fid];
                    if (shifted_val < bit + 1)
                        match = false;
                }
#if NEGATED_LITERALS
                // Check negated literals
                for (int i = 0; i < clause_n_lits_neg && match; ++i) {
                    int lit_idx = lits_neg[i];
                    int fid = lit_to_fid[lit_idx];
                    int bit = lit_idx - literal_offsets[fid];
                    int shifted_val = get_feature_value(X, py, px, fid) - feat_mins[fid];
                    if (shifted_val > bit)
                        match = false;
                }
#endif

                if (match) {
                    co_patchwise_e[clause * N_PATCHES + py * N_PATCHES_X + px] = 1;
                }
            }
        }
    }
}
