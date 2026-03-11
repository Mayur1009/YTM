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
    #define N_FEATURES 100
    #define N_FEAT_LITERALS 100
    #define N_POSITION_FEATS 36
    #define LITERALS 272
#endif

#define INT_SIZE 32
#define S_INV (1.0f / S)

#define N_PATCHES_Y (HEIGHT - PATCH_HEIGHT + 1)
#define N_PATCHES_X (WIDTH - PATCH_WIDTH + 1)
#define PATCHES (N_PATCHES_Y * N_PATCHES_X)

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
    if (u >= 1.0f) u = 0.9999999f;
    return (int)(logf(1.0f - u) / logf(1.0f - p)) + 1;
}

// Literal decrement with prob.
static inline void literal_dec_with_p(uint* restrict rng, uint* restrict ta_state, int start, int end, int offset,
                                      float p) {
    int li = start + geometric_sample(rng, p) - 1;
    while (li < end) {
        if (ta_state[li + offset] > 0) ta_state[li + offset] -= 1;
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

// Convert clauses to ranges. Assumes thermometer encoding.
static inline uint scan_clause(const uint* ta_state, const int* literal_offsets, int* clause_position,
                               int* valid_feat_range, bool* is_valid, int* interesting_fids, int* fid_len) {
    uint num_includes = 0;  // Count includes.
    *is_valid = true;       // Check for contradictions

    clause_position[0] = 0;            // min_row
    clause_position[1] = N_PATCHES_Y;  // max_row
    clause_position[2] = 0;            // min_col
    clause_position[3] = N_PATCHES_X;  // max_col

#if POSITION_LITERALS
    // Processing position literals if they exist.
    for (int lit = 0; lit < HEIGHT - PATCH_HEIGHT; ++lit) {
        if (ta_state[lit] >= INCLUDE_STATE) {
            max(&clause_position[0], lit + 1);
            num_includes++;
        }
    #if NEGATED_LITERALS
        if (ta_state[lit + LITERALS / 2] >= INCLUDE_STATE) {
            min(&clause_position[1], lit + 1);
            num_includes++;
        }
    #endif
    }

    for (int lit = HEIGHT - PATCH_HEIGHT; lit < N_POSITION_FEATS; ++lit) {
        if (ta_state[lit] >= INCLUDE_STATE) {
            int x_lit = lit - (HEIGHT - PATCH_HEIGHT);
            max(&clause_position[2], x_lit + 1);
            num_includes++;
        }
    #if NEGATED_LITERALS
        if (ta_state[lit + LITERALS / 2] >= INCLUDE_STATE) {
            int x_lit = lit - (HEIGHT - PATCH_HEIGHT);
            min(&clause_position[3], x_lit + 1);
            num_includes++;
        }
    #endif
    }

    if (clause_position[0] >= clause_position[1] || clause_position[2] >= clause_position[3]) *is_valid = false;
#endif

    *fid_len = 0;
    for (int fid = 0; fid < N_FEATURES; ++fid) {
        // Number of thermometer bits for this feature.
        int n_bits = literal_offsets[fid + 1] - literal_offsets[fid];
        // Valid range is [0, n_bits] inclusive, stored as closed [min, max]
        valid_feat_range[fid * 2] = 0;
        valid_feat_range[fid * 2 + 1] = n_bits;

        for (int bit = 0; bit < n_bits; ++bit) {
            int lit_pos = N_POSITION_FEATS + literal_offsets[fid] + bit;
            if (ta_state[lit_pos] >= INCLUDE_STATE) {
                // Positive literal at bit k: bit[k] = 1 means k < shifted_val
                // So shifted_val > k, i.e., shifted_val >= k + 1
                max(&valid_feat_range[fid * 2], bit + 1);
                num_includes++;
            }
#if NEGATED_LITERALS
            int lit_neg = lit_pos + LITERALS / 2;
            if (ta_state[lit_neg] >= INCLUDE_STATE) {
                // Negated literal at bit k: bit[k] = 0 means k >= shifted_val
                // So shifted_val <= k (closed interval)
                min(&valid_feat_range[fid * 2 + 1], bit);
                num_includes++;
            }
#endif
        }

        // Track features that have constraints
        if (valid_feat_range[fid * 2] > 0 || valid_feat_range[fid * 2 + 1] < n_bits) {
            interesting_fids[(*fid_len)++] = fid;
        }

        if (*is_valid && valid_feat_range[fid * 2] > valid_feat_range[fid * 2 + 1]) *is_valid = false;
    }

    return num_includes;
}

// Get feature value from X at a given patch position
static inline int32_t get_feature_value(const int32_t* X, int patch_row, int patch_col, int fid) {
    ull rel_y = fid / (PATCH_WIDTH * DEPTH);
    ull rel_x = (fid / DEPTH) % PATCH_WIDTH;
    ull z = fid % DEPTH;
    ull abs_y = patch_row + rel_y;
    ull abs_x = patch_col + rel_x;
    return X[abs_y * (WIDTH * DEPTH) + abs_x * DEPTH + z];
}

// ============================================================================
// Feedback functions
// ============================================================================

// Type 1a feedback - reinforce matching literals
static inline void type1a_fb(uint* restrict rng, uint* restrict ta_state, float* restrict weight, const int32_t* X,
                             int patch_row, int patch_col, const int sign, const int* feat_mins,
                             const int* literal_offsets) {
#if TYPE1A_FB
    #if WEIGHTED
    if (fabs(*weight) < MAX_WEIGHT) (*weight) += sign * 1.0f;
    #endif

    #if POSITION_LITERALS
    // Position Y literals: [0, patch_row) have value 1, [patch_row, HEIGHT-PATCH_HEIGHT) have value 0
    literal_inc(ta_state, 0, patch_row, 0, MAX_TA_STATE);
    literal_dec_with_p(rng, ta_state, patch_row, HEIGHT - PATCH_HEIGHT, 0, S_INV);

    // Position X literals: [0, patch_col) have value 1, [patch_col, WIDTH-PATCH_WIDTH) have value 0
    literal_inc(ta_state, HEIGHT - PATCH_HEIGHT, HEIGHT - PATCH_HEIGHT + patch_col, 0, MAX_TA_STATE);
    literal_dec_with_p(rng, ta_state, HEIGHT - PATCH_HEIGHT + patch_col, N_POSITION_FEATS, 0, S_INV);

        #if NEGATED_LITERALS
    // Negated position Y: [0, patch_row) have value 0, [patch_row, HEIGHT-PATCH_HEIGHT) have value 1
    literal_dec_with_p(rng, ta_state, 0, patch_row, LITERALS / 2, S_INV);
    literal_inc(ta_state, patch_row, HEIGHT - PATCH_HEIGHT, LITERALS / 2, MAX_TA_STATE);

    // Negated position X: [0, patch_col) have value 0, [patch_col, WIDTH-PATCH_WIDTH) have value 1
    literal_dec_with_p(rng, ta_state, HEIGHT - PATCH_HEIGHT, HEIGHT - PATCH_HEIGHT + patch_col, LITERALS / 2, S_INV);
    literal_inc(ta_state, HEIGHT - PATCH_HEIGHT + patch_col, N_POSITION_FEATS, LITERALS / 2, MAX_TA_STATE);
        #endif
    #endif

    // Feature literals with thermometer encoding
    for (int fid = 0; fid < N_FEATURES; ++fid) {
        int n_bits = literal_offsets[fid + 1] - literal_offsets[fid];
        int lit_start = N_POSITION_FEATS + literal_offsets[fid];
        int lit_end = N_POSITION_FEATS + literal_offsets[fid + 1];

        int32_t val = get_feature_value(X, patch_row, patch_col, fid);
        int shifted_val = val - feat_mins[fid];

        // Positive literals: bits [0, shifted_val) are 1, [shifted_val, n_bits) are 0
        literal_inc(ta_state, lit_start, lit_start + shifted_val, 0, MAX_TA_STATE);
        literal_dec_with_p(rng, ta_state, lit_start + shifted_val, lit_end, 0, S_INV);

    #if NEGATED_LITERALS
        // Negated: bits [0, shifted_val) are 0, [shifted_val, n_bits) are 1
        literal_dec_with_p(rng, ta_state, lit_start, lit_start + shifted_val, LITERALS / 2, S_INV);
        literal_inc(ta_state, lit_start + shifted_val, lit_end, LITERALS / 2, MAX_TA_STATE);
    #endif
    }
#endif
}

// Type 1b feedback - decrement all literals with 1 / s
static inline void type1b_fb(uint* restrict rng, uint* restrict ta_state, const int sign) {
#if TYPE1B_FB
    literal_dec_with_p(rng, ta_state, 0, LITERALS, 0, S_INV);
#endif
}

// Type 2 feedback - increment excluded literals in the clause.
static inline void type2_fb(uint* restrict ta_state, float* restrict weight, const int32_t* X, int patch_row,
                            int patch_col, const int sign, const int* feat_mins, const int* literal_offsets) {
#if TYPE2_FB
    #if WEIGHTED
    if (fabs(*weight) < MAX_WEIGHT) (*weight) -= sign * 1.0f;
        #if ALLOW_POLARITY_CHANGE == 0
    if (sign == 1 && *weight < 0) *weight = 1;
    if (sign == -1 && *weight >= 0) *weight = -1;
        #endif
    #endif
    #if NEGATIVE_CLAUSES == 0
    if (*weight < 1) *weight = 1;
    #endif

    #if POSITION_LITERALS
    // Position Y literals: [patch_row, HEIGHT-PATCH_HEIGHT) have value 0
    literal_inc(ta_state, patch_row, HEIGHT - PATCH_HEIGHT, 0, INCLUDE_STATE);

    // Position X literals: [patch_col, WIDTH-PATCH_WIDTH) have value 0
    literal_inc(ta_state, HEIGHT - PATCH_HEIGHT + patch_col, N_POSITION_FEATS, 0, INCLUDE_STATE);

        #if NEGATED_LITERALS
    // Negated position Y: [0, patch_row) have value 0
    literal_inc(ta_state, 0, patch_row, LITERALS / 2, INCLUDE_STATE);

    // Negated position X: [0, patch_col) have value 0
    literal_inc(ta_state, HEIGHT - PATCH_HEIGHT, HEIGHT - PATCH_HEIGHT + patch_col, LITERALS / 2, INCLUDE_STATE);
        #endif
    #endif

    // Feature literals with thermometer encoding
    for (int fid = 0; fid < N_FEATURES; ++fid) {
        int n_bits = literal_offsets[fid + 1] - literal_offsets[fid];
        int lit_start = N_POSITION_FEATS + literal_offsets[fid];
        int lit_end = N_POSITION_FEATS + literal_offsets[fid + 1];

        int32_t val = get_feature_value(X, patch_row, patch_col, fid);
        int shifted_val = val - feat_mins[fid];

        // Positive literals: increment where value is 0, i.e., [shifted_val, n_bits)
        literal_inc(ta_state, lit_start + shifted_val, lit_end, 0, INCLUDE_STATE);

    #if NEGATED_LITERALS
        // Negated: increment where value is 0, i.e., [0, shifted_val)
        literal_inc(ta_state, lit_start, lit_start + shifted_val, LITERALS / 2, INCLUDE_STATE);
    #endif
    }
#endif
}

// ============================================================================
// Clause evaluation and update
// ============================================================================

void eval_clauses(uint* restrict rng, const int32_t* restrict X, const uint* restrict global_ta_states,
                  const int8_t* restrict clause_drop_mask, int* restrict selected_patch_ids,
                  uint* restrict clause_num_includes, const int* restrict feat_mins,
                  const int* restrict literal_offsets) {
    OMP_PARALLEL_FOR
    for (ull clause = 0; clause < TOTAL_CLAUSES; clause++) {
        // Skip dropped clauses
        if (clause_drop_mask[clause] == 1) {
            selected_patch_ids[clause] = -1;
            clause_num_includes[clause] = 0;
            continue;
        }

        const uint* ta_state = &global_ta_states[clause * LITERALS];

        int clause_positions[4];                // min_row, max_row, min_col, max_col
        int valid_feat_ranges[N_FEATURES * 2];  // min and max valid value
        int interesting_fids[N_FEATURES], interesting_fid_len = 0;
        bool is_valid;
        uint num_includes = scan_clause(ta_state, literal_offsets, clause_positions, valid_feat_ranges, &is_valid,
                                        interesting_fids, &interesting_fid_len);

        clause_num_includes[clause] = num_includes;

        if (!is_valid) {
            // Clause has contradiction
            selected_patch_ids[clause] = -1;
            continue;
        }

        if (num_includes == 0) {
            // Empty clause: randomly select a patch
            selected_patch_ids[clause] = (int)(xorshift32(&rng[GET_THREAD_ID]) * PATCHES);
            continue;
        }

        int selected_patch = -1;
        int active_patch_count = 0;

        for (int patch_row = clause_positions[0]; patch_row < clause_positions[1]; patch_row++) {
            for (int patch_col = clause_positions[2]; patch_col < clause_positions[3]; patch_col++) {
                bool matches = true;

                for (int i = 0; matches && i < interesting_fid_len; ++i) {
                    int fid = interesting_fids[i];
                    int32_t val = get_feature_value(X, patch_row, patch_col, fid);
                    int shifted_val = val - feat_mins[fid];
                    // Range check: shifted_val must be in [min, max] (closed)
                    if (shifted_val < valid_feat_ranges[fid * 2] || shifted_val > valid_feat_ranges[fid * 2 + 1]) {
                        matches = false;
                    }
                }

                if (matches) {
                    active_patch_count++;
                    if (xorshift32(&rng[GET_THREAD_ID]) < 1.0f / active_patch_count) {
                        selected_patch = patch_row * N_PATCHES_X + patch_col;
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
                    const int* restrict literal_offsets) {
    for (ull clause = 0; clause < TOTAL_CLAUSES; clause++) {
        // Skip dropped clauses
        if (clause_drop_mask[clause] == 1) continue;

        uint* ta_state = &global_ta_states[clause * LITERALS];
        int local_clause_output = selected_patch_ids[clause] > -1 ? 1 : 0;

        // Get patch coordinates if clause was active
        int patch_row = -1, patch_col = -1;
        if (local_clause_output) {
            patch_row = selected_patch_ids[clause] / N_PATCHES_X;
            patch_col = selected_patch_ids[clause] % N_PATCHES_X;
        }

        uint num_includes = clause_num_includes[clause];

        ull class_id, rel_clause = clause % CLAUSES_PER_CLASS;
        LOOP_CLASS_ID(class_id, clause) {
            int local_target = targets[class_id];
            if (local_target == 0) continue;

            float* local_weight = &clause_weights[class_id * CLAUSES_PER_CLASS + rel_clause];
            int sign = (*local_weight >= 0) - (*local_weight < 0);

            float update_prob = prob[class_id];
            bool should_update = (xorshift32(&rng[GET_THREAD_ID]) <= update_prob);
            bool clause_has_space = (num_includes <= (uint)MAX_INCLUDED_LITERALS);
            bool t1 = (local_target * sign) > 0;

            // Type 1a feedback - TP - clause is active with correct polarity and has space
            if (should_update && t1 && local_clause_output && clause_has_space) {
                type1a_fb(&rng[GET_THREAD_ID], ta_state, local_weight, X, patch_row, patch_col, sign, feat_mins,
                          literal_offsets);
            }

            // Type 1b feedback - FN - clause is inactive or overflowing, but should have been active
            if (should_update && t1 && !(local_clause_output && clause_has_space)) {
                type1b_fb(&rng[GET_THREAD_ID], ta_state, sign);
            }

            // Type 2 feedback - FP - clause is active but has wrong polarity
            if (should_update && (local_target * sign) < 0 && local_clause_output) {
                type2_fb(ta_state, local_weight, X, patch_row, patch_col, sign, feat_mins, literal_offsets);
            }
        }
    }
}

// ============================================================================
// Main entry points
// ============================================================================

void fit_sample(uint* restrict rng, uint* restrict global_ta_states, float* restrict clause_weights,
                int* restrict patch_weights, const int8_t* restrict clause_drop_mask, const int32_t* restrict X,
                const int8_t* restrict targets, const int* restrict feat_mins, const int* restrict literal_offsets) {
    /*
     * Fit a single sample.
     *
     * Inputs:
     * rng => RNG state array.
     * global_ta_states => (TOTAL_CLAUSES * LITERALS) .
     * clause_weights => (CLASSES * CLAUSES_PER_CLASS) .
     * patch_weights => (TOTAL_CLAUSES * PATCHES)
     * clause_drop_mask => (TOTAL_CLAUSES)
     * X => (HEIGHT * WIDTH * DEPTH) - single sample, int32
     * targets => (CLASSES) - target for this sample
     * feat_mins => (N_FEATURES)
     * literal_offsets => (N_FEATURES + 1)
     */

    // Step 1: Evaluate clauses and select patches
    int selected_patch_ids[TOTAL_CLAUSES];
    uint clause_num_includes[TOTAL_CLAUSES];
    eval_clauses(rng, X, global_ta_states, clause_drop_mask, selected_patch_ids, clause_num_includes, feat_mins,
                 literal_offsets);

    // Step 2: Count votes
    float votes[CLASSES];
    memset(votes, 0, sizeof(votes));

    for (ull clause = 0; clause < TOTAL_CLAUSES; clause++) {
        if (selected_patch_ids[clause] != -1) {
            ull class_id, rel_clause = clause % CLAUSES_PER_CLASS;
            LOOP_CLASS_ID(class_id, clause) {
                // This needs to be atomic, so do not parallelize
                votes[class_id] += clause_weights[class_id * CLAUSES_PER_CLASS + rel_clause];
            }
            patch_weights[clause * PATCHES + selected_patch_ids[clause]]++;
        }
    }

    // Step 3: Calculate update probabilities
    float prob[CLASSES];
    OMP_PARALLEL_FOR
    for (ull class_id = 0; class_id < CLASSES; class_id++) {
        int local_target = targets[class_id];
        if (local_target == 0) {
            prob[class_id] = 0.0f;
            continue;
        }

        float y = (float)THRESH * (float)local_target;
        float class_sum = (float)CLIP(votes[class_id], -THRESH, THRESH);
        prob[class_id] = uprob_fun(class_sum, y);
    }

    // Step 4: Update clauses
    update_clauses(rng, selected_patch_ids, clause_num_includes, global_ta_states, clause_weights, clause_drop_mask, X,
                   targets, prob, feat_mins, literal_offsets);
}

// Convert clauses to ranges, check for contradictions, check if clause is empty, and count includes. Done once before
// inference.
void infer_clauses(const uint* restrict global_ta_states, int* restrict clause_positions,
                   int* restrict valid_feat_ranges, bool* restrict clause_valid, const int* restrict feat_mins,
                   const int* restrict literal_offsets, uint* restrict num_includes, int* restrict interesting_fids,
                   int* restrict interesting_fid_len) {
    OMP_PARALLEL_FOR
    for (ull clause = 0; clause < TOTAL_CLAUSES; clause++) {
        const uint* ta_state = &global_ta_states[clause * LITERALS];
        int* local_clause_position = &clause_positions[clause * 4];
        int* local_valid_feat_range = &valid_feat_ranges[clause * N_FEATURES * 2];
        bool* is_valid = &clause_valid[clause];
        uint* local_num_includes = &num_includes[clause];
        int* local_interesting_fids = &interesting_fids[clause * N_FEATURES];
        int* local_interesting_fid_len = &interesting_fid_len[clause];

        *local_num_includes = scan_clause(ta_state, literal_offsets, local_clause_position, local_valid_feat_range,
                                          is_valid, local_interesting_fids, local_interesting_fid_len);
    }
}

void infer_sample(const int32_t* restrict X, const float* restrict clause_weights, float* restrict class_sums,
                  const int* restrict feat_mins, const int* restrict clause_positions,
                  const int* restrict valid_feat_ranges, const bool* restrict clause_valid,
                  const uint* restrict num_includes, const int* restrict interesting_fids,
                  const int* restrict interesting_fid_lens) {
    /*
     * Inference for a single sample using pre-computed clause information.
     *
     * Inputs:
     * X => (HEIGHT * WIDTH * DEPTH).
     * clause_weights => (CLASSES * CLAUSES_PER_CLASS).
     * feat_mins => (N_FEATURES).
     * clause_positions => (TOTAL_CLAUSES * 4) - min_row, max_row, min_col, max_col per clause.
     * valid_feat_ranges => (TOTAL_CLAUSES * N_FEATURES * 2) - min/max per feature per clause.
     * clause_valid => (TOTAL_CLAUSES) - whether clause has no contradictions.
     * num_includes => (TOTAL_CLAUSES) - number of includes per clause.
     * interesting_fids => (TOTAL_CLAUSES * N_FEATURES) - feature ids with constraints.
     * interesting_fid_lens => (TOTAL_CLAUSES) - length of interesting_fids per clause.
     *
     * Outputs:
     * class_sums => (CLASSES) - accumulated votes per class
     */

    memset(class_sums, 0, sizeof(float) * CLASSES);

    OMP_PARALLEL_FOR
    for (ull clause = 0; clause < TOTAL_CLAUSES; clause++) {
        if (!clause_valid[clause]) {
            // Clause has contradiction
            continue;
        }

        if (num_includes[clause] == 0) {
            // Skip empty clauses during inference
            continue;
        }

        const int* local_clause_positions = &clause_positions[clause * 4];
        const int* local_valid_feat_ranges = &valid_feat_ranges[clause * N_FEATURES * 2];
        const int* local_interesting_fids = &interesting_fids[clause * N_FEATURES];
        int local_interesting_fid_len = interesting_fid_lens[clause];

        bool clause_matched = false;

        for (int patch_row = local_clause_positions[0]; patch_row < local_clause_positions[1] && !clause_matched;
             patch_row++) {
            for (int patch_col = local_clause_positions[2]; patch_col < local_clause_positions[3] && !clause_matched;
                 patch_col++) {
                bool matches = true;

                for (int i = 0; matches && i < local_interesting_fid_len; ++i) {
                    int fid = local_interesting_fids[i];
                    int32_t val = get_feature_value(X, patch_row, patch_col, fid);
                    int shifted_val = val - feat_mins[fid];
                    // Range check: shifted_val must be in [min, max] (closed)
                    if (shifted_val < local_valid_feat_ranges[fid * 2] ||
                        shifted_val > local_valid_feat_ranges[fid * 2 + 1]) {
                        matches = false;
                    }
                }

                if (matches) clause_matched = true;
            }
        }

        if (clause_matched) {
            ull class_id, rel_clause = clause % CLAUSES_PER_CLASS;
            LOOP_CLASS_ID(class_id, clause) {
                OMP_ATOMIC
                class_sums[class_id] += clause_weights[class_id * CLAUSES_PER_CLASS + rel_clause];
            }
        }
    }
}

void infer_batch(const int32_t* restrict X, const float* restrict clause_weights, float* restrict class_sums, int N,
                 const int* restrict feat_mins, const int* restrict clause_positions,
                 const int* restrict valid_feat_ranges, const bool* restrict clause_valid,
                 const uint* restrict num_includes, const int* restrict interesting_fids,
                 const int* restrict interesting_fid_lens) {
    /*
     * Inference on N samples using pre-computed clause information.
     *
     * Inputs:
     * X => (N * HEIGHT * WIDTH * DEPTH).
     * clause_weights => (CLASSES * CLAUSES_PER_CLASS).
     * N => number of samples.
     * feat_mins => (N_FEATURES).
     * clause_positions => (TOTAL_CLAUSES * 4) - min_row, max_row, min_col, max_col per clause.
     * valid_feat_ranges => (TOTAL_CLAUSES * N_FEATURES * 2) - min/max per feature per clause.
     * clause_valid => (TOTAL_CLAUSES) - whether clause has no contradictions.
     * num_includes => (TOTAL_CLAUSES) - number of includes per clause.
     * interesting_fids => (TOTAL_CLAUSES * N_FEATURES) - feature ids with constraints.
     * interesting_fid_lens => (TOTAL_CLAUSES) - length of interesting_fids per clause.
     *
     * Outputs:
     * class_sums => (N * CLASSES) - accumulated votes per class per sample.
     */

    OMP_PARALLEL_FOR
    for (int e = 0; e < N; e++) {
        const int32_t* X_sample = &X[e * HEIGHT * WIDTH * DEPTH];
        float* cs_sample = &class_sums[e * CLASSES];

        infer_sample(X_sample, clause_weights, cs_sample, feat_mins, clause_positions, valid_feat_ranges, clause_valid,
                     num_includes, interesting_fids, interesting_fid_lens);
    }
}

void fit_batch(uint* restrict rng, uint* restrict global_ta_states, float* restrict clause_weights,
               int* restrict patch_weights, const int8_t* restrict clause_drop_mask, const int32_t* restrict X,
               const int8_t* restrict targets, int N, const int* restrict feat_mins,
               const int* restrict literal_offsets) {
    /*
     * Batch training for N samples.
     * Processes samples sequentially to preserve learning dynamics.
     *
     * Inputs:
     * rng => RNG state array.
     * global_ta_states => (TOTAL_CLAUSES * LITERALS)
     * clause_weights => (CLASSES * CLAUSES_PER_CLASS)
     * patch_weights => (TOTAL_CLAUSES * PATCHES)
     * clause_drop_mask => (TOTAL_CLAUSES)
     * X => (N * HEIGHT * WIDTH * DEPTH) int32
     * targets => (N * CLASSES)
     * N => number of samples
     * feat_mins => (N_FEATURES)
     * literal_offsets => (N_FEATURES + 1)
     */

    for (int e = 0; e < N; e++) {
        const int32_t* X_sample = &X[e * HEIGHT * WIDTH * DEPTH];
        const int8_t* targets_sample = &targets[e * CLASSES];

        fit_sample(rng, global_ta_states, clause_weights, patch_weights, clause_drop_mask, X_sample, targets_sample,
                   feat_mins, literal_offsets);
    }
}

void transform_patchwise(const int32_t* restrict X, int8_t* restrict patch_output, int N, const int* restrict feat_mins,
                         const int* restrict clause_positions, const int* restrict valid_feat_ranges,
                         const bool* restrict clause_valid, const uint* restrict num_includes,
                         const int* restrict interesting_fids, const int* restrict interesting_fid_lens) {
    OMP_PARALLEL_FOR
    for (int e = 0; e < N; e++) {
        const int32_t* X_sample = &X[e * HEIGHT * WIDTH * DEPTH];
        int8_t* po = &patch_output[e * TOTAL_CLAUSES * PATCHES];

        memset(po, 0, sizeof(int8_t) * TOTAL_CLAUSES * PATCHES);

        OMP_PARALLEL_FOR
        for (ull clause = 0; clause < TOTAL_CLAUSES; clause++) {
            if (!clause_valid[clause]) {
                // Clause is false for all patches
                continue;
            }

            if (num_includes[clause] == 0) {
                // Clause is true for all patches
                memset(&po[clause * PATCHES], 1, sizeof(int8_t) * PATCHES);
                continue;
            }

            const int* local_clause_positions = &clause_positions[clause * 4];
            const int* local_valid_feat_ranges = &valid_feat_ranges[clause * N_FEATURES * 2];
            const int* local_interesting_fids = &interesting_fids[clause * N_FEATURES];
            int local_interesting_fid_len = interesting_fid_lens[clause];

            // Clause can only be true within these patch ranges.
            for (int patch_row = local_clause_positions[0]; patch_row < local_clause_positions[1]; patch_row++) {
                for (int patch_col = local_clause_positions[2]; patch_col < local_clause_positions[3]; patch_col++) {
                    bool matches = true;
                    for (int i = 0; matches && i < local_interesting_fid_len; ++i) {
                        int fid = local_interesting_fids[i];
                        int32_t val = get_feature_value(X_sample, patch_row, patch_col, fid);
                        int shifted_val = val - feat_mins[fid];
                        // Range check: shifted_val must be in [min, max] (closed)
                        if (shifted_val < local_valid_feat_ranges[fid * 2] ||
                            shifted_val > local_valid_feat_ranges[fid * 2 + 1]) {
                            matches = false;
                        }
                    }
                    po[clause * PATCHES + patch_row * N_PATCHES_X + patch_col] = matches;
                }
            }
        }
    }
}
