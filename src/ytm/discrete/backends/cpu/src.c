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
#define TRACK_PATCH_WEIGHTS 1
#define BOOST_TP_FB 1
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
#include <string.h>

#if USE_OMP
#include <omp.h>
#define GET_THREAD_ID omp_get_thread_num()
void set_num_threads(int num_threads) { omp_set_num_threads(num_threads); }
#else
#define GET_THREAD_ID 0
void set_num_threads(int num_threads) {}
#endif
#define UINT_MAX_INV (1.0f / UINT_MAX)

typedef unsigned long long ull;
typedef unsigned int uint;

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
#if USE_OMP
#pragma omp simd
#endif
    for (int li = start; li < end; ++li) {
        ta_state[li + offset] += (ta_state[li + offset] < max_val);
    }
}

static inline void literal_inc_maybe_p(uint* restrict rng, uint* restrict ta_state, int start, int end, int offset,
                                       float p) {
#if BOOST_TP_FB
    literal_inc(ta_state, start, end, offset, MAX_TA_STATE);
#else
    int li = start + geometric_sample(rng, p) - 1;
    while (li < end) {
        if (ta_state[li + offset] < MAX_TA_STATE)
            ta_state[li + offset] += 1;
        li += geometric_sample(rng, p);
    }
#endif
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
    literal_inc_maybe_p(rng, ta_state, 0, patch_idx_y, 0, 1 - S_INV);
    literal_dec_with_p(rng, ta_state, patch_idx_y, N_POSITION_FEATS_Y, 0, S_INV);

    // Position X literals: [0, patch_idx_x) have value 1, [patch_idx_x, N_POSITION_FEATS_X) have value 0
    literal_inc_maybe_p(rng, ta_state, N_POSITION_FEATS_Y, N_POSITION_FEATS_Y + patch_idx_x, 0, 1 - S_INV);
    literal_dec_with_p(rng, ta_state, N_POSITION_FEATS_Y + patch_idx_x, N_POSITION_FEATS, 0, S_INV);

#if NEGATED_LITERALS
    // Negated position Y: [0, patch_idx_y) have value 0, [patch_idx_y, N_POSITION_FEATS_Y) have value 1
    literal_dec_with_p(rng, ta_state, 0, patch_idx_y, N_LITERALS / 2, S_INV);
    literal_inc_maybe_p(rng, ta_state, patch_idx_y, N_POSITION_FEATS_Y, N_LITERALS / 2, 1 - S_INV);

    // Negated position X: [0, patch_idx_x) have value 0, [patch_idx_x, N_POSITION_FEATS_X) have value 1
    literal_dec_with_p(rng, ta_state, N_POSITION_FEATS_Y, N_POSITION_FEATS_Y + patch_idx_x, N_LITERALS / 2, S_INV);
    literal_inc_maybe_p(rng, ta_state, N_POSITION_FEATS_Y + patch_idx_x, N_POSITION_FEATS, N_LITERALS / 2, 1 - S_INV);
#endif
#endif

    // Feature literals with thermometer encoding
    for (int fid = 0; fid < N_RAW_PATCH_FEATS; ++fid) {
        int lit_start = N_POSITION_FEATS + literal_offsets[fid];
        int lit_end = N_POSITION_FEATS + literal_offsets[fid + 1];
        int n_bits = lit_end - lit_start;
        int shifted_val = CLIP(get_feature_value(X, patch_idx_y, patch_idx_x, fid) - feat_mins[fid], 0, n_bits);
        // Positive literals: bits [0, shifted_val) are 1, [shifted_val, lit_end) are 0
        literal_inc_maybe_p(rng, ta_state, lit_start, lit_start + shifted_val, 0, 1 - S_INV);
        literal_dec_with_p(rng, ta_state, lit_start + shifted_val, lit_end, 0, S_INV);
#if NEGATED_LITERALS
        // Negated: bits [0, shifted_val) are 0, [shifted_val, n_bits) are 1
        literal_dec_with_p(rng, ta_state, lit_start, lit_start + shifted_val, N_LITERALS / 2, S_INV);
        literal_inc_maybe_p(rng, ta_state, lit_start + shifted_val, lit_end, N_LITERALS / 2, 1 - S_INV);
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

        int n_bits = lit_end - lit_start;
        int shifted_val = CLIP(get_feature_value(X, patch_idx_y, patch_idx_x, fid) - feat_mins[fid], 0, n_bits);

        // Positive literals: increment where value is 0, i.e., [shifted_val, lit_end)
        literal_inc(ta_state, lit_start + shifted_val, lit_end, 0, INCLUDE_STATE);

#if NEGATED_LITERALS
        // Negated: increment where value is 0, i.e., [lit_start, shifted_val)
        literal_inc(ta_state, lit_start, lit_start + shifted_val, N_LITERALS / 2, INCLUDE_STATE);
#endif
    }
#endif
}

static inline bool is_included(uint ta_state) { return ta_state >= INCLUDE_STATE; }
static inline int min(int a, int b) { return (a < b) ? a : b; }
static inline int max(int a, int b) { return (a > b) ? a : b; }

static inline bool match_patch(const int* X, int patch_idx_y, int patch_idx_x, const int* cfmin, const int* cfmax,
                               const int* feat_mins, const int* literal_offsets, const int* constrained_fids,
                               int n_constrained) {
    for (int i = 0; i < n_constrained; ++i) {
        int fid = constrained_fids[i];
        int n_bits = literal_offsets[fid + 1] - literal_offsets[fid];
        int shifted_val = CLIP(get_feature_value(X, patch_idx_y, patch_idx_x, fid) - feat_mins[fid], 0, n_bits);
        if (shifted_val < cfmin[fid] || shifted_val > cfmax[fid])
            return false;
    }
    return true;
}

void pack_clauses(const uint* restrict global_ta_states, const int* literal_offsets, int* restrict clause_positions,
                  int* restrict clause_feat_min, int* restrict clause_feat_max, int* restrict constrained_fids,
                  int* restrict n_constrained, uint* restrict num_includes, int8_t* is_clause_valid,
                  int8_t* restrict is_clause_synced) {
    /*
     * Create a sparse ranged representation for the clauses. This is possible since all the features in the clause are
     * encoded using thermometer encoding.
     *
     * Inputs:
     * - global_ta_states[TOTAL_CLAUSES * N_LITERALS]: the state of each TA for each clause
     * - literal_offsets[N_RAW_PATCH_FEATS + 1]: Prefix sum array of the number of bins for each feature.
     *
     * Outputs:
     * - clause_positions[TOTAL_CLAUSES * 4]: the postional bounds(inclusive) for the clause in the format [min_row,
     * max_row, min_col, max_col]
     * - clause_feat_min[TOTAL_CLAUSES * N_RAW_PATCH_FEATS]: The inclusive lower bound for each feature in the clause
     * - clause_feat_max[TOTAL_CLAUSES * N_RAW_PATCH_FEATS]: The inclusive upper bound for each feature in the clause
     * - constrained_fids[TOTAL_CLAUSES * N_RAW_PATCH_FEATS]: Indices of features with constraints for each clause
     * - n_constrained[TOTAL_CLAUSES]: Number of constrained features for each clause
     * - num_includes[TOTAL_CLAUSES]: The number of included literals in the clause.
     * - is_clause_synced[TOTAL_CLAUSES]: Boolean indicating if the packed clause is in sync with the actual TA states.
     * If not, it needs to be repacked.
     */

#if USE_OMP
#pragma omp parallel for schedule(static)
#endif
    for (ull clause = 0; clause < (ull)TOTAL_CLAUSES; clause++) {
        // Skip if clause and packed clause is in sync
        if (is_clause_synced[clause])
            continue;

        const uint* ta_state = &global_ta_states[clause * (ull)N_LITERALS];
        uint* total_includes = &num_includes[clause];
        int* pos = &clause_positions[clause * 4]; // [min_row, max_row, min_col, max_col]
        is_clause_valid[clause] = 1;

        // Initialize position bounds
        pos[0] = 0;           // min_row
        pos[1] = N_PATCHES_Y; // max_row
        pos[2] = 0;           // min_col
        pos[3] = N_PATCHES_X; // max_col
        (*total_includes) = 0;

        // Scaning of thermometer literals:
        // Scan positive literals in reverse order to find the first included literal. The the bound for that feature is
        // >= bit+1. Scan negative literals in normal order to find the first included literal. Then the bound for that
        // feature is <= bit.

#if POSITION_LITERALS
        // Scan Y position literals
        for (int lit = 0; lit < N_POSITION_FEATS_Y; ++lit) {
            if (is_included(ta_state[lit])) {
                pos[0] = max(pos[0], lit + 1);
                (*total_includes)++;
            }
#if NEGATED_LITERALS
            if (is_included(ta_state[lit + N_LITERALS / 2])) {
                pos[1] = min(pos[1], lit + 1);
                (*total_includes)++;
            }
#endif
        }

        // Scan X position literals
        int offset = N_POSITION_FEATS_Y;
        for (int lit = 0; lit < N_POSITION_FEATS_X; ++lit) {
            if (is_included(ta_state[offset + lit])) {
                pos[2] = max(pos[2], lit + 1);
                (*total_includes)++;
            }
#if NEGATED_LITERALS
            if (is_included(ta_state[offset + lit + N_LITERALS / 2])) {
                pos[3] = min(pos[3], lit + 1);
                (*total_includes)++;
            }
#endif
        }
#endif

        if (pos[0] > pos[1] || pos[2] > pos[3]) {
            // Clause has contradiction
            is_clause_valid[clause] = 0;
            is_clause_synced[clause] = 1;
            continue;
        }

        // Scan features literals
        int* cfmin = &clause_feat_min[clause * N_RAW_PATCH_FEATS];
        int* cfmax = &clause_feat_max[clause * N_RAW_PATCH_FEATS];
        int* cfids = &constrained_fids[clause * N_RAW_PATCH_FEATS];
        int local_n_constrained = 0;

        for (int fid = 0; fid < N_RAW_PATCH_FEATS; ++fid) {
            int n_bits = literal_offsets[fid + 1] - literal_offsets[fid];
            int lstart = N_POSITION_FEATS + literal_offsets[fid];
            bool has_constraint = false;

            // Init the bounds to max
            cfmin[fid] = 0;
            cfmax[fid] = n_bits;

            for (int bit = 0; bit < n_bits; ++bit) {
                if (is_included(ta_state[lstart + bit])) {
                    cfmin[fid] = max(cfmin[fid], bit + 1);
                    (*total_includes)++;
                    has_constraint = true;
                }
#if NEGATED_LITERALS
                if (is_included(ta_state[lstart + bit + N_LITERALS / 2])) {
                    cfmax[fid] = min(cfmax[fid], bit);
                    (*total_includes)++;
                    has_constraint = true;
                }
#endif
            }
            if (cfmin[fid] > cfmax[fid]) {
                is_clause_valid[clause] = 0;
            }
            if (has_constraint) {
                cfids[local_n_constrained++] = fid;
            }
        }
        n_constrained[clause] = local_n_constrained;

        // Syncing complete
        is_clause_synced[clause] = 1;
    }
}

void eval_clauses(uint* restrict rng, const int* restrict clause_positions, const int* restrict clause_feat_min,
                  const int* restrict clause_feat_max, const int* restrict constrained_fids,
                  const int* restrict n_constrained, const uint* restrict num_includes,
                  const int8_t* restrict is_clause_valid, const int8_t* restrict clause_drop_mask,
                  const int* restrict feat_mins, const int* restrict literal_offsets, const int* restrict X,
                  const int e, int* selected_pids, int* patch_weights) {
    /*
     * Evaluate clauses on a input, and randomly select a patch which is matching.
     * Inputs:
     * - rng: RNG states for each thread
     * - clause_positions[TOTAL_CLAUSES * 4]: the postional bounds(inclusive) of the clause.
     * - clause_feat_min[TOTAL_CLAUSES * N_RAW_PATCH_FEATS]: The inclusive lower bound for each feature in the clause.
     * - clause_feat_max[TOTAL_CLAUSES * N_RAW_PATCH_FEATS]: The inclusive upper bound for each feature in the clause.
     * - constrained_fids[TOTAL_CLAUSES * N_RAW_PATCH_FEATS]: Indices of constrained features for each clause.
     * - n_constrained[TOTAL_CLAUSES]: Number of constrained features for each clause.
     * - num_includes[TOTAL_CLAUSES]: The number of included literals in the clause.
     * - is_clause_valid[TOTAL_CLAUSES]: invalid if clause has contradiction, and can never be true.
     * - clause_drop_mask[TOTAL_CLAUSES]: 1 if the clause is dropped.
     * - feat_mins[N_RAW_PATCH_FEATS]: The minimum value for each feature across the dataset.
     * - X[Samples * HEIGHT * WIDTH * DEPTH]: The input samples.
     * - e: the index of the sample to evaluate on.
     * Outputs:
     * - selected_pids[TOTAL_CLAUSES]: the selected patch id for each clause.
     */

#if USE_OMP
#pragma omp parallel for schedule(static)
#endif
    for (ull clause = 0; clause < TOTAL_CLAUSES; clause++) {
        // Skip dropped clauses and invalid clauses
        if (clause_drop_mask[clause] == 1 || is_clause_valid[clause] == 0) {
            selected_pids[clause] = -1;
            continue;
        }

        if (num_includes[clause] == 0) {
            // Empty clause: randomly select a patch
            selected_pids[clause] = (int)(xorshift32(&rng[GET_THREAD_ID]) * (ull)N_PATCHES);
            continue;
        }

        const int* Xe = &X[(ull)e * (ull)HEIGHT * (ull)WIDTH * (ull)DEPTH];
        const int* pos = &clause_positions[clause * 4];
        const int* cfmin = &clause_feat_min[clause * N_RAW_PATCH_FEATS];
        const int* cfmax = &clause_feat_max[clause * N_RAW_PATCH_FEATS];
        const int* cfids = &constrained_fids[clause * N_RAW_PATCH_FEATS];
        int clause_n_constrained = n_constrained[clause];
        int* selected_patch = &selected_pids[clause];
        int active_patch_count = 0;
        *selected_patch = -1; // -1 means no patch matches the clause

        // Check only the postions where clause can be true.
        for (int patch_idx_y = pos[0]; patch_idx_y < pos[1]; patch_idx_y++) {
            for (int patch_idx_x = pos[2]; patch_idx_x < pos[3]; patch_idx_x++) {
                bool patch_matches =
                    match_patch(Xe, patch_idx_y, patch_idx_x, cfmin, cfmax, feat_mins, literal_offsets, cfids, clause_n_constrained);
                if (patch_matches) {
                    // Reservoir sampling to select a patch.
                    active_patch_count++;
                    if (xorshift32(&rng[GET_THREAD_ID]) < 1.0f / active_patch_count) {
                        *selected_patch = patch_idx_y * N_PATCHES_X + patch_idx_x;
                    }
                }
            }
        }

        if (*selected_patch != -1) {
#if TRACK_PATCH_WEIGHTS
            patch_weights[clause * N_PATCHES + *selected_patch]++;
#endif
        }
    }
}

void update_clauses(uint* restrict rng, const int* restrict selected_patch_ids, const uint* restrict num_includes,
                    const int8_t* restrict clause_drop_mask, const int* restrict X, const float* restrict targets,
                    const int e, const float* restrict prob, const int* restrict feat_mins,
                    const int* restrict literal_offsets, int8_t* restrict is_clause_synced,
                    uint* restrict global_ta_states, float* restrict clause_weights) {
/*
 * Update clauses.
 * Inputs:
 *   - rng: RNG states for each thread
 *   - selected_patch_ids[TOTAL_CLAUSES]: the selected patch id for each clause. -1 if no patch matches the clause.
 *   - num_includes[TOTAL_CLAUSES]: The number of included literals in the clause.
 *   - clause_drop_mask[TOTAL_CLAUSES]: 1 if the clause is dropped.
 *   - X[Samples * HEIGHT * WIDTH * DEPTH]: The input samples.
 *   - targets[Samples * CLASSES]: The target labels for each sample and class
 *   - e: the index of the sample to evaluate on.
 *   - prob[CLASSES]: The probability to update for each class.
 *   - feat_mins[N_RAW_PATCH_FEATS]: The minimum value for each feature across the dataset.
 *   - literal_offsets[N_RAW_PATCH_FEATS + 1]: Prefix sum array of the number of bins for each feature.
 *   - is_clause_synced[TOTAL_CLAUSES]: Boolean array indicating if the packed clause is in sync with the actual TA
 * states.
 *   - global_ta_states[TOTAL_CLAUSES * N_LITERALS]: the state of each TA for each clause
 *   - clause_weights[CLAUSES_PER_CLASS * CLASSES]: the weight for each clause and class.
 */
#if USE_OMP
#pragma omp parallel for schedule(static)
#endif
    for (ull clause = 0; clause < TOTAL_CLAUSES; clause++) {
        // Skip dropped clauses
        if (clause_drop_mask[clause] == 1)
            continue;

        const int* Xe = &X[(ull)e * (ull)HEIGHT * (ull)WIDTH * (ull)DEPTH];
        uint* ta_state = &global_ta_states[clause * N_LITERALS];
        bool clause_has_space = num_includes[clause] <= (uint)MAX_INCLUDED_LITERALS;
        int local_clause_output = selected_patch_ids[clause] > -1 ? 1 : 0;

        // Get patch coordinates if clause was active
        int patch_idx_y = -1, patch_idx_x = -1;
        if (local_clause_output) {
            patch_idx_y = selected_patch_ids[clause] / N_PATCHES_X;
            patch_idx_x = selected_patch_ids[clause] % N_PATCHES_X;
        }

        ull class_id, rel_clause = clause % CLAUSES_PER_CLASS;
        LOOP_CLASS_ID(class_id, clause) {
            float q_prob = targets[e * CLASSES + class_id]; // Can be [-1, 1]

            if (q_prob == 0.0f || xorshift32(&rng[GET_THREAD_ID]) > fabs(q_prob))
                continue;
            int local_target = (q_prob > 0.0f) ? 1 : -1;

            float* local_weight = &clause_weights[class_id * CLAUSES_PER_CLASS + rel_clause];
            int sign = (*local_weight >= 0) - (*local_weight < 0);

            bool should_update = (xorshift32(&rng[GET_THREAD_ID]) <= prob[class_id]);
            bool t1 = (local_target * sign) > 0;
            bool t2 = (local_target * sign) < 0;

            // Type 1a feedback - TP - clause is active with correct polarity and has space
            if (should_update && t1 && local_clause_output && clause_has_space) {
                type1a_fb(&rng[GET_THREAD_ID], ta_state, local_weight, Xe, patch_idx_y, patch_idx_x, sign, feat_mins,
                          literal_offsets);
                is_clause_synced[clause] = 0;
            }

            // Type 1b feedback - FN - clause is inactive or overflowing, but should have been active
            if (should_update && t1 && !(local_clause_output && clause_has_space)) {
                type1b_fb(&rng[GET_THREAD_ID], ta_state, sign);
                is_clause_synced[clause] = 0;
            }

            // Type 2 feedback - FP - clause is active but has wrong polarity
            if (should_update && t2 && local_clause_output) {
                type2_fb(ta_state, local_weight, Xe, patch_idx_y, patch_idx_x, sign, feat_mins, literal_offsets);
                is_clause_synced[clause] = 0;
            }
        }
    }
}

void fit_sample(uint* restrict rng, uint* restrict global_ta_states, float* restrict clause_weights,
                int* restrict patch_weights, const int* restrict feat_mins, const int* restrict literal_offsets,
                const int8_t* restrict clause_drop_mask, const int32_t* restrict X, const float* restrict targets,
                const int e,
                // Array allocations
                int* restrict clause_positions, int* restrict clause_feat_min, int* restrict clause_feat_max,
                int* restrict constrained_fids, int* restrict n_constrained, uint* restrict num_includes,
                int8_t* restrict is_clause_valid, int8_t* restrict is_clause_synced, int* restrict selected_pids,
                float* restrict votes, float* restrict prob) {

    pack_clauses(global_ta_states, literal_offsets, clause_positions, clause_feat_min, clause_feat_max,
                 constrained_fids, n_constrained, num_includes, is_clause_valid, is_clause_synced);

    eval_clauses(rng, clause_positions, clause_feat_min, clause_feat_max, constrained_fids, n_constrained, num_includes,
                 is_clause_valid, clause_drop_mask, feat_mins, literal_offsets, X, e, selected_pids, patch_weights);

    memset(votes, 0, sizeof(float) * CLASSES);
#if USE_OMP
#pragma omp parallel for schedule(static) reduction(+ : votes[ : CLASSES])
#endif
    for (ull clause = 0; clause < TOTAL_CLAUSES; clause++) {
        if (selected_pids[clause] != -1) {
            ull class_id, rel_clause = clause % CLAUSES_PER_CLASS;
            LOOP_CLASS_ID(class_id, clause) {
                votes[class_id] += clause_weights[class_id * CLAUSES_PER_CLASS + rel_clause];
            }
        }
    }

#if USE_OMP
#pragma omp parallel for schedule(static)
#endif
    for (ull class_id = 0; class_id < CLASSES; class_id++) {
        float local_target = targets[e * CLASSES + class_id];
        if (local_target == 0.0f) {
            prob[class_id] = 0.0f;
            continue;
        }

        float y = (float)THRESH * (float)(local_target > 0 ? 1 : -1);
        float class_sum = (float)CLIP(votes[class_id], -THRESH, THRESH);
        prob[class_id] = uprob_fun(class_sum, y);
    }

    update_clauses(rng, selected_pids, num_includes, clause_drop_mask, X, targets, e, prob, feat_mins, literal_offsets,
                   is_clause_synced, global_ta_states, clause_weights);
}

void infer_sample(const float* restrict clause_weights, const int* restrict clause_positions,
                  const int* restrict clause_feat_min, const int* restrict clause_feat_max,
                  const int* restrict constrained_fids, const int* restrict n_constrained,
                  const uint* restrict num_includes, const int8_t* restrict is_clause_valid,
                  const int* restrict feat_mins, const int* restrict literal_offsets, const int* restrict X,
                  const int e, float* restrict class_sums) {

    const int* Xe = &X[(ull)e * HEIGHT * WIDTH * DEPTH];
    float* cse = &class_sums[(ull)e * CLASSES];
    memset(cse, 0, sizeof(float) * CLASSES);

#if USE_OMP
#pragma omp parallel for schedule(static) reduction(+ : cse[ : CLASSES])
#endif
    for (ull clause = 0; clause < TOTAL_CLAUSES; clause++) {
        // Skip empty and invalid clauses
        if (num_includes[clause] == 0 || is_clause_valid[clause] == 0) {
            continue;
        }

        const int* pos = &clause_positions[clause * 4];
        const int* cfmin = &clause_feat_min[clause * N_RAW_PATCH_FEATS];
        const int* cfmax = &clause_feat_max[clause * N_RAW_PATCH_FEATS];
        const int* cfids = &constrained_fids[clause * N_RAW_PATCH_FEATS];
        int clause_n_constrained = n_constrained[clause];
        bool matching_patch_found = false;

        // Early exit on first matching patch
        for (int py = pos[0]; !matching_patch_found && py < pos[1]; ++py) {
            for (int px = pos[2]; !matching_patch_found && px < pos[3]; ++px) {
                matching_patch_found = match_patch(Xe, py, px, cfmin, cfmax, feat_mins, literal_offsets, cfids, clause_n_constrained);
            }
        }

        // If a matching patch is found, add the clause weight to the class sum.
        if (matching_patch_found) {
            ull class_id, rel_clause = clause % CLAUSES_PER_CLASS;
            LOOP_CLASS_ID(class_id, clause) {
                cse[class_id] += clause_weights[class_id * CLAUSES_PER_CLASS + rel_clause];
            }
        }
    }
}

void eval_sample_patchwise(const float* restrict clause_weights, const int* restrict clause_positions,
                           const int* restrict clause_feat_min, const int* restrict clause_feat_max,
                           const int* restrict constrained_fids, const int* restrict n_constrained,
                           const uint* restrict num_includes, const int8_t* restrict is_clause_valid,
                           const int* restrict feat_mins, const int* restrict literal_offsets,
                           const int* restrict X, const int e, int8_t* restrict co_patchwise) {

    const int* Xe = &X[(ull)e * HEIGHT * WIDTH * DEPTH];
    int8_t* copwe = &co_patchwise[(ull)e * TOTAL_CLAUSES * N_PATCHES];
    memset(copwe, 0, sizeof(int8_t) * TOTAL_CLAUSES * N_PATCHES); // Initialize with 0

#if USE_OMP
#pragma omp parallel for schedule(static)
#endif
    for (ull clause = 0; clause < TOTAL_CLAUSES; clause++) {
        if (num_includes[clause] == 0) {
            // Empty clause: matches all patches
            memset(&copwe[clause * (ull)N_PATCHES], 1, sizeof(int8_t) * (ull)N_PATCHES);
            continue;
        }

        if (is_clause_valid[clause] == 0) {
            // Invalid clause: matches no patches
            continue;
        }

        const int* pos = &clause_positions[clause * 4];
        const int* cfmin = &clause_feat_min[clause * (ull)N_RAW_PATCH_FEATS];
        const int* cfmax = &clause_feat_max[clause * (ull)N_RAW_PATCH_FEATS];
        const int* cfids = &constrained_fids[clause * N_RAW_PATCH_FEATS];
        int clause_n_constrained = n_constrained[clause];

        for (int py = pos[0]; py < pos[1]; ++py) {
            for (int px = pos[2]; px < pos[3]; ++px) {
                copwe[clause * (ull)N_PATCHES + py * (ull)N_PATCHES_X + px] =
                    match_patch(Xe, py, px, cfmin, cfmax, feat_mins, literal_offsets, cfids, clause_n_constrained);
            }
        }
    }
}
