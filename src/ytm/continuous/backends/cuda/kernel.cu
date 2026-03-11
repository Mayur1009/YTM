// CUDA kernels for Continuous Tsetlin Machine
// Uses thermometer encoding with variable bit-widths per feature

#ifdef IS_NEOVIM_CLANGD_ENV
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

// Derived macros
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

#define CLIP(val, lo, hi) ((val < lo) ? lo : ((val > hi) ? hi : val))

#include <curand_kernel.h>

typedef unsigned long long ull;
typedef unsigned int uint;

extern "C" {

// ============================================================================
// Device helper functions
// ============================================================================

__device__ static inline void d_max(int* a, int b) { *a = (*a > b) ? *a : b; }
__device__ static inline void d_min(int* a, int b) { *a = (*a < b) ? *a : b; }

// Sample from geometric distribution with probability p
__device__ static inline int geometric_sample(curandState* rng, float p) {
    float u = curand_uniform(rng);
    if (u >= 1.0f) u = 0.9999999f;
    return (int)(logf(1.0f - u) / logf(1.0f - p)) + 1;
}

// Probabilistic literal decrement
__device__ static inline void literal_dec_with_p(curandState* rng, uint* ta_state, int start, int end, int offset,
                                                  float p) {
    int li = start + geometric_sample(rng, p) - 1;
    while (li < end) {
        if (ta_state[li + offset] > 0) ta_state[li + offset] -= 1;
        li += geometric_sample(rng, p);
    }
}

// Literal increment (branchless)
__device__ static inline void literal_inc(uint* ta_state, int start, int end, int offset, uint max_val) {
    for (int li = start; li < end; ++li) {
        ta_state[li + offset] += (ta_state[li + offset] < max_val);
    }
}

// Update probability function
__device__ static inline float uprob_fun(float v, float y) {
    float prob = (y - v) / (2 * y);
    return prob;
}

// Get feature value from X at a given patch position
__device__ static inline int get_feature_value(const int* X, int patch_row, int patch_col, int fid) {
    ull rel_y = fid / (PATCH_WIDTH * DEPTH);
    ull rel_x = (fid / DEPTH) % PATCH_WIDTH;
    ull z = fid % DEPTH;
    ull abs_y = patch_row + rel_y;
    ull abs_x = patch_col + rel_x;
    return X[abs_y * (WIDTH * DEPTH) + abs_x * DEPTH + z];
}

// ============================================================================
// Scan clause - convert TA states to ranges for thermometer encoding
// ============================================================================

__device__ static inline uint scan_clause(const uint* ta_state, const int* literal_offsets, int* clause_position,
                                          int* valid_feat_range, bool* is_valid, int* interesting_fids, int* fid_len) {
    uint num_includes = 0;
    *is_valid = true;
    clause_position[0] = 0;            // min_row
    clause_position[1] = N_PATCHES_Y;  // max_row
    clause_position[2] = 0;            // min_col
    clause_position[3] = N_PATCHES_X;  // max_col

#if POSITION_LITERALS
    // Initialize position ranges

    // Process Y position literals
    for (int lit = 0; lit < HEIGHT - PATCH_HEIGHT; ++lit) {
        if (ta_state[lit] >= INCLUDE_STATE) {
            d_max(&clause_position[0], lit + 1);
            num_includes++;
        }
    #if NEGATED_LITERALS
        if (ta_state[lit + LITERALS / 2] >= INCLUDE_STATE) {
            d_min(&clause_position[1], lit + 1);
            num_includes++;
        }
    #endif
    }

    // Process X position literals
    for (int lit = HEIGHT - PATCH_HEIGHT; lit < N_POSITION_FEATS; ++lit) {
        if (ta_state[lit] >= INCLUDE_STATE) {
            int x_lit = lit - (HEIGHT - PATCH_HEIGHT);
            d_max(&clause_position[2], x_lit + 1);
            num_includes++;
        }
    #if NEGATED_LITERALS
        if (ta_state[lit + LITERALS / 2] >= INCLUDE_STATE) {
            int x_lit = lit - (HEIGHT - PATCH_HEIGHT);
            d_min(&clause_position[3], x_lit + 1);
            num_includes++;
        }
    #endif
    }

    if (clause_position[0] >= clause_position[1] || clause_position[2] >= clause_position[3]) *is_valid = false;
#endif

    *fid_len = 0;
    for (int fid = 0; fid < N_FEATURES; ++fid) {
        int n_bits = literal_offsets[fid + 1] - literal_offsets[fid];
        // Valid range is [0, n_bits] inclusive, stored as closed [min, max]
        valid_feat_range[fid * 2] = 0;
        valid_feat_range[fid * 2 + 1] = n_bits;

        for (int bit = 0; bit < n_bits; ++bit) {
            int lit_pos = N_POSITION_FEATS + literal_offsets[fid] + bit;
            if (ta_state[lit_pos] >= INCLUDE_STATE) {
                // Positive literal at bit k: shifted_val >= k + 1
                d_max(&valid_feat_range[fid * 2], bit + 1);
                num_includes++;
            }
#if NEGATED_LITERALS
            int lit_neg = lit_pos + LITERALS / 2;
            if (ta_state[lit_neg] >= INCLUDE_STATE) {
                // Negated literal at bit k: shifted_val <= k (closed interval)
                d_min(&valid_feat_range[fid * 2 + 1], bit);
                num_includes++;
            }
#endif
        }

        // Track features that have constraints (range narrowed from full [0, n_bits])
        if (valid_feat_range[fid * 2] > 0 || valid_feat_range[fid * 2 + 1] < n_bits) {
            interesting_fids[(*fid_len)++] = fid;
        }

        if (*is_valid && valid_feat_range[fid * 2] > valid_feat_range[fid * 2 + 1]) *is_valid = false;
    }

    return num_includes;
}

// ============================================================================
// Feedback functions
// ============================================================================

__device__ static inline void type1a_fb(curandState* rng, uint* ta_state, float* weight, const int* X, int patch_row,
                                        int patch_col, int sign, const int* feat_mins, const int* literal_offsets) {
#if TYPE1A_FB
    #if WEIGHTED
    if (fabs(*weight) < MAX_WEIGHT) (*weight) += sign * 1.0f;
    #endif

    #if POSITION_LITERALS
    // Position Y literals
    literal_inc(ta_state, 0, patch_row, 0, MAX_TA_STATE);
    literal_dec_with_p(rng, ta_state, patch_row, HEIGHT - PATCH_HEIGHT, 0, S_INV);

    // Position X literals
    literal_inc(ta_state, HEIGHT - PATCH_HEIGHT, HEIGHT - PATCH_HEIGHT + patch_col, 0, MAX_TA_STATE);
    literal_dec_with_p(rng, ta_state, HEIGHT - PATCH_HEIGHT + patch_col, N_POSITION_FEATS, 0, S_INV);

        #if NEGATED_LITERALS
    // Negated position Y
    literal_dec_with_p(rng, ta_state, 0, patch_row, LITERALS / 2, S_INV);
    literal_inc(ta_state, patch_row, HEIGHT - PATCH_HEIGHT, LITERALS / 2, MAX_TA_STATE);

    // Negated position X
    literal_dec_with_p(rng, ta_state, HEIGHT - PATCH_HEIGHT, HEIGHT - PATCH_HEIGHT + patch_col, LITERALS / 2, S_INV);
    literal_inc(ta_state, HEIGHT - PATCH_HEIGHT + patch_col, N_POSITION_FEATS, LITERALS / 2, MAX_TA_STATE);
        #endif
    #endif

    // Feature literals with thermometer encoding
    for (int fid = 0; fid < N_FEATURES; ++fid) {
        int lit_start = N_POSITION_FEATS + literal_offsets[fid];
        int lit_end = N_POSITION_FEATS + literal_offsets[fid + 1];

        int val = get_feature_value(X, patch_row, patch_col, fid);
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

__device__ static inline void type1b_fb(curandState* rng, uint* ta_state, int sign) {
#if TYPE1B_FB
    literal_dec_with_p(rng, ta_state, 0, LITERALS, 0, S_INV);
#endif
}

__device__ static inline void type2_fb(uint* ta_state, float* weight, const int* X, int patch_row, int patch_col,
                                       int sign, const int* feat_mins, const int* literal_offsets) {
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
        int lit_start = N_POSITION_FEATS + literal_offsets[fid];
        int lit_end = N_POSITION_FEATS + literal_offsets[fid + 1];

        int val = get_feature_value(X, patch_row, patch_col, fid);
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
// Kernels
// ============================================================================

// Pre-compute clause information for inference
__global__ void infer_clauses(const uint* __restrict__ global_ta_states, int* __restrict__ clause_positions,
                              int* __restrict__ valid_feat_ranges, bool* __restrict__ clause_valid,
                              const int* __restrict__ feat_mins, const int* __restrict__ literal_offsets,
                              uint* __restrict__ num_includes, int* __restrict__ interesting_fids,
                              int* __restrict__ interesting_fid_lens) {
    ull index = blockIdx.x * blockDim.x + threadIdx.x;
    ull stride = blockDim.x * gridDim.x;

    for (ull clause = index; clause < TOTAL_CLAUSES; clause += stride) {
        const uint* ta_state = &global_ta_states[clause * LITERALS];
        int* local_clause_position = &clause_positions[clause * 4];
        int* local_valid_feat_range = &valid_feat_ranges[clause * N_FEATURES * 2];
        bool* is_valid = &clause_valid[clause];
        uint* local_num_includes = &num_includes[clause];
        int* local_interesting_fids = &interesting_fids[clause * N_FEATURES];
        int* local_interesting_fid_len = &interesting_fid_lens[clause];

        *local_num_includes =
            scan_clause(ta_state, literal_offsets, local_clause_position, local_valid_feat_range, is_valid,
                        local_interesting_fids, local_interesting_fid_len);
    }
}

// Inference for batch of samples
__global__ void infer_sample(const int* __restrict__ X, const float* __restrict__ clause_weights,
                             float* __restrict__ class_sums, int N, const int* __restrict__ feat_mins,
                             const int* __restrict__ clause_positions, const int* __restrict__ valid_feat_ranges,
                             const bool* __restrict__ clause_valid, const uint* __restrict__ num_includes,
                             const int* __restrict__ interesting_fids, const int* __restrict__ interesting_fid_lens) {
    ull index = blockIdx.x * blockDim.x + threadIdx.x;
    ull stride = blockDim.x * gridDim.x;

    for (ull e_clause = index; e_clause < (ull)N * TOTAL_CLAUSES; e_clause += stride) {
        ull e = e_clause / TOTAL_CLAUSES;
        ull clause = e_clause % TOTAL_CLAUSES;

        if (!clause_valid[clause]) continue;
        if (num_includes[clause] == 0) continue;

        const int* X_sample = &X[e * HEIGHT * WIDTH * DEPTH];
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
                    int val = get_feature_value(X_sample, patch_row, patch_col, fid);
                    int shifted_val = val - feat_mins[fid];
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
                atomicAdd(&class_sums[e * CLASSES + class_id],
                          clause_weights[class_id * CLAUSES_PER_CLASS + rel_clause]);
            }
        }
    }
}

// Evaluate clauses and select patches for training (single sample)
__global__ void eval_clauses(curandState* rng, const int* __restrict__ X, const uint* __restrict__ global_ta_states,
                             const int8_t* __restrict__ clause_drop_mask, int* __restrict__ selected_patch_ids,
                             uint* __restrict__ clause_num_includes, const int* __restrict__ feat_mins,
                             const int* __restrict__ literal_offsets) {
    ull index = blockIdx.x * blockDim.x + threadIdx.x;
    ull stride = blockDim.x * gridDim.x;

    curandState local_rng = rng[index];

    for (ull clause = index; clause < TOTAL_CLAUSES; clause += stride) {
        if (clause_drop_mask[clause] == 1) {
            selected_patch_ids[clause] = -1;
            clause_num_includes[clause] = 0;
            continue;
        }

        const uint* ta_state = &global_ta_states[clause * LITERALS];

        int clause_positions[4];
        int valid_feat_ranges[N_FEATURES * 2];
        int interesting_fids[N_FEATURES];
        int interesting_fid_len = 0;
        bool is_valid;

        uint num_includes =
            scan_clause(ta_state, literal_offsets, clause_positions, valid_feat_ranges, &is_valid, interesting_fids,
                        &interesting_fid_len);

        clause_num_includes[clause] = num_includes;

        if (!is_valid) {
            selected_patch_ids[clause] = -1;
            continue;
        }

        if (num_includes == 0) {
            // Empty clause: randomly select a patch
            selected_patch_ids[clause] = (int)(curand_uniform(&local_rng) * PATCHES);
            continue;
        }

        int selected_patch = -1;
        int active_patch_count = 0;

        for (int patch_row = clause_positions[0]; patch_row < clause_positions[1]; patch_row++) {
            for (int patch_col = clause_positions[2]; patch_col < clause_positions[3]; patch_col++) {
                bool matches = true;

                for (int i = 0; matches && i < interesting_fid_len; ++i) {
                    int fid = interesting_fids[i];
                    int val = get_feature_value(X, patch_row, patch_col, fid);
                    int shifted_val = val - feat_mins[fid];
                    if (shifted_val < valid_feat_ranges[fid * 2] || shifted_val > valid_feat_ranges[fid * 2 + 1]) {
                        matches = false;
                    }
                }

                if (matches) {
                    active_patch_count++;
                    if (curand_uniform(&local_rng) < 1.0f / active_patch_count) {
                        selected_patch = patch_row * N_PATCHES_X + patch_col;
                    }
                }
            }
        }

        selected_patch_ids[clause] = selected_patch;
    }

    rng[index] = local_rng;
}

// Count votes from active clauses
__global__ void count_votes(const int* __restrict__ selected_patch_ids, const float* __restrict__ clause_weights,
                            int* __restrict__ patch_weights, float* __restrict__ votes) {
    ull index = blockIdx.x * blockDim.x + threadIdx.x;
    ull stride = blockDim.x * gridDim.x;

    for (ull clause = index; clause < TOTAL_CLAUSES; clause += stride) {
        if (selected_patch_ids[clause] != -1) {
            patch_weights[clause * PATCHES + selected_patch_ids[clause]]++;

            ull class_id, rel_clause = clause % CLAUSES_PER_CLASS;
            LOOP_CLASS_ID(class_id, clause) {
                atomicAdd(&votes[class_id], clause_weights[class_id * CLAUSES_PER_CLASS + rel_clause]);
            }
        }
    }
}

// Calculate update probabilities
__global__ void calc_update_prob(const float* __restrict__ votes, const int8_t* __restrict__ targets,
                                 float* __restrict__ prob) {
    ull index = blockIdx.x * blockDim.x + threadIdx.x;
    ull stride = blockDim.x * gridDim.x;

    for (ull class_id = index; class_id < CLASSES; class_id += stride) {
        int local_target = targets[class_id];
        if (local_target == 0) {
            prob[class_id] = 0.0f;
            continue;
        }

        float y = (float)THRESH * (float)local_target;
        float class_sum = (float)CLIP(votes[class_id], -THRESH, THRESH);
        prob[class_id] = uprob_fun(class_sum, y);
    }
}

// Update clauses with feedback
__global__ void update_clauses(curandState* rng, const int* __restrict__ selected_patch_ids,
                               const uint* __restrict__ clause_num_includes,
                               const int8_t* __restrict__ clause_drop_mask, const int* __restrict__ X,
                               const int8_t* __restrict__ targets, const float* __restrict__ prob,
                               uint* __restrict__ global_ta_states, float* __restrict__ clause_weights,
                               const int* __restrict__ feat_mins, const int* __restrict__ literal_offsets) {
    ull index = blockIdx.x * blockDim.x + threadIdx.x;
    ull stride = blockDim.x * gridDim.x;

    curandState local_rng = rng[index];

    for (ull clause = index; clause < TOTAL_CLAUSES; clause += stride) {
        if (clause_drop_mask[clause] == 1) continue;

        uint* ta_state = &global_ta_states[clause * LITERALS];
        int local_clause_output = selected_patch_ids[clause] > -1 ? 1 : 0;

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
            bool should_update = (curand_uniform(&local_rng) <= update_prob);
            bool clause_has_space = (num_includes <= (uint)MAX_INCLUDED_LITERALS);
            bool t1 = (local_target * sign) > 0;

            // Type 1a feedback
            if (should_update && t1 && local_clause_output && clause_has_space) {
                type1a_fb(&local_rng, ta_state, local_weight, X, patch_row, patch_col, sign, feat_mins, literal_offsets);
            }

            // Type 1b feedback
            if (should_update && t1 && !(local_clause_output && clause_has_space)) {
                type1b_fb(&local_rng, ta_state, sign);
            }

            // Type 2 feedback
            if (should_update && (local_target * sign) < 0 && local_clause_output) {
                type2_fb(ta_state, local_weight, X, patch_row, patch_col, sign, feat_mins, literal_offsets);
            }
        }
    }

    rng[index] = local_rng;
}

// Transform: compute clause output for each patch
__global__ void transform_patchwise(const int* __restrict__ X, int8_t* __restrict__ patch_output, int N,
                                    const int* __restrict__ feat_mins, const int* __restrict__ clause_positions,
                                    const int* __restrict__ valid_feat_ranges, const bool* __restrict__ clause_valid,
                                    const uint* __restrict__ num_includes, const int* __restrict__ interesting_fids,
                                    const int* __restrict__ interesting_fid_lens) {
    ull index = blockIdx.x * blockDim.x + threadIdx.x;
    ull stride = blockDim.x * gridDim.x;

    for (ull e_clause = index; e_clause < (ull)N * TOTAL_CLAUSES; e_clause += stride) {
        ull e = e_clause / TOTAL_CLAUSES;
        ull clause = e_clause % TOTAL_CLAUSES;

        const int* X_sample = &X[e * HEIGHT * WIDTH * DEPTH];
        int8_t* po = &patch_output[e * TOTAL_CLAUSES * PATCHES + clause * PATCHES];

        // Initialize all patches to 0 for this clause
        for (int p = 0; p < PATCHES; p++) po[p] = 0;

        if (!clause_valid[clause]) {
            // Clause is false for all patches
            continue;
        }

        if (num_includes[clause] == 0) {
            // Clause is true for all patches
            for (int p = 0; p < PATCHES; p++) po[p] = 1;
            continue;
        }

        const int* local_clause_positions = &clause_positions[clause * 4];
        const int* local_valid_feat_ranges = &valid_feat_ranges[clause * N_FEATURES * 2];
        const int* local_interesting_fids = &interesting_fids[clause * N_FEATURES];
        int local_interesting_fid_len = interesting_fid_lens[clause];

        for (int patch_row = local_clause_positions[0]; patch_row < local_clause_positions[1]; patch_row++) {
            for (int patch_col = local_clause_positions[2]; patch_col < local_clause_positions[3]; patch_col++) {
                bool matches = true;

                for (int i = 0; matches && i < local_interesting_fid_len; ++i) {
                    int fid = local_interesting_fids[i];
                    int val = get_feature_value(X_sample, patch_row, patch_col, fid);
                    int shifted_val = val - feat_mins[fid];
                    if (shifted_val < local_valid_feat_ranges[fid * 2] ||
                        shifted_val > local_valid_feat_ranges[fid * 2 + 1]) {
                        matches = false;
                    }
                }

                po[patch_row * N_PATCHES_X + patch_col] = matches;
            }
        }
    }
}

}  // extern "C"
