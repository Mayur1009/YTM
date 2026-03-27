/**
 * CUDA kernels for no-encoding Tsetlin Machine implementation.
 * Computes patch matching on-the-fly instead of pre-encoding all patches.
 * Uses sparse representation - only iterates over included literals.
 */

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
    #define MAX_INCLUDED_LITERALS 10
    #define INCLUDE_STATE 128
    #define MAX_TA_STATE 255
    #define TYPE1A_FB 1
    #define TYPE1B_FB 1
    #define TYPE2_FB 1
#endif

#define INT_SIZE 32
#define S_INV (1.0f / S)

#define N_POSITION_FEATS (HEIGHT - PATCH_HEIGHT + WIDTH - PATCH_WIDTH)
#define N_FEATURE_FEATS (PATCH_HEIGHT * PATCH_WIDTH * DEPTH)
#if NEGATED_LITERALS
    #define LITERALS (2 * (N_POSITION_FEATS + N_FEATURE_FEATS))
#else
    #define LITERALS (N_POSITION_FEATS + N_FEATURE_FEATS)
#endif

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

#include <curand_kernel.h>

typedef unsigned long long ull;
typedef unsigned int uint;

extern "C" {

    //==========================================================================
    // Helper device functions
    //==========================================================================

    __device__ static inline float uprob_fun(float v, float y) {
        float prob = (y - v) / (2 * y);
        return prob;
    }

    // Sample from geometric distribution with probability p
    // Returns the number of trials until first success (1-indexed)
    __device__ static inline int geometric_sample(curandState* rng, float p) {
        float u = curand_uniform(rng);
        if (u >= 1.0f) u = 0.9999999f;
        return (int)(logf(1.0f - u) / logf(1.0f - p)) + 1;
    }

    // Probabilistically decrement literals in range [start, end) with probability p
    // offset is added to index (use LITERALS/2 for negated, 0 otherwise)
    __device__ static inline void literal_dec_with_p(curandState* rng, uint* ta_state, int start, int end, int offset,
                                                     float p) {
        int li = start + geometric_sample(rng, p) - 1;
        while (li < end) {
            if (ta_state[li + offset] > 0) ta_state[li + offset] -= 1;
            li += geometric_sample(rng, p);
        }
    }

    // Increment literals in range [start, end) up to max_val (branchless)
    // offset is added to index (use LITERALS/2 for negated, 0 otherwise)
    __device__ static inline void literal_inc(uint* ta_state, int start, int end, int offset, uint max_val) {
        for (int li = start; li < end; ++li) {
            ta_state[li + offset] += (ta_state[li + offset] < max_val);
        }
    }

    // Get the literal value for a given literal index and patch position
    // Works for position literals (Y, X) and feature literals
    // For negated literals, caller should invert the result
    __device__ static inline int8_t get_X_fid(const int8_t* X, int patch_row, int patch_col, int lit) {
        if (lit < HEIGHT - PATCH_HEIGHT) {
            // Y position literal: lit < patch_row → 1, else 0
            return (lit < patch_row) ? 1 : 0;
        } else if (lit < N_POSITION_FEATS) {
            // X position literal: (lit - Y_offset) < patch_col → 1, else 0
            int x_lit = lit - (HEIGHT - PATCH_HEIGHT);
            return (x_lit < patch_col) ? 1 : 0;
        } else {
            // Feature literal
            int fid = lit - N_POSITION_FEATS;
            int rel_y = fid / (PATCH_WIDTH * DEPTH);
            int rel_x = (fid / DEPTH) % PATCH_WIDTH;
            int z = fid % DEPTH;
            int abs_y = patch_row + rel_y;
            int abs_x = patch_col + rel_x;
            return X[abs_y * (WIDTH * DEPTH) + abs_x * DEPTH + z];
        }
    }

    // Scan ta_state to collect included literals and compute valid patch ranges
    // Returns the number of included literals (0 if no includes)
    __device__ static inline uint scan_clause(
        const uint* ta_state,
        int* included_feats_pos,
        int* included_feats_neg,
        int* n_feats_pos,
        int* n_feats_neg,
        int* min_row,
        int* max_row,
        int* min_col,
        int* max_col
    ) {
        *min_row = 0; *max_row = N_PATCHES_Y;
        *min_col = 0; *max_col = N_PATCHES_X;

        *n_feats_pos = 0;
        *n_feats_neg = 0;
        uint total_includes = 0;
        bool has_includes = false;

        for (int lit = 0; lit < HEIGHT - PATCH_HEIGHT; ++lit) {
            if (ta_state[lit] >= INCLUDE_STATE) {
                if (lit + 1 > *min_row) *min_row = lit + 1;
                total_includes++; has_includes = true;
            }
#if NEGATED_LITERALS
            if (ta_state[lit + LITERALS / 2] >= INCLUDE_STATE) {
                if (lit + 1 < *max_row) *max_row = lit + 1;
                total_includes++; has_includes = true;
            }
#endif
        }

        for (int lit = HEIGHT - PATCH_HEIGHT; lit < N_POSITION_FEATS; ++lit) {
            int x_lit = lit - (HEIGHT - PATCH_HEIGHT);
            if (ta_state[lit] >= INCLUDE_STATE) {
                if (x_lit + 1 > *min_col) *min_col = x_lit + 1;
                total_includes++; has_includes = true;
            }
#if NEGATED_LITERALS
            if (ta_state[lit + LITERALS / 2] >= INCLUDE_STATE) {
                if (x_lit + 1 < *max_col) *max_col = x_lit + 1;
                total_includes++; has_includes = true;
            }
#endif
        }

        for (int fid = 0; fid < N_FEATURE_FEATS; ++fid) {
            int lit_pos = N_POSITION_FEATS + fid;
            if (ta_state[lit_pos] >= INCLUDE_STATE) {
                included_feats_pos[(*n_feats_pos)++] = fid;
                total_includes++; has_includes = true;
            }
#if NEGATED_LITERALS
            int lit_neg = lit_pos + LITERALS / 2;
            if (ta_state[lit_neg] >= INCLUDE_STATE) {
                included_feats_neg[(*n_feats_neg)++] = fid;
                total_includes++; has_includes = true;
            }
#endif
        }

        return has_includes ? total_includes : 0;
    }

    /**
     * Type 1a feedback - reinforce matching literals.
     * Increments TA states for literals with value=1, probabilistically decrements for value=0
     */
    __device__ static inline void type1a_fb(curandState* rng, uint* ta_state, float* weight, const int8_t* X,
                                            int patch_row, int patch_col, int sign) {
#if TYPE1A_FB
    #if WEIGHTED
        if (fabs(*weight) < MAX_WEIGHT) (*weight) += sign * 1.0f;
    #endif

        // Position Y literals: [0, patch_row) have value 1, [patch_row, HEIGHT-PATCH_HEIGHT) have value 0
        literal_inc(ta_state, 0, patch_row, 0, MAX_TA_STATE);
        literal_dec_with_p(rng, ta_state, patch_row, HEIGHT - PATCH_HEIGHT, 0, S_INV);

        // Position X literals: [0, patch_col) have value 1, [patch_col, WIDTH-PATCH_WIDTH) have value 0
        literal_inc(ta_state, HEIGHT - PATCH_HEIGHT, HEIGHT - PATCH_HEIGHT + patch_col, 0, MAX_TA_STATE);
        literal_dec_with_p(rng, ta_state, HEIGHT - PATCH_HEIGHT + patch_col, N_POSITION_FEATS, 0, S_INV);

        // Feature literals: check pixel value
        for (int lit = N_POSITION_FEATS; lit < N_POSITION_FEATS + N_FEATURE_FEATS; ++lit) {
            int8_t val = get_X_fid(X, patch_row, patch_col, lit);

            if (val == 1) {
                if (ta_state[lit] < MAX_TA_STATE) ta_state[lit] += 1;
            } else {
                if (ta_state[lit] > 0 && curand_uniform(rng) <= S_INV) ta_state[lit] -= 1;
            }

    #if NEGATED_LITERALS
            int lit_neg = lit + LITERALS / 2;
            if (val == 0) {  // inverted
                if (ta_state[lit_neg] < MAX_TA_STATE) ta_state[lit_neg] += 1;
            } else {
                if (ta_state[lit_neg] > 0 && curand_uniform(rng) <= S_INV) ta_state[lit_neg] -= 1;
            }
    #endif
        }

    #if NEGATED_LITERALS
        // Negated position Y: [0, patch_row) have value 0, [patch_row, HEIGHT-PATCH_HEIGHT) have value 1
        literal_dec_with_p(rng, ta_state, 0, patch_row, LITERALS / 2, S_INV);
        literal_inc(ta_state, patch_row, HEIGHT - PATCH_HEIGHT, LITERALS / 2, MAX_TA_STATE);

        // Negated position X: [0, patch_col) have value 0, [patch_col, WIDTH-PATCH_WIDTH) have value 1
        literal_dec_with_p(rng, ta_state, HEIGHT - PATCH_HEIGHT, HEIGHT - PATCH_HEIGHT + patch_col, LITERALS / 2, S_INV);
        literal_inc(ta_state, HEIGHT - PATCH_HEIGHT + patch_col, N_POSITION_FEATS, LITERALS / 2, MAX_TA_STATE);
    #endif
#endif
    }

    /**
     * Type 1b feedback - probabilistically decrement all literals
     */
    __device__ static inline void type1b_fb(curandState* rng, uint* ta_state, int sign) {
#if TYPE1B_FB
        literal_dec_with_p(rng, ta_state, 0, LITERALS, 0, S_INV);
#endif
    }

    /**
     * Type 2 feedback - include absent literals.
     * Increments TA states for literals with value=0 (to include them and break the clause)
     */
    __device__ static inline void type2_fb(uint* ta_state, float* weight, const int8_t* X, int patch_row, int patch_col,
                                           int sign) {
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

        // Position Y literals: [patch_row, HEIGHT-PATCH_HEIGHT) have value 0
        literal_inc(ta_state, patch_row, HEIGHT - PATCH_HEIGHT, 0, INCLUDE_STATE);

        // Position X literals: [patch_col, WIDTH-PATCH_WIDTH) have value 0
        literal_inc(ta_state, HEIGHT - PATCH_HEIGHT + patch_col, N_POSITION_FEATS, 0, INCLUDE_STATE);

        // Feature literals: increment where pixel=0
        for (int lit = N_POSITION_FEATS; lit < N_POSITION_FEATS + N_FEATURE_FEATS; ++lit) {
            int8_t val = get_X_fid(X, patch_row, patch_col, lit);

            if (val == 0 && ta_state[lit] < INCLUDE_STATE) {
                ta_state[lit] += 1;
            }

    #if NEGATED_LITERALS
            int lit_neg = lit + LITERALS / 2;
            if (val == 1 && ta_state[lit_neg] < INCLUDE_STATE) {  // inverted
                ta_state[lit_neg] += 1;
            }
    #endif
        }

    #if NEGATED_LITERALS
        // Negated position Y: [0, patch_row) have value 0
        literal_inc(ta_state, 0, patch_row, LITERALS / 2, INCLUDE_STATE);

        // Negated position X: [0, patch_col) have value 0
        literal_inc(ta_state, HEIGHT - PATCH_HEIGHT, HEIGHT - PATCH_HEIGHT + patch_col, LITERALS / 2, INCLUDE_STATE);
    #endif
#endif
    }

    //==========================================================================
    // Clause packing kernel - pre-computes sparse representation
    //==========================================================================

    /**
     * Pack clauses into sparse representation.
     * Pre-computes included feature literals and valid patch position ranges.
     * Call this once before eval_clauses (for training) or infer_batch (for inference).
     *
     * Parallelism: 1 thread per clause
     *
     * Inputs:
     *   global_ta_states => (TOTAL_CLAUSES * LITERALS)
     *
     * Outputs:
     *   included_feats_pos => (TOTAL_CLAUSES * N_FEATURE_FEATS) - positive feature literal indices
     *   included_feats_neg => (TOTAL_CLAUSES * N_FEATURE_FEATS) - negated feature literal indices
     *   n_feats_pos => (TOTAL_CLAUSES) - count of positive feature literals per clause
     *   n_feats_neg => (TOTAL_CLAUSES) - count of negated feature literals per clause
     *   patch_ranges => (TOTAL_CLAUSES * 4) - [min_row, max_row, min_col, max_col] per clause
     *   num_includes => (TOTAL_CLAUSES) - total number of included literals per clause
     */
    __global__ void pack_clauses(const uint* global_ta_states, int* included_feats_pos, int* included_feats_neg,
                                 int* n_feats_pos, int* n_feats_neg, int* patch_ranges, uint* num_includes) {
        ull index = blockIdx.x * blockDim.x + threadIdx.x;
        ull stride = blockDim.x * gridDim.x;

        for (ull clause = index; clause < TOTAL_CLAUSES; clause += stride) {
            const uint* ta_state = &global_ta_states[clause * LITERALS];

            int* clause_feats_pos = &included_feats_pos[clause * N_FEATURE_FEATS];
            int* clause_feats_neg = &included_feats_neg[clause * N_FEATURE_FEATS];
            int* clause_patch_range = &patch_ranges[clause * 4];

            int local_n_feats_pos, local_n_feats_neg;
            int min_row, max_row, min_col, max_col;

            uint total_includes = scan_clause(ta_state, clause_feats_pos, clause_feats_neg,
                                              &local_n_feats_pos, &local_n_feats_neg, &min_row, &max_row, &min_col, &max_col);

            n_feats_pos[clause] = local_n_feats_pos;
            n_feats_neg[clause] = local_n_feats_neg;
            clause_patch_range[0] = min_row;
            clause_patch_range[1] = max_row;
            clause_patch_range[2] = min_col;
            clause_patch_range[3] = max_col;
            num_includes[clause] = total_includes;
        }
    }

    //==========================================================================
    // Inference kernel
    //==========================================================================

    /**
     * Batch inference for N samples.
     * Early-exits on first matching patch per clause.
     * Reads pre-computed sparse representation from pack_clauses.
     *
     * Parallelism: 1 thread per (sample, clause) pair
     *
     * Inputs:
     *   X => (N * HEIGHT * WIDTH * DEPTH) - all samples
     *   clause_weights => (CLASSES * CLAUSES_PER_CLASS)
     *   included_feats_pos => (TOTAL_CLAUSES * N_FEATURE_FEATS) - from pack_clauses
     *   included_feats_neg => (TOTAL_CLAUSES * N_FEATURE_FEATS) - from pack_clauses
     *   n_feats_pos => (TOTAL_CLAUSES) - from pack_clauses
     *   n_feats_neg => (TOTAL_CLAUSES) - from pack_clauses
     *   patch_ranges => (TOTAL_CLAUSES * 4) - from pack_clauses
     *   num_includes => (TOTAL_CLAUSES) - from pack_clauses
     *   N => number of samples
     *
     * Outputs:
     *   class_sums => (N * CLASSES) - accumulated votes per class per sample
     */
    __global__ void infer_batch(const int8_t* X, const float* clause_weights, const int* included_feats_pos,
                                const int* included_feats_neg, const int* n_feats_pos, const int* n_feats_neg,
                                const int* patch_ranges, const uint* num_includes, const int N, float* class_sums) {
        ull index = blockIdx.x * blockDim.x + threadIdx.x;
        ull stride = blockDim.x * gridDim.x;

        for (ull e_clause = index; e_clause < (ull)N * (ull)TOTAL_CLAUSES; e_clause += stride) {
            ull e = e_clause / TOTAL_CLAUSES;
            ull clause = e_clause % TOTAL_CLAUSES;

            const int8_t* X_sample = &X[e * HEIGHT * WIDTH * DEPTH];

            // Read pre-computed sparse representation
            const int* clause_feats_pos = &included_feats_pos[clause * N_FEATURE_FEATS];
            const int* clause_feats_neg = &included_feats_neg[clause * N_FEATURE_FEATS];
            int clause_n_feats_pos = n_feats_pos[clause];
            int clause_n_feats_neg = n_feats_neg[clause];
            const int* clause_patch_range = &patch_ranges[clause * 4];
            int min_row = clause_patch_range[0];
            int max_row = clause_patch_range[1];
            int min_col = clause_patch_range[2];
            int max_col = clause_patch_range[3];
            uint total_includes = num_includes[clause];

            // Empty clauses are skipped during inference
            if (total_includes == 0) {
                continue;
            }

            // Check if valid range is empty (no patches can match)
            if (min_row >= max_row || min_col >= max_col) {
                continue;
            }

            // Check if clause matches any patch (early exit on first match)
            bool clause_matched = false;

            for (int patch_row = min_row; patch_row < max_row && !clause_matched; patch_row++) {
                for (int patch_col = min_col; patch_col < max_col && !clause_matched; patch_col++) {
                    bool matches = true;

                    // Check positive feature literals
                    for (int i = 0; matches && i < clause_n_feats_pos; ++i) {
                        int lit = N_POSITION_FEATS + clause_feats_pos[i];
                        if (get_X_fid(X_sample, patch_row, patch_col, lit) != 1) matches = false;
                    }

#if NEGATED_LITERALS
                    // Check negated feature literals
                    for (int i = 0; matches && i < clause_n_feats_neg; ++i) {
                        int lit = N_POSITION_FEATS + clause_feats_neg[i];
                        if (get_X_fid(X_sample, patch_row, patch_col, lit) != 0) matches = false;
                    }
#endif

                    if (matches) clause_matched = true;
                }
            }

            // Add votes if clause matched
            if (clause_matched) {
                ull class_id, rel_clause = clause % CLAUSES_PER_CLASS;
                LOOP_CLASS_ID(class_id, clause) {
                    atomicAdd(&class_sums[e * CLASSES + class_id],
                              clause_weights[class_id * CLAUSES_PER_CLASS + rel_clause]);
                }
            }
        }
    }

    //==========================================================================
    // Fused 2-kernel approach for training
    //==========================================================================

    /**
     * Fused kernel 1: eval_clauses + count_votes
     * Evaluates all clauses for a sample and counts votes atomically.
     *
     * Parallelism: 1 thread per clause
     *
     * Inputs:
     *   rng => curand RNG states
     *   X => (N * HEIGHT * WIDTH * DEPTH) - batch of samples
     *   e => sample index within batch
     *   global_ta_states => (TOTAL_CLAUSES * LITERALS)
     *   clause_drop_mask => (TOTAL_CLAUSES)
     *   clause_weights => (CLASSES * CLAUSES_PER_CLASS)
     *
     * Outputs:
     *   selected_patch_ids => (TOTAL_CLAUSES) - selected patch per clause (-1 if no match)
     *   patch_ranges => (TOTAL_CLAUSES * 4) - [min_row, max_row, min_col, max_col] for update
     *   num_includes => (TOTAL_CLAUSES) - number of included literals
     *   patch_weights => (TOTAL_CLAUSES * PATCHES) - incremented for selected patches
     *   pos_votes => (CLASSES) - positive polarity votes (atomically accumulated)
     *   neg_votes => (CLASSES) - negative polarity votes (atomically accumulated)
     */
    __global__ void eval_and_count(curandState* rng, const int8_t* X, const int e, const uint* global_ta_states,
                                   const int8_t* clause_drop_mask, const float* clause_weights, int* selected_patch_ids,
                                   int* patch_ranges, uint* num_includes, int* patch_weights, float* pos_votes,
                                   float* neg_votes) {
        ull index = blockIdx.x * blockDim.x + threadIdx.x;
        ull stride = blockDim.x * gridDim.x;

        // Index into batch
        const int8_t* X_sample = &X[e * HEIGHT * WIDTH * DEPTH];

        curandState localRNG = rng[index];

        for (ull clause = index; clause < TOTAL_CLAUSES; clause += stride) {
            // Skip dropped clauses
            if (clause_drop_mask[clause] == 1) {
                selected_patch_ids[clause] = -1;
                num_includes[clause] = 0;
                continue;
            }

            const uint* ta_state = &global_ta_states[clause * LITERALS];

            int included_feats_pos[N_FEATURE_FEATS];
            int included_feats_neg[N_FEATURE_FEATS];
            int n_feats_pos, n_feats_neg;
            int min_row, max_row, min_col, max_col;

            uint total_includes = scan_clause(ta_state, included_feats_pos, included_feats_neg,
                                              &n_feats_pos, &n_feats_neg, &min_row, &max_row, &min_col, &max_col);

            int* clause_patch_range = &patch_ranges[clause * 4];
            clause_patch_range[0] = min_row;
            clause_patch_range[1] = max_row;
            clause_patch_range[2] = min_col;
            clause_patch_range[3] = max_col;
            num_includes[clause] = total_includes;

            if (total_includes == 0) {
                int selected_id = (int)(curand_uniform(&localRNG) * PATCHES);
                selected_patch_ids[clause] = selected_id;

#if TRACK_PATCH_WEIGHTS
                patch_weights[clause * PATCHES + selected_id]++;
#endif
                ull class_id, rel_clause = clause % CLAUSES_PER_CLASS;
                LOOP_CLASS_ID(class_id, clause) {
                    float w = clause_weights[class_id * CLAUSES_PER_CLASS + rel_clause];
                    if (w >= 0)
                        atomicAdd(&pos_votes[class_id], w);
                    else
                        atomicAdd(&neg_votes[class_id], w);
                }
                continue;
            }

            if (min_row >= max_row || min_col >= max_col) {
                selected_patch_ids[clause] = -1;
                continue;
            }

            int selected_patch = -1;
            int active_patch_count = 0;

            for (int patch_row = min_row; patch_row < max_row; patch_row++) {
                for (int patch_col = min_col; patch_col < max_col; patch_col++) {
                    bool matches = true;

                    for (int i = 0; matches && i < n_feats_pos; ++i) {
                        int lit = N_POSITION_FEATS + included_feats_pos[i];
                        if (get_X_fid(X_sample, patch_row, patch_col, lit) != 1) matches = false;
                    }

#if NEGATED_LITERALS
                    for (int i = 0; matches && i < n_feats_neg; ++i) {
                        int lit = N_POSITION_FEATS + included_feats_neg[i];
                        if (get_X_fid(X_sample, patch_row, patch_col, lit) != 0) matches = false;
                    }
#endif

                    if (matches) {
                        active_patch_count++;
                        if (curand_uniform(&localRNG) < 1.0f / active_patch_count) {
                            selected_patch = patch_row * N_PATCHES_X + patch_col;
                        }
                    }
                }
            }

            selected_patch_ids[clause] = selected_patch;

            if (selected_patch != -1) {
#if TRACK_PATCH_WEIGHTS
                patch_weights[clause * PATCHES + selected_patch]++;
#endif
                ull class_id, rel_clause = clause % CLAUSES_PER_CLASS;
                LOOP_CLASS_ID(class_id, clause) {
                    float w = clause_weights[class_id * CLAUSES_PER_CLASS + rel_clause];
                    if (w >= 0)
                        atomicAdd(&pos_votes[class_id], w);
                    else
                        atomicAdd(&neg_votes[class_id], w);
                }
            }
        }

        rng[index] = localRNG;
    }

    /**
     * Fused kernel 2: calc_update_prob + update_clauses
     * Computes update probabilities and applies feedback in one kernel.
     *
     * Parallelism: 1 thread per clause
     *
     * Inputs:
     *   rng => curand RNG states
     *   selected_patch_ids => (TOTAL_CLAUSES) - from eval_and_count
     *   patch_ranges => (TOTAL_CLAUSES * 4) - from eval_and_count
     *   num_includes => (TOTAL_CLAUSES) - from eval_and_count
     *   clause_drop_mask => (TOTAL_CLAUSES)
     *   X => (N * HEIGHT * WIDTH * DEPTH) - batch of samples
     *   targets => (N * CLASSES) - targets for batch
     *   e => sample index within batch
     *   pos_votes => (CLASSES) - positive polarity votes
     *   neg_votes => (CLASSES) - negative polarity votes
     *
     * Outputs:
     *   global_ta_states => (TOTAL_CLAUSES * LITERALS) - modified in place
     *   clause_weights => (CLASSES * CLAUSES_PER_CLASS) - modified in place
     */
    __global__ void prob_and_update(curandState* rng, const int* selected_patch_ids, const int* patch_ranges,
                                    const uint* num_includes, const int8_t* clause_drop_mask, const int8_t* X,
                                    const float* targets, const int e, const float* pos_votes, const float* neg_votes,
                                    uint* global_ta_states, float* clause_weights) {
        ull index = blockIdx.x * blockDim.x + threadIdx.x;
        ull stride = blockDim.x * gridDim.x;

        // Index into batch
        const int8_t* X_sample = &X[e * HEIGHT * WIDTH * DEPTH];
        const float* targets_sample = &targets[e * CLASSES];

        curandState localRNG = rng[index];

        for (ull clause = index; clause < TOTAL_CLAUSES; clause += stride) {
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

            uint clause_num_includes = num_includes[clause];

            // Process each class
            ull class_id, rel_clause = clause % CLAUSES_PER_CLASS;
            LOOP_CLASS_ID(class_id, clause) {
                float q_prob = targets_sample[class_id];
                if (q_prob == 0.0f || curand_uniform(&localRNG) > fabsf(q_prob)) continue;
                int local_target = (q_prob > 0.0f) ? 1 : -1;

                // Compute update probability for this class (inlined from calc_update_prob)
                float y = (float)THRESH * (local_target > 0 ? 1.0f : -1.0f);
                float class_sum = (float)CLIP(pos_votes[class_id] + neg_votes[class_id], -THRESH, THRESH);
                float update_prob = uprob_fun(class_sum, y);

                // Determine sign based on clause weight
                float* local_weight = &clause_weights[class_id * CLAUSES_PER_CLASS + rel_clause];
                int sign = (*local_weight >= 0) - (*local_weight < 0);

                bool should_update = (curand_uniform(&localRNG) <= update_prob);
                bool clause_has_space = (clause_num_includes <= (uint)MAX_INCLUDED_LITERALS);
                bool t1 = (local_target * sign) > 0;

                // Type 1a feedback - TP - clause is active with correct polarity and has space
                if (should_update && t1 && local_clause_output && clause_has_space) {
                    type1a_fb(&localRNG, ta_state, local_weight, X_sample, patch_row, patch_col, sign);
                }

                // Type 1b feedback - FN - clause is inactive or overflowing, but should have been active
                if (should_update && t1 && !(local_clause_output && clause_has_space)) {
                    type1b_fb(&localRNG, ta_state, sign);
                }

                // Type 2 feedback - FP - clause is active but has wrong polarity
                if (should_update && (local_target * sign) < 0 && local_clause_output) {
                    type2_fb(ta_state, local_weight, X_sample, patch_row, patch_col, sign);
                }
            }
        }

        rng[index] = localRNG;
    }
}
