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
#define NUM_LITERAL_CHUNKS ((LITERALS + INT_SIZE - 1) / INT_SIZE)

#define N_PATCHES_Y (HEIGHT - PATCH_HEIGHT + 1)
#define N_PATCHES_X (WIDTH - PATCH_WIDTH + 1)
#define PATCHES (N_PATCHES_Y * N_PATCHES_X)

#if ((LITERALS % INT_SIZE) != 0)
    #define FILTER (~(0xFFFFFFFF << (LITERALS % INT_SIZE)))
#else
    #define FILTER 0xFFFFFFFF
#endif

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

// Probabilistically decrement literals in range [start, end) with probability p
// offset is added to index (use LITERALS/2 for negated, 0 otherwise)
static inline void literal_dec_with_p(uint* restrict rng, uint* restrict ta_state, int start, int end, int offset,
                                      float p) {
    int li = start + geometric_sample(rng, p) - 1;
    while (li < end) {
        if (ta_state[li + offset] > 0) ta_state[li + offset] -= 1;
        li += geometric_sample(rng, p);
    }
}

// Increment literals in range [start, end) up to max_val (branchless, SIMD-friendly)
// offset is added to index (use LITERALS/2 for negated, 0 otherwise)
static inline void literal_inc(uint* restrict ta_state, int start, int end, int offset, uint max_val) {
    OMP_SIMD
    for (int li = start; li < end; ++li) {
        ta_state[li + offset] += (ta_state[li + offset] < max_val);
    }
}

static inline int clause_match_fun(const uint* restrict ta_state, const uint* restrict X) {
    for (int chunk = 0; chunk < NUM_LITERAL_CHUNKS - 1; ++chunk)
        if ((ta_state[chunk] & X[chunk]) != ta_state[chunk]) return 0;
    if ((ta_state[NUM_LITERAL_CHUNKS - 1] & (X[NUM_LITERAL_CHUNKS - 1] & FILTER)) !=
        (ta_state[NUM_LITERAL_CHUNKS - 1] & FILTER))
        return 0;

    return 1;
}

static inline void type1a_fb(uint* restrict rng, uint* restrict ta_state, float* restrict weight,
                             const uint* restrict patch, const int sign) {
#if TYPE1A_FB
    float s_inv = S_INV;
    #if WEIGHTED
    if (fabs(*weight) < MAX_WEIGHT) (*weight) += sign * 1.0f;
    #endif

    for (int chunk = 0; chunk < NUM_LITERAL_CHUNKS; ++chunk) {
        uint patch_bits = patch[chunk];
        for (int bit = 0; bit < INT_SIZE && (chunk * INT_SIZE + bit) < LITERALS; ++bit) {
            uint patch_bit = (patch_bits >> bit) & 1u;
            int li = chunk * INT_SIZE + bit;
            if (patch_bit == 1 && ta_state[li] < MAX_TA_STATE) {
                ta_state[li] += 1;
            } else if (patch_bit == 0 && ta_state[li] > 0 && xorshift32(rng) <= s_inv) {
                ta_state[li] -= 1;
            }
        }
    }

#endif
}

static inline void type1b_fb(uint* restrict rng, uint* restrict ta_state, const int sign) {
#if TYPE1B_FB
    literal_dec_with_p(rng, ta_state, 0, LITERALS, 0, S_INV);
#endif
}

static inline void type2_fb(uint* restrict ta_state, float* restrict weight, const uint* restrict patch,
                            const int sign) {
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

    for (int chunk = 0; chunk < NUM_LITERAL_CHUNKS; ++chunk) {
        uint patch_bits = patch[chunk];
        for (int bit = 0; bit < INT_SIZE && (chunk * INT_SIZE + bit) < LITERALS; ++bit) {
            uint patch_bit = (patch_bits >> bit) & 1u;
            int li = chunk * INT_SIZE + bit;
            if (patch_bit == 0 && ta_state[li] < INCLUDE_STATE) {
                ta_state[li] += 1;
            }
        }
    }

#endif
}

static inline float uprob_fun(float v, float y) {
    float prob = (y - v) / (2 * y);
    return prob;
}

static inline void set_bit(uint* arr, int bit) { arr[bit / INT_SIZE] |= (1u << (bit % INT_SIZE)); }
static inline void unset_bit(uint* arr, int bit) { arr[bit / INT_SIZE] &= ~(1u << (bit % INT_SIZE)); }

void encode(const int8_t* restrict X, const int N, uint* restrict encoded_X) {
    /*
     * Inputs:
     * X => (N * HEIGHT * WIDTH * DEPTH)
     * N => Number of examples in the batch
     *
     * Outputs:
     * encoded_X => (N * PATCHES * NUM_LITERAL_CHUNKS)
     */
    OMP_PARALLEL_FOR
    for (ull e_patch = 0; e_patch < (ull)(PATCHES * N); e_patch++) {
        ull e = e_patch / PATCHES;
        ull patch_id = e_patch % PATCHES;

        // Calculate the starting point of the patch in the original image
        int patch_row = patch_id / N_PATCHES_X;
        int patch_col = patch_id % N_PATCHES_X;
        // Patch is at coord (patch_row, patch _col)

        ull encX_offset = e * (ull)(PATCHES * NUM_LITERAL_CHUNKS) + patch_id * (ull)NUM_LITERAL_CHUNKS;
        uint* patch_output = &encoded_X[encX_offset];

        // Initialization.
        // By default, all values in encoded_X are set to 0 (in python code).
#if NEGATED_LITERALS
        // So, only need to initialize all negated literals to 1.
        for (int literal = LITERALS / 2; literal < LITERALS; ++literal) {
            set_bit(patch_output, literal);
        }
#endif

        // Encoding the location and features of the patch
        // First (HEIGHT - PATCH_HEIGHT) literals encode the row number.
        for (int i = 0; i < patch_row; ++i) {
            set_bit(patch_output, i);
#if NEGATED_LITERALS
            unset_bit(patch_output, i + (LITERALS / 2));
#endif
        }

        // Next (WIDTH - PATCH_WIDTH) literals encode the column number.
        for (int i = 0; i < patch_col; ++i) {
            set_bit(patch_output, (HEIGHT - PATCH_HEIGHT) + i);
#if NEGATED_LITERALS
            unset_bit(patch_output, (HEIGHT - PATCH_HEIGHT) + i + (LITERALS / 2));
#endif
        }

        // Next N_FEATURE_FEATS literals encode the features of the patch.
        for (int fid = 0; fid < N_FEATURE_FEATS; ++fid) {
            ull rel_y = fid / (ull)(PATCH_WIDTH * DEPTH);
            ull rem = fid % (ull)(PATCH_WIDTH * DEPTH);
            ull rel_x = rem / DEPTH;
            ull z = rem % DEPTH;

            ull abs_y = patch_row + rel_y;
            ull abs_x = patch_col + rel_x;

            ull feat = X[e * (ull)(HEIGHT * WIDTH * DEPTH) + abs_y * (ull)(WIDTH * DEPTH) + abs_x * (ull)DEPTH + z];

            int lid = N_POSITION_FEATS + fid;
            if (feat == 1) {
                set_bit(patch_output, lid);
#if NEGATED_LITERALS
                unset_bit(patch_output, lid + (LITERALS / 2));
#endif
            } else if (feat == 0) {
                // No need to do anything, negated is already 1 and non-negated is already 0.
            }
        }
    }
}

void pack_clauses(const uint* restrict global_ta_states, uint* restrict packed_clauses, uint* restrict num_includes) {
    /*
     * Pack the TA states into chunks of 32 bits. Each chunk represents a set of 32 literals.
     * The number of included literals is also calculated here.
     *
     * Inputs:
     * global_ta_states => (TOTAL_CLAUSES * LITERALS)
     *
     * Outputs:
     * packed_clauses => (TOTAL_CLAUSES * NUM_LITERAL_CHUNKS)
     * num_includes => (TOTAL_CLAUSES)
     */

    OMP_PARALLEL_FOR
    for (ull clause = 0; clause < TOTAL_CLAUSES; clause++) {
        const uint* ta_state = &global_ta_states[clause * LITERALS];
        uint total_count = 0;
        uint* packed_clause = &packed_clauses[clause * NUM_LITERAL_CHUNKS];
        for (int li = 0; li < LITERALS; ++li) {
            if (ta_state[li] >= INCLUDE_STATE) {
                packed_clause[li / INT_SIZE] |= (1u << (li % INT_SIZE));
                total_count++;
            }
        }
        num_includes[clause] = total_count;
    }
}

void eval_clauses(const uint* restrict packed_clauses, const uint* restrict num_includes,
                  const int8_t* restrict clause_drop_mask, const uint* restrict encoded_X, const int e,
                  int8_t* restrict clause_outputs) {
    /*
     * Evaluate each clause on the input `e`.
     *
     * Inputs:
     * packed_ta_states => (TOTAL_CLAUSES * NUM_LITERAL_CHUNKS)
     * num_includes => (TOTAL_CLAUSES)
     * clause_drop_mask => (TOTAL_CLAUSES)
     * encoded_X => (N * PATCHES * NUM_LITERAL_CHUNKS)
     * e => index of the example in the batch to evaluate on
     *
     * Outputs:
     * clause_outputs => (TOTAL_CLAUSES * PATCHES)
     *
     */

    OMP_PARALLEL_FOR
    for (ull clause_patch = 0; clause_patch < (ull)TOTAL_CLAUSES * (ull)PATCHES; clause_patch++) {
        int8_t* clause_output = &clause_outputs[clause_patch];

        ull clause = clause_patch / PATCHES;
        ull patch_id = clause_patch % PATCHES;

        // Skip dropped clauses
        if (clause_drop_mask[clause] == 1) {
            *clause_output = 0;
            continue;
        }

        // Skip empty clauses
        if (num_includes[clause] == 0) {
            *clause_output = 1;
            continue;
        }

        *clause_output =
            clause_match_fun(&packed_clauses[clause * NUM_LITERAL_CHUNKS],
                             &encoded_X[(ull)e * (ull)(PATCHES * NUM_LITERAL_CHUNKS) + patch_id * NUM_LITERAL_CHUNKS]);
    }
}

void select_patch_and_count_votes(uint* restrict rng, const float* restrict clause_weights,
                                  const int8_t* restrict clause_outputs, int* restrict patch_weights,
                                  int* restrict selected_patch_ids, float* restrict pos_votes,
                                  float* restrict neg_votes) {
    /*
     * Voting.
     *
     * Inputs:
     * rng => RNG
     * clause_weights => (CLASSES * CLAUSES_PER_CLASS)
     * clause_outputs => (TOTAL_CLAUSES * PATCHES)
     *
     * Outputs:
     * patch_weights => (TOTAL_CLAUSES * PATCHES)
     * selected_patch_ids => (TOTAL_CLAUSES)
     * pos_votes => (CLASSES)
     * neg_votes => (CLASSES)
     *
     */

    OMP_PARALLEL_FOR
    for (ull clause = 0; clause < TOTAL_CLAUSES; clause++) {
#if PATCHES > 1
        int count = 0;
        int selected_id = -1;
        for (int patch_id = 0; patch_id < PATCHES; ++patch_id) {
            if (clause_outputs[clause * PATCHES + patch_id]) {
                count++;
                if (xorshift32(&rng[GET_THREAD_ID]) < 1.0f / count) {
                    selected_id = patch_id;
                }
            }
        }
#else
        int selected_id = clause_outputs[clause] ? 0 : -1;
#endif
        selected_patch_ids[clause] = selected_id;
        if (selected_id != -1) {
#if TRACK_PATCH_WEIGHTS
            patch_weights[clause * PATCHES + selected_id]++;
#endif
            ull class_id, rel_clause = clause % CLAUSES_PER_CLASS;
            LOOP_CLASS_ID(class_id, clause) {
                float w = clause_weights[class_id * CLAUSES_PER_CLASS + rel_clause];
                if (w >= 0)
                    // Positive polarity clauses
                    OMP_ATOMIC
                pos_votes[class_id] += w;
                else
                    // Negative polarity clauses
                    OMP_ATOMIC neg_votes[class_id] += w;
            }
        }
    }
}

void evidence_to_update_prob(const float* restrict pos_votes, const float* restrict neg_votes,
                             const int8_t* restrict targets, const int e, float* restrict prob) {
    /*
     * Convert the votes to update probability.
     *
     * Inputs:
     * pos_votes => (CLASSES)
     * neg_votes => (CLASSES)
     * targets => (N * CLASSES)
     * e => sample index
     *
     * Outputs:
     * prob => (CLASSES)
     */

    OMP_PARALLEL_FOR
    for (ull class_id = 0; class_id < CLASSES; class_id++) {
        int local_target = targets[e * CLASSES + class_id];
        if (local_target == 0) {
            prob[class_id] = 0.0f;
            continue;
        }

        float y = (float)THRESH * (float)local_target;
        float class_sum = (float)CLIP(pos_votes[class_id] + neg_votes[class_id], -THRESH, THRESH);
        prob[class_id] = uprob_fun(class_sum, y);
    }
}

void update_clauses(uint* restrict rng, const int* restrict selected_patch_ids, const uint* restrict num_includes,
                    const int8_t* restrict clause_drop_mask, const uint* restrict encoded_X,
                    const int8_t* restrict targets, const float* restrict prob, const int e,
                    uint* restrict global_ta_states, float* restrict clause_weights) {
    /*
     * Update clauses.
     *
     * Inputs:
     * rng => RNG
     * selected_patch_ids => (TOTAL_CLAUSES)
     * num_includes => (TOTAL_CLAUSES)
     * clause_drop_mask => (TOTAL_CLAUSES)
     * encoded_X => (N * PATCHES * NUM_LITERAL_CHUNKS)
     * targets => (N * CLASSES)
     * prob => (CLASSES)
     * e => sample index
     *
     * Outputs:
     * global_ta_states => (TOTAL_CLAUSES * LITERALS)
     * clause_weights => (CLASSES * CLAUSES_PER_CLASS)
     */

    OMP_PARALLEL_FOR
    for (ull clause = 0; clause < TOTAL_CLAUSES; clause++) {
        // Skip dropped clauses
        if (clause_drop_mask[clause] == 1) continue;

        uint* ta_state = &global_ta_states[clause * LITERALS];
        int local_clause_output = selected_patch_ids[clause] > -1 ? 1 : 0;
        const uint* X = &encoded_X[(ull)e * (ull)(PATCHES * NUM_LITERAL_CHUNKS)];
        const uint* patch =
            selected_patch_ids[clause] > -1 ? &X[selected_patch_ids[clause] * NUM_LITERAL_CHUNKS] : NULL;

        ull class_id, rel_clause = clause % CLAUSES_PER_CLASS;
        LOOP_CLASS_ID(class_id, clause) {
            int local_target = targets[e * CLASSES + class_id];
            if (local_target == 0) continue;

            float* local_weight = &clause_weights[class_id * CLAUSES_PER_CLASS + rel_clause];
            int sign = (*local_weight >= 0) - (*local_weight < 0);

            float update_prob = prob[class_id];
            bool should_update = (xorshift32(&rng[GET_THREAD_ID]) <= update_prob);
            bool clause_has_space = (num_includes[clause] <= MAX_INCLUDED_LITERALS);
            bool t1 = (local_target * sign) > 0;

            // Type 1a feedback - TP - if the clause is active and has the correct polarity for the target class,
            // and has space
            if (should_update && t1 && local_clause_output && clause_has_space) {
                type1a_fb(&rng[GET_THREAD_ID], ta_state, local_weight, patch, sign);
            }

            // Type 1b feedback - FN - If clause is inactive, but should have been active (has correct polarity for
            // target), OR if the clause is not overflowing
            if (should_update && t1 && !(local_clause_output && clause_has_space)) {
                type1b_fb(&rng[GET_THREAD_ID], ta_state, sign);
            }

            // Type 2 feedback - FP - if the clause is active, but has the wrong polarity for the target class
            if (should_update && (local_target * sign) < 0 && local_clause_output) {
                type2_fb(ta_state, local_weight, patch, sign);
            }
        }
    }
}

void clause_inference(const uint* restrict packed_clauses, const float* restrict clause_weights,
                      const uint* restrict num_includes, const uint* restrict encoded_X, const int N,
                      float* restrict class_sums) {
    /*
     * Inference. Faster inference kernel, that parallelizes over all the samples, and calculates the class sums.
     *
     * Inputs:
     * packed_ta_states => (TOTAL_CLAUSES * NUM_LITERAL_CHUNKS)
     * clause_weights => (CLASSES * CLAUSES_PER_CLASS)
     * num_includes => (TOTAL_CLAUSES)
     * encoded_X => (N * PATCHES * NUM_LITERAL_CHUNKS)
     * N => Number of examples
     *
     * Outputs:
     * class_sums => (N * CLASSES)
     */

    OMP_PARALLEL_FOR
    for (ull e_clause = 0; e_clause < (ull)N * (ull)TOTAL_CLAUSES; e_clause++) {
        ull e = e_clause / TOTAL_CLAUSES;
        ull clause = e_clause % TOTAL_CLAUSES;
        if (num_includes[clause] == 0) continue;  // Skip empty clauses
        int clause_output = 0;
        for (int patch_id = 0; patch_id < PATCHES; ++patch_id) {
            if (clause_match_fun(&packed_clauses[clause * NUM_LITERAL_CHUNKS],
                                 &encoded_X[e * (ull)(PATCHES * NUM_LITERAL_CHUNKS) + patch_id * NUM_LITERAL_CHUNKS])) {
                clause_output = 1;
                break;
            }
        }
        if (clause_output) {
            ull class_id, rel_clause = clause % CLAUSES_PER_CLASS;
            LOOP_CLASS_ID(class_id, clause) {
                OMP_ATOMIC
                class_sums[e * CLASSES + class_id] += clause_weights[class_id * CLAUSES_PER_CLASS + rel_clause];
            }
        }
    }
}
