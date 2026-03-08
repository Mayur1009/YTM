#ifdef IS_NEOVIM_CLANGD_ENV
    #define USE_OMP 1
    #define TOTAL_CLAUSES 1000
    #define THRESH 100
    #define S 10.0
    #define CLASSES 10
    #define DIM0 28
    #define DIM1 28
    #define DIM2 1
    #define PATCH_DIM0 10
    #define PATCH_DIM1 10
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
    #define PATCHES 361
    #define LITERALS 272
#endif

#define INT_SIZE 32
#define NUM_LITERAL_CHUNKS ((LITERALS + INT_SIZE - 1) / INT_SIZE)
#define S_INV (1.0f / S)

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
    #define GET_THREAD_ID omp_get_thread_num()
void set_num_threads(int num_threads) { omp_set_num_threads(num_threads); }
#else
    #define OMP_PARALLEL_FOR
    #define OMP_ATOMIC
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
    float s_inv = S_INV;
    for (int li = 0; li < LITERALS; ++li) {
        if (ta_state[li] > 0 && xorshift32(rng) <= s_inv) {
            ta_state[li] -= 1;
        }
    }
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

void encode(const int8_t* restrict X, const int N, uint* restrict encoded_X) {
    /*
     * Inputs:
     * X => (N * DIM0 * DIM1 * DIM2)
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
        int patch_coordinate_y = patch_id / (DIM0 - PATCH_DIM0 + 1);
        int patch_coordinate_x = patch_id % (DIM0 - PATCH_DIM0 + 1);

        ull encX_offset = e * (ull)(PATCHES * NUM_LITERAL_CHUNKS) + patch_id * (ull)NUM_LITERAL_CHUNKS;
        uint* patch_output = &encoded_X[encX_offset];

        // Initialization.
        // By default, all values in encoded_X are set to 0 (in python code).
#if NEGATED_LITERALS
        // So, only need to initialize all negated literals to 1.
        for (int literal = LITERALS / 2; literal < LITERALS; ++literal) {
            int chunk_nr = literal / INT_SIZE;
            int chunk_pos = literal % INT_SIZE;
            patch_output[chunk_nr] |= (1u << chunk_pos);
        }
#endif

        // Encoding the location of the patch with thermometer encoding
        for (int lit = 0; lit < patch_coordinate_y; ++lit) {
            int chunk_nr = lit / INT_SIZE;
            int chunk_pos = lit % INT_SIZE;
            patch_output[chunk_nr] |= (1u << chunk_pos);
#if NEGATED_LITERALS
            int neg_chunk_nr = (lit + (LITERALS / 2)) / INT_SIZE;
            int neg_chunk_pos = (lit + (LITERALS / 2)) % INT_SIZE;
            patch_output[neg_chunk_nr] &= ~(1u << neg_chunk_pos);
#endif
        }

        for (int lit = 0; lit < patch_coordinate_x; ++lit) {
            int chunk_nr = (DIM1 - PATCH_DIM1 + lit) / INT_SIZE;
            int chunk_pos = (DIM1 - PATCH_DIM1 + lit) % INT_SIZE;
            patch_output[chunk_nr] |= (1u << chunk_pos);
#if NEGATED_LITERALS
            int neg_chunk_nr = ((DIM1 - PATCH_DIM1 + lit) + (LITERALS / 2)) / INT_SIZE;
            int neg_chunk_pos = ((DIM1 - PATCH_DIM1 + lit) + (LITERALS / 2)) % INT_SIZE;
            patch_output[neg_chunk_nr] &= ~(1u << neg_chunk_pos);
#endif
        }

        // Iterate over features in a patch, that are either 1 (present) or 0(absent)
        // taken care of in the initialization.
        for (ull p_y = patch_coordinate_y; p_y < patch_coordinate_y + PATCH_DIM1; ++p_y) {
            for (ull p_x = patch_coordinate_x; p_x < patch_coordinate_x + PATCH_DIM0; ++p_x) {
                for (int z = 0; z < DIM2; ++z) {
                    ull dense_idx = e * (ull)(DIM0 * DIM1 * DIM2) + p_y * (ull)(DIM0 * DIM2) + p_x * (ull)DIM2 + z;

                    int rel_y = p_y - patch_coordinate_y;
                    int rel_x = p_x - patch_coordinate_x;
#if POSITION_LITERALS
                    int patch_pos =
                        (DIM1 - PATCH_DIM1) + (DIM0 - PATCH_DIM0) + rel_y * PATCH_DIM0 * DIM2 + rel_x * DIM2 + z;
#else
                    int patch_pos = rel_y * PATCH_DIM0 * DIM2 + rel_x * DIM2 + z;
#endif
                    if (X[dense_idx] == 1) {
                        int chunk_nr = patch_pos / INT_SIZE;
                        int chunk_pos = patch_pos % INT_SIZE;
                        patch_output[chunk_nr] |= (1u << chunk_pos);
#if NEGATED_LITERALS
                        int neg_chunk_nr = (patch_pos + (LITERALS / 2)) / INT_SIZE;
                        int neg_chunk_pos = (patch_pos + (LITERALS / 2)) % INT_SIZE;
                        patch_output[neg_chunk_nr] &= ~(1u << neg_chunk_pos);
#endif
                    } else if (X[dense_idx] == 0) {
                        // No need to do anything, negated is already 1 and non-negated is already 0.
                    }
                }
            }
        }
    }
}

void decode(const uint* restrict encoded_X, const int N, int8_t* restrict X) {
    /*
     * Mainly for testing and debugging purposes, to verify that encoding and decoding are consistent. Completely
     * ignores the negations, assumes the input to be encoded only using the encode function.
     *
     * Inputs: encoded_X =>
     * (N * PATCHES * NUM_LITERAL_CHUNKS) N => Number of examples in the batch
     *
     * Outputs:
     * X => (N * DIM0 * DIM1 * DIM2)
     */

    OMP_PARALLEL_FOR
    for (ull e_patch = 0; e_patch < (ull)(PATCHES * N); e_patch++) {
        ull e = e_patch / PATCHES;
        ull patch_id = e_patch % PATCHES;

        ull encX_offset = e * (ull)(PATCHES * NUM_LITERAL_CHUNKS) + patch_id * (ull)NUM_LITERAL_CHUNKS;
        const uint* patch_input = &encoded_X[encX_offset];

        // Calculate the starting point of the patch in the original image
        int patch_coordinate_y = patch_id / (DIM0 - PATCH_DIM0 + 1);
        int patch_coordinate_x = patch_id % (DIM0 - PATCH_DIM0 + 1);

        for (ull p_y = patch_coordinate_y; p_y < patch_coordinate_y + PATCH_DIM1; ++p_y) {
            for (ull p_x = patch_coordinate_x; p_x < patch_coordinate_x + PATCH_DIM0; ++p_x) {
                for (int z = 0; z < DIM2; ++z) {
                    ull dense_idx = e * (ull)(DIM0 * DIM1 * DIM2) + p_y * (ull)(DIM0 * DIM2) + p_x * (ull)DIM2 + z;

                    int rel_y = p_y - patch_coordinate_y;
                    int rel_x = p_x - patch_coordinate_x;
#if POSITION_LITERALS
                    int patch_pos =
                        (DIM1 - PATCH_DIM1) + (DIM0 - PATCH_DIM0) + rel_y * PATCH_DIM0 * DIM2 + rel_x * DIM2 + z;
#else
                    int patch_pos = rel_y * PATCH_DIM0 * DIM2 + rel_x * DIM2 + z;
#endif

                    int chunk_nr = patch_pos / INT_SIZE;
                    int chunk_pos = patch_pos % INT_SIZE;
                    if ((patch_input[chunk_nr] & (1u << chunk_pos)) != 0) {
                        X[dense_idx] = 1;
                    } else {
                        X[dense_idx] = 0;
                    }
                }
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
                  uint* restrict clause_outputs) {
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
        uint* clause_output = &clause_outputs[clause_patch];

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
                                  const uint* restrict clause_outputs, int* restrict patch_weights,
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
            patch_weights[clause * PATCHES + selected_id]++;
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
