// Ignore this block, it is used to only for neovim clangd lsp.
#ifdef IS_NEOVIM_CLANGD_ENV
#define CLAUSES 100
#define THRESH 500
#define S 10
#define Q 1
#define DIM0 28
#define DIM1 28
#define DIM2 1
#define PATCH_DIM0 10
#define PATCH_DIM1 10
#define PATCHES 361
#define LITERALS 272
#define MAX_INCLUDED_LITERALS LITERALS
#define APPEND_NEGATED 1
#define NEGATIVE_CLAUSES 1
#define CLASSES 10
#define MAX_TA_STATE 255
#define ENCODE_LOC 1
#endif

#include <curand_kernel.h>

#define VECTORIZED_LIMIT (LITERALS & ~3)
#define S_INV (1.0f / S)
#define Q_PROB (1.0f * Q / max(1, CLASSES - 1))
#define HALF_STATE (MAX_TA_STATE / 2)
#define INT_SIZE 32
#define NUM_LITERAL_CHUNKS (((LITERALS - 1) / INT_SIZE) + 1)
#if ((LITERALS % INT_SIZE) != 0)
#define FILTER (~(0xFFFFFFFF << (LITERALS % INT_SIZE)))
#else
#define FILTER 0xFFFFFFFF
#endif

extern "C" {
__global__ void initialize(curandState *rng, unsigned int *global_ta_states, float *clause_weights) {
    /*
     * Initialize the TA states to middle state and clause weights(randomly to -1 and 1).
     */
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    curandState localState = rng[index];

    for (int clause = index; clause < CLAUSES; clause += stride) {
        for (int li = 0; li < LITERALS; ++li) {
            global_ta_states[clause * LITERALS + li] = HALF_STATE;  // Initialize TA states to 0
        }
        for (int class_id = 0; class_id < CLASSES; ++class_id) {
#if NEGATIVE_CLAUSES
            clause_weights[clause * CLASSES + class_id] = (1.0f - 2.0f * (float)(curand(&localState) % 2));
#else
            clause_weights[clause * CLASSES + class_id] = 1.0f;
#endif
        }
    }
    rng[index] = localState;
}

__global__ void encode(const unsigned int *X_indptr, const unsigned int *X_indices, unsigned int *encoded_X, const int e) {
    /*
     * Encode the input data X into patches. Each patch has a set of literals, which are packed into chunks of 32 bits.
     * TODO: This is done for sample e. But it might be better to process multiple samples at once. Since the input data
     * can be huge, some sort of batching would be better.
     */
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    const unsigned int *indices = &X_indices[X_indptr[e]];
    int number_of_indices = X_indptr[e + 1] - X_indptr[e];

    for (int patch = index; patch < PATCHES; patch += stride) {
        // Pre-calculate patch boundaries once per thread
        int patch_coordinate_y = patch / (DIM0 - PATCH_DIM0 + 1);
        int patch_coordinate_x = patch % (DIM0 - PATCH_DIM0 + 1);

        unsigned int *patch_output = &encoded_X[patch * NUM_LITERAL_CHUNKS];

        for (int k = 0; k < number_of_indices; ++k) {
            int idx = indices[k];
            int y = idx / (DIM0 * DIM2);
            int x = (idx % (DIM0 * DIM2)) / DIM2;
            int z = (idx % (DIM0 * DIM2)) % DIM2;

            // Check if this coordinate falls in this patch
            if (y >= patch_coordinate_y && y < patch_coordinate_y + PATCH_DIM1 && x >= patch_coordinate_x &&
                x < patch_coordinate_x + PATCH_DIM0) {
                // Calculate bit position and set
                int p_y = y - patch_coordinate_y;
                int p_x = x - patch_coordinate_x;

#if ENCODE_LOC
                int patch_pos = (DIM1 - PATCH_DIM1) + (DIM0 - PATCH_DIM0) + p_y * PATCH_DIM0 * DIM2 + p_x * DIM2 + z;
#else
                int patch_pos = p_y * PATCH_DIM0 * DIM2 + p_x * DIM2 + z;
#endif

                int chunk_nr = patch_pos / INT_SIZE;
                int chunk_pos = patch_pos % INT_SIZE;
                patch_output[chunk_nr] |= (1u << chunk_pos);

#if APPEND_NEGATED
                int neg_chunk_nr = (patch_pos + (LITERALS / 2)) / INT_SIZE;
                int neg_chunk_pos = (patch_pos + (LITERALS / 2)) % INT_SIZE;
                patch_output[neg_chunk_nr] &= ~(1u << neg_chunk_pos);
#endif
            }
        }
    }
}


__global__ void pack_clauses(const unsigned int *global_ta_states, unsigned int *packed_clauses, int *num_includes) {
    /*
     * Pack the TA states into chunks of 32 bits. Each chunk represents a set of literals.
     * The number of included literals is also calculated here.
     */
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int clause = index; clause < CLAUSES; clause += stride) {
        const unsigned int *ta_state = &global_ta_states[clause * LITERALS];
        unsigned int *packed_clause = &packed_clauses[clause * NUM_LITERAL_CHUNKS];
        int total_count = 0;

        #pragma unroll 4
        for (int chunk = 0; chunk < NUM_LITERAL_CHUNKS; ++chunk) {
            unsigned int packed_value = 0;
            int start_lit = chunk * INT_SIZE;
            int end_lit = min(start_lit + INT_SIZE, LITERALS);

            int vectorized_end = start_lit + ((end_lit - start_lit) & ~3);  // Ensure vectorized end is a multiple of 4
            for (int li = start_lit; li < vectorized_end; li += 4) {
                uint4 ta_vec = *((uint4 *)&ta_state[li]);
                if (ta_vec.x > HALF_STATE) {
                    packed_value |= (1u << (li % INT_SIZE));
                    total_count++;
                }
                if (ta_vec.y > HALF_STATE) {
                    packed_value |= (1u << ((li + 1) % INT_SIZE));
                    total_count++;
                }
                if (ta_vec.z > HALF_STATE) {
                    packed_value |= (1u << ((li + 2) % INT_SIZE));
                    total_count++;
                }
                if (ta_vec.w > HALF_STATE) {
                    packed_value |= (1u << ((li + 3) % INT_SIZE));
                    total_count++;
                }
            }
            for (int li = vectorized_end; li < end_lit; ++li) {
                if (ta_state[li] > HALF_STATE) {
                    packed_value |= (1u << (li % INT_SIZE));
                    total_count++;
                }
            }
            packed_clause[chunk] = packed_value;
        }
        num_includes[clause] = total_count;
    }
}

__device__ inline int clause_match(const unsigned int *ta_state, const unsigned int *X) {
    /*
     * Check if the TA state matches the patch X.
     * Returns 1 if it matches, 0 otherwise.
     */

    #pragma unroll 4
    for (int chunk = 0; chunk < NUM_LITERAL_CHUNKS - 1; ++chunk)
        if ((ta_state[chunk] & X[chunk]) != ta_state[chunk]) return 0;

    if ((ta_state[NUM_LITERAL_CHUNKS - 1] & (X[NUM_LITERAL_CHUNKS - 1] & FILTER)) !=
        (ta_state[NUM_LITERAL_CHUNKS - 1] & FILTER))
        return 0;

    return 1;
}

__global__ void clause_eval(curandState *rng, const unsigned int *packed_ta_states, const float *clause_weights,
                            const unsigned int *X, int *selected_patch_ids) {
    /*
     * Calculate clause activations and select a patch for each active clause. If a clause is active, the
     * selected_patch_ids will be int between 0 and PATCHES - 1, else it will be -1.
     */
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    curandState localRNG = rng[index];

    for (int clause_patch = index; clause_patch < CLAUSES * PATCHES; clause_patch += stride) {
        int clause = clause_patch / PATCHES;
        int patch_id = clause_patch % PATCHES;

        if (selected_patch_ids[clause] > -1) continue;  // Skip already selected clauses
        int patch_matched =
            clause_match(&packed_ta_states[clause * NUM_LITERAL_CHUNKS], &X[patch_id * NUM_LITERAL_CHUNKS]);
        if (patch_matched && (PATCHES == 1 || curand_uniform(&localRNG) <= 0.5f)) {
            selected_patch_ids[clause] = patch_id;
            break;
        }
    }
    rng[index] = localRNG;
}

__global__ void calc_class_sums(int *selected_patch_ids, const float *clause_weights, float *class_sums) {
    /*
     * Calculate the class sums for each clause. If selected_patch_ids[clause] > -1, then the clause is active.
     * The class_sum is given by, W[:, class] . Clause_activations.
     * Can this dot product be optimized?
     */
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int clause = index; clause < CLAUSES; clause += stride) {
        if (selected_patch_ids[clause] > -1) {
            for (int class_id = 0; class_id < CLASSES; ++class_id) {
                atomicAdd(&class_sums[class_id], clause_weights[clause * CLASSES + class_id]);
            }
        }
    }
}

__device__ inline float clip_cs(float cs) { return (cs > THRESH) ? THRESH : ((cs < -THRESH) ? -THRESH : cs); }

__device__ inline void type1a_fb(curandState *rng, unsigned int *ta_state, float *clause_weight,
                                 const unsigned int *patch, int sign) {
    /*
     * Type Ia feedback - Vectorized version:
     * 1. Increment states for literal present in the patch.
     * 2. Decrement states for literal not present in the patch with probability 1/S.
     * 3. Increase clause weight.
     */

    // Update weight
    (*clause_weight) += sign * 1.0f;

    #pragma unroll 4
    for (int li = 0; li < VECTORIZED_LIMIT; li += 4) {
        uint4 ta_vec = *((uint4 *)&ta_state[li]);
        uint4 patch_vec = {
            (patch[li / INT_SIZE] >> (li % INT_SIZE)) & 1u,
            (patch[(li + 1) / INT_SIZE] >> ((li + 1) % INT_SIZE)) & 1u,
            (patch[(li + 2) / INT_SIZE] >> ((li + 2) % INT_SIZE)) & 1u,
            (patch[(li + 3) / INT_SIZE] >> ((li + 3) % INT_SIZE)) & 1u,
        };

        ta_vec.x += (patch_vec.x == 1 && ta_vec.x < MAX_TA_STATE);
        ta_vec.y += (patch_vec.y == 1 && ta_vec.y < MAX_TA_STATE);
        ta_vec.z += (patch_vec.z == 1 && ta_vec.z < MAX_TA_STATE);
        ta_vec.w += (patch_vec.w == 1 && ta_vec.w < MAX_TA_STATE);

        ta_vec.x -= (patch_vec.x == 0 && ta_vec.x > 0 && curand_uniform(rng) <= S_INV);
        ta_vec.y -= (patch_vec.y == 0 && ta_vec.y > 0 && curand_uniform(rng) <= S_INV);
        ta_vec.z -= (patch_vec.z == 0 && ta_vec.z > 0 && curand_uniform(rng) <= S_INV);
        ta_vec.w -= (patch_vec.w == 0 && ta_vec.w > 0 && curand_uniform(rng) <= S_INV);

        // Write back the vectorized results
        *((uint4 *)&ta_state[li]) = ta_vec;
    }

    // Handle remaining literals (when LITERALS % 4 != 0)
    #pragma unroll 4
    for (int li = VECTORIZED_LIMIT; li < LITERALS; ++li) {
        unsigned int patch_bit = (patch[li / INT_SIZE] >> (li % INT_SIZE)) & 1u;
        if (patch_bit == 1 && ta_state[li] < MAX_TA_STATE) {
            ta_state[li] += 1;
        } else if (patch_bit == 0 && ta_state[li] > 0 && curand_uniform(rng) < S_INV) {
            ta_state[li] -= 1;
        }
    }
}

__device__ inline void type1b_fb(curandState *rng, unsigned int *ta_state) {
    /*
     * Type Ib feedback - Vectorized version:
     * 1. Decrement states for all literals with probability 1/S.
     */

    #pragma unroll 4
    for (int li = 0; li < VECTORIZED_LIMIT; li += 4) {
        uint4 ta_vec = *((uint4 *)&ta_state[li]);

        ta_vec.x -= (ta_vec.x > 0 && curand_uniform(rng) <= S_INV);
        ta_vec.y -= (ta_vec.y > 0 && curand_uniform(rng) <= S_INV);
        ta_vec.z -= (ta_vec.z > 0 && curand_uniform(rng) <= S_INV);
        ta_vec.w -= (ta_vec.w > 0 && curand_uniform(rng) <= S_INV);

        *((uint4 *)&ta_state[li]) = ta_vec;
    }

    #pragma unroll 4
    for (int li = VECTORIZED_LIMIT; li < LITERALS; ++li) {
        if (ta_state[li] > 0 && curand_uniform(rng) < S_INV) {
            ta_state[li] -= 1;
        }
    }
}

__device__ inline void type2_fb(unsigned int *ta_state, float *clause_weight, const unsigned int *patch, int sign) {
    /*
     * Type II feedback - Vectorized version with macro constants:
     * 1. Increment states for literals not present in patch.
     * 2. Decrement clause weight.
     */

    // Update clause weight
    (*clause_weight) -= sign * 1.0f;

#if NEGATIVE_CLAUSES == 0
    if (*clause_weight < 1) *clause_weight = 1;
#endif

    // Use predefined macro instead of runtime computation
    #pragma unroll 4
    for (int li = 0; li < VECTORIZED_LIMIT; li += 4) {
        uint4 ta_vec = *((uint4 *)&ta_state[li]);
        uint4 patch_vec = {
            (patch[li / INT_SIZE] >> (li % INT_SIZE)) & 1u,
            (patch[(li + 1) / INT_SIZE] >> ((li + 1) % INT_SIZE)) & 1u,
            (patch[(li + 2) / INT_SIZE] >> ((li + 2) % INT_SIZE)) & 1u,
            (patch[(li + 3) / INT_SIZE] >> ((li + 3) % INT_SIZE)) & 1u,
        };

        // Increment ta_state elements where patch is 0
        ta_vec.x += (patch_vec.x == 0);
        ta_vec.y += (patch_vec.y == 0);
        ta_vec.z += (patch_vec.z == 0);
        ta_vec.w += (patch_vec.w == 0);

        *((uint4 *)&ta_state[li]) = ta_vec;
    }

    // Handle remaining literals using macro constant
    #pragma unroll 4
    for (int li = VECTORIZED_LIMIT; li < LITERALS; ++li) {
        unsigned int patch_bit = (patch[li / INT_SIZE] >> (li % INT_SIZE)) & 1u;
        if (patch_bit == 0) {
            ta_state[li] += 1;
        }
    }
}

__global__ void clause_update(curandState *rng, unsigned int *global_ta_states, float *clause_weights,
                              const float *class_sums, const int *selected_patch_ids, const int *num_includes,
                              const unsigned int *X, const int *Y, const int e) {
    /*
     * Update the clauses based on the class sum and Y.
     *
     * Params:
     * - rng: Random number generator.
     * - global_ta_states: TA states. Shape: CLAUSES * LITERALS.
     * - clause_weights: Weights of the clauses. Shape: CLAUSES * CLASSES.
     * - class_sums: Shape: CLASSES.
     * - selected_patch_ids: Selected patch ids for each clause. Shape: CLAUSES.
     * - num_includes: Number of included literals for each clause. Shape: CLAUSES.
     * - X: Packed Input data. Shape: PATCHES * NUM_LITERAL_CHUNKS.
     * - Y: Labels for the sample. This is encoded, i.e., y == 0 -> -T and y == 1 -> T. Shape: CLASSES.
     */

    __shared__ float prob[CLASSES];
    __shared__ int tar[CLASSES];
    for (int class_id = threadIdx.x; class_id < CLASSES; class_id += blockDim.x) {
        float clipped = clip_cs(class_sums[class_id]);
        const int y = Y[e * CLASSES + class_id];
        prob[class_id] = abs((float)y - clipped) / (2.0 * THRESH);
        tar[class_id] = 1 - 2 * (clipped > y);
    }
    __syncthreads();

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    curandState localRNG = rng[index];

    for (int clause = index; clause < CLAUSES; clause += stride) {
        unsigned int *ta_state = &global_ta_states[clause * LITERALS];
        int local_clause_output = selected_patch_ids[clause] > -1 ? 1 : 0;
        const unsigned int *patch =
            selected_patch_ids[clause] > -1 ? &X[selected_patch_ids[clause] * NUM_LITERAL_CHUNKS] : nullptr;

        for (unsigned int class_id = 0; class_id < CLASSES; ++class_id) {
            if (tar[class_id] == -1 && curand_uniform(&localRNG) > Q_PROB) {
                continue;  // Skip the class.
            }

            float *local_weight = &clause_weights[clause * CLASSES + class_id];
            int sign = (*local_weight >= 0) - (*local_weight < 0);
            bool should_upate = (curand_uniform(&localRNG) <= prob[class_id]);
            bool type1 = ((tar[class_id] * sign) > 0);
            bool type2 = ((tar[class_id] * sign) < 0 && local_clause_output);

            if (should_upate) {  // CLause update with prob update_p else skip
                if (type1 && local_clause_output) {
                    type1a_fb(&localRNG, ta_state, local_weight, patch, sign);
                } else if (type1 && !local_clause_output) {
                    type1b_fb(&localRNG, ta_state);
                } else if (type2) {
                    type2_fb(ta_state, local_weight, patch, sign);
                }
            }
        }
    }

    rng[index] = localRNG;
}
}
