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

#define VECTORIZED_LIMIT (LITERALS - (LITERALS % 4))
#define S_INV (1.0f / S)
#define Q_PROB (1.0f * Q / max(1, CLASSES - 1))

extern "C" {
__global__ void initialize(curandState *rng, unsigned int *global_ta_states, float *clause_weights) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    curandState localState = rng[index];

    for (int clause = index; clause < CLAUSES; clause += stride) {
        for (int li = 0; li < LITERALS; ++li) {
            global_ta_states[clause * LITERALS + li] = MAX_TA_STATE / 2;  // Initialize TA states to 0
        }
        for (int class_id = 0; class_id < CLASSES; ++class_id) {
#if NEGATIVE_CLAUSES
            clause_weights[clause * CLASSES + class_id] = (1.0f - 2.0f * (float)(curand(&localState) % 2));
#else
            clause_weights[clause * CLASSES + class_id] = 1;
#endif
        }
    }
    rng[index] = localState;
}

__global__ void encode(unsigned int *X_indptr, unsigned int *X_indices, unsigned int *encoded_X, int e) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    unsigned int *indices = &X_indices[X_indptr[e]];
    int number_of_indices = X_indptr[e + 1] - X_indptr[e];

    for (int k = 0; k < number_of_indices; ++k) {
        int y = indices[k] / (DIM0 * DIM2);
        int x = (indices[k] % (DIM0 * DIM2)) / DIM2;
        int z = (indices[k] % (DIM0 * DIM2)) % DIM2;

        for (int patch = index; patch < PATCHES; patch += stride) {
            int patch_coordinate_y = patch / (DIM0 - PATCH_DIM0 + 1);
            int patch_coordinate_x = patch % (DIM0 - PATCH_DIM0 + 1);

            if ((y < patch_coordinate_y) || (y >= patch_coordinate_y + PATCH_DIM1) || (x < patch_coordinate_x) ||
                (x >= patch_coordinate_x + PATCH_DIM0)) {
                continue;
            }

            int p_y = y - patch_coordinate_y;
            int p_x = x - patch_coordinate_x;

#if ENCODE_LOC
            int patch_pos = (DIM1 - PATCH_DIM1) + (DIM0 - PATCH_DIM0) + p_y * PATCH_DIM0 * DIM2 + p_x * DIM2 + z;
#else
            int patch_pos = p_y * PATCH_DIM0 * DIM2 + p_x * DIM2 + z;
#endif
            encoded_X[patch * LITERALS + patch_pos] = 1;
#if APPEND_NEGATED
            encoded_X[patch * LITERALS + patch_pos + LITERALS / 2] = 0;
#endif
        }
    }
}

__device__ inline int clause_match(const unsigned int *ta_state, const unsigned int *X) {
    const uint4 half_state = {MAX_TA_STATE / 2, MAX_TA_STATE / 2, MAX_TA_STATE / 2, MAX_TA_STATE / 2};
#pragma unroll 4
    for (int li = 0; li < VECTORIZED_LIMIT; li += 4) {
        uint4 ta_vec = *((uint4 *)&ta_state[li]);
        uint4 x_vec = *((uint4 *)&X[li]);
        unsigned int lit_x = (ta_vec.x > half_state.x);
        unsigned int lit_y = (ta_vec.y > half_state.y);
        unsigned int lit_z = (ta_vec.z > half_state.z);
        unsigned int lit_w = (ta_vec.w > half_state.w);
        if ((lit_x & (x_vec.x == 0)) || (lit_y & (x_vec.y == 0)) || (lit_z & (x_vec.z == 0)) ||
            (lit_w & (x_vec.w == 0))) {
            return 0;
        }
    }
    for (int li = VECTORIZED_LIMIT; li < LITERALS; ++li) {
        unsigned int lit = (ta_state[li] > MAX_TA_STATE / 2) ? 1u : 0u;
        if (lit * X[li] != lit) {
            return 0;  // Not matching
        }
    }
    return 1;  // Matching
}

__global__ void clause_eval(curandState *rng, const unsigned int *global_ta_states, const float *clause_weights,
                            const unsigned int *X, int *selected_patch_ids) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    curandState localRNG = rng[index];

    // curandState localRNG = rng[ty * sx + tx];
    for (int clause_patch = index; clause_patch < CLAUSES * PATCHES; clause_patch += stride) {
        int clause = clause_patch / PATCHES;
        int patch_id = clause_patch % PATCHES;

        // for (int clause = tx; clause < CLAUSES; clause += sx) {
        //     for (int patch_id = ty; patch_id < PATCHES; patch_id += sy) {
        if (selected_patch_ids[clause] > -1) continue;  // Skip already selected clauses
        const unsigned int *ta_state = &global_ta_states[clause * LITERALS];
        const unsigned int *patch = &X[patch_id * LITERALS];
        int patch_matched = clause_match(ta_state, patch);
        // printf("clause: %d, patch_id: %d, patch_matched: %d\n", clause, patch_id, patch_matched);
        if (patch_matched && curand_uniform(&localRNG) <= 0.5f) {
            // int old_val = atomicCAS(&selected_patch_ids[clause], -1, patch_id);
            selected_patch_ids[clause] = patch_id;
            break;
        }
    }
    rng[index] = localRNG;
}

__global__ void calc_class_sums(int *selected_patch_ids, const float *clause_weights, float *class_sums) {
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

__global__ void calc_num_includes(unsigned int *global_ta_states, int *num_includes) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int clause = index; clause < CLAUSES; clause += stride) {
        int count = 0;
        for (int li = 0; li < LITERALS; ++li) {
            if (global_ta_states[clause * LITERALS + li] > MAX_TA_STATE / 2) {
                count++;
            }
        }
        num_includes[clause] = count;
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

    for (int li = 0; li < VECTORIZED_LIMIT; li += 4) {
        uint4 ta_vec = *((uint4 *)&ta_state[li]);
        uint4 patch_vec = *((uint4 *)&patch[li]);

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
    for (int li = VECTORIZED_LIMIT; li < LITERALS; ++li) {
        if (patch[li] == 1 && ta_state[li] < MAX_TA_STATE) {
            ta_state[li] += 1;
        } else if (patch[li] == 0 && ta_state[li] > 0 && curand_uniform(rng) < S_INV) {
            ta_state[li] -= 1;
        }
    }
}

__device__ inline void type1b_fb(curandState *rng, unsigned int *ta_state) {
    /*
     * Type Ib feedback - Vectorized version:
     * 1. Decrement states for all literals with probability 1/S.
     */

    for (int li = 0; li < VECTORIZED_LIMIT; li += 4) {
        uint4 ta_vec = *((uint4 *)&ta_state[li]);

        ta_vec.x -= (ta_vec.x > 0 && curand_uniform(rng) <= S_INV);
        ta_vec.y -= (ta_vec.y > 0 && curand_uniform(rng) <= S_INV);
        ta_vec.z -= (ta_vec.z > 0 && curand_uniform(rng) <= S_INV);
        ta_vec.w -= (ta_vec.w > 0 && curand_uniform(rng) <= S_INV);

        *((uint4 *)&ta_state[li]) = ta_vec;
    }

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
    for (int li = 0; li < VECTORIZED_LIMIT; li += 4) {
        uint4 ta_vec = *((uint4 *)&ta_state[li]);
        uint4 patch_vec = *((uint4 *)&patch[li]);

        // Increment ta_state elements where patch is 0
        ta_vec.x += (patch_vec.x == 0);
        ta_vec.y += (patch_vec.y == 0);
        ta_vec.z += (patch_vec.z == 0);
        ta_vec.w += (patch_vec.w == 0);

        *((uint4 *)&ta_state[li]) = ta_vec;
    }

    // Handle remaining literals using macro constant
    for (int li = VECTORIZED_LIMIT; li < LITERALS; ++li) {
        if (patch[li] == 0) {
            ta_state[li] += 1;
        }
    }
}

__global__ void clause_update(curandState *rng, unsigned int *global_ta_states, float *clause_weights,
                              const float *class_sums, const int *selected_patch_ids, const int *num_includes,
                              const unsigned int *X, const int *Y) {
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
     * - X: Input data. Shape: PATCHES * LITERALS.
     * - Y: Labels for the sample. This is encoded, i.e., y == 0 -> -T and y == 1 -> T. Shape: CLASSES.
     */

    __shared__ float prob[CLASSES];
    __shared__ int tar[CLASSES];
    for (int class_id = threadIdx.x; class_id < CLASSES; class_id += blockDim.x) {
        float clipped = clip_cs(class_sums[class_id]);
        prob[class_id] = abs((float)Y[class_id] - clipped) / (2.0 * THRESH);
        tar[class_id] = 1 - 2 * (clipped > Y[class_id]);
        // printf("class_id: %d, clipped: %f, prob: %f, tar: %d, Y: %d\n", class_id, clipped, prob[class_id],
        //        tar[class_id], Y[class_id]);
    }
    __syncthreads();

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    curandState localRNG = rng[index];

    for (int clause = index; clause < CLAUSES; clause += stride) {
        unsigned int *ta_state = &global_ta_states[clause * LITERALS];
        int local_clause_output = selected_patch_ids[clause] > -1 ? 1 : 0;
        const unsigned int *patch =
            selected_patch_ids[clause] > -1 ? &X[selected_patch_ids[clause] * LITERALS] : nullptr;

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
