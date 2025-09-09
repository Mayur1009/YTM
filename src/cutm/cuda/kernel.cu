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
    #define MAX_INCLUDED_LITERALS 272
    #define APPEND_NEGATED 1
    #define INIT_NEG_WEIGHTS 1
    #define NEGATIVE_CLAUSES 1
    #define CLASSES 10
    #define MAX_TA_STATE 255
    #define ENCODE_LOC 1
    #define COALESCED 1
    #define CLAUSE_BANKS 1
__constant__ const double H[CLASSES] = {1};
    #define BIAS 0
#endif

#include <curand_kernel.h>

#define CLAUSES_PER_BANK (CLAUSES / CLAUSE_BANKS)
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

typedef unsigned long long ull;

extern "C" {
    /***********INITIALIZATION***********/
    __global__ void init_weights(curandState *rng, float *clause_weights) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;
        curandState localState = rng[index];

        for (int clause = index; clause < CLAUSES; clause += stride) {
            for (int class_id = 0; class_id < CLASSES; ++class_id) {
#if COALESCED  // Coalesced -- all clauses have weight for all classes.
                clause_weights[clause * CLASSES + class_id] = 1.0f;
    #if INIT_NEG_WEIGHTS
                clause_weights[clause * CLASSES + class_id] = (1.0f - 2.0f * (float)(curand(&localState) % 2));
    #endif
#else
                if (class_id == clause / CLAUSES_PER_BANK) {  // Clause belongs to this class.
                    clause_weights[clause * CLASSES + class_id] = 1.0f;
    #if INIT_NEG_WEIGHTS  // Initialize negative polarity in second half of the clause bank.
                    if ((clause % CLAUSES_PER_BANK) >= (CLAUSES_PER_BANK / 2))
                        clause_weights[clause * CLASSES + class_id] = -1.0f;
    #endif
                } else {
                    clause_weights[clause * CLASSES + class_id] = 0.0f;
                }
#endif
            }
        }

        rng[index] = localState;
    }

    /***********INPUT ENCODING***********/
    __global__ void encode_batch(const unsigned int *X, unsigned int *encoded_X, const int N) {
        // X -> (N * DIM0 * DIM1 * DIM2)
        // encoded_X -> (N * PATCHES * NUM_LITERAL_CHUNKS)
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;

        for (ull e_patch = index; e_patch < (ull)(PATCHES * N); e_patch += stride) {
            ull e = e_patch / PATCHES;
            ull patch_id = e_patch % PATCHES;

            // Calculate the starting point of the patch in the original image
            int patch_coordinate_y = patch_id / (DIM0 - PATCH_DIM0 + 1);
            int patch_coordinate_x = patch_id % (DIM0 - PATCH_DIM0 + 1);

            ull encX_offset = e * (ull)(PATCHES * NUM_LITERAL_CHUNKS) + patch_id * (ull)NUM_LITERAL_CHUNKS;
            unsigned int *patch_output = &encoded_X[encX_offset];

#if APPEND_NEGATED
            for (int literal = LITERALS / 2; literal < LITERALS; ++literal) {
                int chunk_nr = literal / INT_SIZE;
                int chunk_pos = literal % INT_SIZE;
                patch_output[chunk_nr] |= (1u << chunk_pos);
            }
#endif

            for (int lit = 0; lit < patch_coordinate_y; ++lit) {
                int chunk_nr = lit / INT_SIZE;
                int chunk_pos = lit % INT_SIZE;
                patch_output[chunk_nr] |= (1u << chunk_pos);
#if APPEND_NEGATED
                int neg_chunk_nr = (lit + (LITERALS / 2)) / INT_SIZE;
                int neg_chunk_pos = (lit + (LITERALS / 2)) % INT_SIZE;
                patch_output[neg_chunk_nr] &= ~(1u << neg_chunk_pos);
#endif
            }

            for (int lit = 0; lit < patch_coordinate_x; ++lit) {
                int chunk_nr = (DIM1 - PATCH_DIM1 + lit) / INT_SIZE;
                int chunk_pos = (DIM1 - PATCH_DIM1 + lit) % INT_SIZE;
                patch_output[chunk_nr] |= (1u << chunk_pos);
#if APPEND_NEGATED
                int neg_chunk_nr = ((DIM1 - PATCH_DIM1 + lit) + (LITERALS / 2)) / INT_SIZE;
                int neg_chunk_pos = ((DIM1 - PATCH_DIM1 + lit) + (LITERALS / 2)) % INT_SIZE;
                patch_output[neg_chunk_nr] &= ~(1u << neg_chunk_pos);
#endif
            }

            // Iterate over all pixels in the patch
            for (ull p_y = patch_coordinate_y; p_y < patch_coordinate_y + PATCH_DIM1; ++p_y) {
                for (ull p_x = patch_coordinate_x; p_x < patch_coordinate_x + PATCH_DIM0; ++p_x) {
                    for (int z = 0; z < DIM2; ++z) {
                        unsigned long long dense_idx =
                            e * (ull)(DIM0 * DIM1 * DIM2) + p_y * (ull)(DIM0 * DIM2) + p_x * (ull)DIM2 + z;

                        if (X[dense_idx] > 0) {
                            int rel_y = p_y - patch_coordinate_y;
                            int rel_x = p_x - patch_coordinate_x;
#if ENCODE_LOC
                            int patch_pos = (DIM1 - PATCH_DIM1) + (DIM0 - PATCH_DIM0) + rel_y * PATCH_DIM0 * DIM2 +
                                            rel_x * DIM2 + z;
#else
                            int patch_pos = rel_y * PATCH_DIM0 * DIM2 + rel_x * DIM2 + z;
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
        }
    }

    /***********CLAUSE PACKING***********/
    __global__ void pack_clauses(const unsigned int *global_ta_states, unsigned int *packed_clauses,
                                 int *num_includes) {
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

            for (int chunk = 0; chunk < NUM_LITERAL_CHUNKS; ++chunk) {
                unsigned int packed_value = 0;
                int start_lit = chunk * INT_SIZE;
                int end_lit = min(start_lit + INT_SIZE, LITERALS);

                int vectorized_end =
                    start_lit + ((end_lit - start_lit) & ~3);  // Ensure vectorized end is a multiple of 4
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

    /***********CLAUSE EVALUATION***********/
    __device__ inline int clause_match(const unsigned int *ta_state, const unsigned int *X) {
        for (int chunk = 0; chunk < NUM_LITERAL_CHUNKS - 1; ++chunk)
            if ((ta_state[chunk] & X[chunk]) != ta_state[chunk]) return 0;

        if ((ta_state[NUM_LITERAL_CHUNKS - 1] & (X[NUM_LITERAL_CHUNKS - 1] & FILTER)) !=
            (ta_state[NUM_LITERAL_CHUNKS - 1] & FILTER))
            return 0;

        return 1;
    }

    /***********CLAUSE EVALUATION---SLOWER***********/
    __global__ void clause_eval(curandState *rng, const unsigned int *packed_ta_states, const float *clause_weights,
                                int *patch_weights, const unsigned int *X_batch, int *selected_patch_ids,
                                float *class_sums, const int e) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;

        curandState localRNG = rng[index];

        for (int clause = index; clause < CLAUSES; clause += stride) {
            int active_patches[PATCHES];
            int active_count = 0;

            for (int patch_id = 0; patch_id < PATCHES; ++patch_id) {
                int patch_matched = clause_match(
                    &packed_ta_states[clause * NUM_LITERAL_CHUNKS],
                    &X_batch[(ull)e * (ull)(PATCHES * NUM_LITERAL_CHUNKS) + patch_id * NUM_LITERAL_CHUNKS]);
                if (patch_matched) {
                    active_patches[active_count] = patch_id;
                    active_count++;
                }
            }
            if (active_count > 0) {
                int random_index = curand(&localRNG) % active_count;
                selected_patch_ids[clause] = active_patches[random_index];
                patch_weights[clause * PATCHES + active_patches[random_index]] = 1;
                for (int class_id = 0; class_id < CLASSES; ++class_id) {
                    atomicAdd(&class_sums[0 * CLASSES + class_id], clause_weights[clause * CLASSES + class_id]);
                }
            } else {
                selected_patch_ids[clause] = -1;
            }
        }
        rng[index] = localRNG;
    }

    /***********FAST EVALUATION KERNELS***********/
    __global__ void fast_eval(const unsigned int *packed_ta_states, const int *num_includes,
                              const unsigned int *clause_drop_mask, const unsigned int *X_batch,
                              unsigned int *clause_outputs, const int e) {
        // clause_outputs => (N * CLAUSES * PATCHES)
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;
        for (ull clause_patch = index; clause_patch < (ull)CLAUSES * (ull)PATCHES; clause_patch += stride) {
            unsigned int *clause_output = &clause_outputs[clause_patch];

            ull clause = clause_patch / PATCHES;
            ull patch_id = clause_patch % PATCHES;

            // Skip dropped clauses
            if (clause_drop_mask[clause] == 1) {
                *clause_output = 0;
                continue;
            }

            if (num_includes[clause] == 0) {
                *clause_output = 1;
                continue;
            }

            *clause_output =
                clause_match(&packed_ta_states[clause * NUM_LITERAL_CHUNKS],
                             &X_batch[(ull)e * (ull)(PATCHES * NUM_LITERAL_CHUNKS) + patch_id * NUM_LITERAL_CHUNKS]);
        }
    }

    /***********SELECT ACTIVE CLAUSES AND CALCULATE CLASS SUMS***********/
    __global__ void select_active(curandState *rng, const float *clause_weights, const unsigned int *clause_outputs,
                                  int *patch_weights, int *selected_patch_ids, float *class_sums) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;

        curandState localRNG = rng[index];

        for (int clause = index; clause < CLAUSES; clause += stride) {
            int count = 0;
            int selected_id = -1;
            for (int patch_id = 0; patch_id < PATCHES; ++patch_id) {
                if (clause_outputs[clause * PATCHES + patch_id]) {
                    count++;
                    if (curand_uniform(&localRNG) < 1.0f / count) {
                        selected_id = patch_id;
                    }
                }
            }
            selected_patch_ids[clause] = selected_id;
            if (selected_id != -1) {
                patch_weights[clause * PATCHES + selected_id]++;
#if COALESCED == 0
                int class_id = (ull)clause / CLAUSES_PER_BANK;
#else
                for (int class_id = 0; class_id < CLASSES; ++class_id)
#endif
                atomicAdd(&class_sums[class_id],
                          clause_outputs[clause * PATCHES + selected_id] * clause_weights[clause * CLASSES + class_id]);
            }
        }
        rng[index] = localRNG;
    }

    /***********FAST CLASS SUMS CALCULATION FOR INFERENCE***********/
    __global__ void calc_class_sums_infer_batch(const unsigned int *packed_ta_states, const float *clause_weights,
                                                const int *num_includes, const unsigned int *X_batch, const int N,
                                                float *class_sums_batch) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;

        for (ull e_clause = index; e_clause < (ull)N * (ull)CLAUSES; e_clause += stride) {
            ull e = e_clause / CLAUSES;
            ull clause = e_clause % CLAUSES;
            if (num_includes[clause] == 0) continue;  // Skip empty clauses
            int clause_output = 0;
            for (int patch_id = 0; patch_id < PATCHES; ++patch_id) {
                if (clause_match(&packed_ta_states[clause * NUM_LITERAL_CHUNKS],
                                 &X_batch[e * (ull)(PATCHES * NUM_LITERAL_CHUNKS) + patch_id * NUM_LITERAL_CHUNKS])) {
                    clause_output = 1;
                    break;
                }
            }
            if (clause_output) {
                for (int class_id = 0; class_id < CLASSES; ++class_id) {
                    atomicAdd(&class_sums_batch[e * CLASSES + class_id], clause_weights[clause * CLASSES + class_id]);
                }
            }
        }
    }

    /***********TRNAFORM KERNELS***********/
    __global__ void transform(const unsigned int *packed_ta_states, const int *num_includes,
                              const unsigned int *X_batch, const int N, unsigned int *clause_outputs) {
        // clause_outputs => (N * CLAUSES)
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;
        for (ull e_clause = index; e_clause < (ull)N * (ull)CLAUSES; e_clause += stride) {
            ull e = e_clause / CLAUSES;
            ull clause = e_clause % CLAUSES;
            if (num_includes[clause] == 0) {
                clause_outputs[e * CLAUSES + clause] = 1;
                continue;
            }
            int clause_output = 0;
            for (int patch_id = 0; patch_id < PATCHES; ++patch_id) {
                if (clause_match(&packed_ta_states[clause * NUM_LITERAL_CHUNKS],
                                 &X_batch[e * (ull)(PATCHES * NUM_LITERAL_CHUNKS) + patch_id * NUM_LITERAL_CHUNKS])) {
                    clause_output = 1;
                    break;
                }
            }
            clause_outputs[e * CLAUSES + clause] = clause_output;
        }
    }

    __global__ void transform_patchwise(const unsigned int *packed_ta_states, const int *num_includes,
                                        const unsigned int *X_batch, const int N, unsigned int *clause_outputs) {
        // clause_outputs => (N * CLAUSES * PATCHES)
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;
        for (ull e_clause_patch = index; e_clause_patch < (ull)N * (ull)CLAUSES * (ull)PATCHES;
             e_clause_patch += stride) {
            unsigned int *clause_output = &clause_outputs[e_clause_patch];

            ull e_clause = e_clause_patch / PATCHES;
            ull patch_id = e_clause_patch % PATCHES;

            ull e = e_clause / CLAUSES;
            ull clause = e_clause % CLAUSES;

            if (num_includes[clause] == 0) {
                *clause_output = 1;
                continue;
            }

            *clause_output =
                clause_match(&packed_ta_states[clause * NUM_LITERAL_CHUNKS],
                             &X_batch[e * (ull)(PATCHES * NUM_LITERAL_CHUNKS) + patch_id * NUM_LITERAL_CHUNKS]);
        }
    }

    /***********CLAUSE UPDATE KERNELS***********/
    __device__ inline float clip_cs(float cs) { return (cs > THRESH) ? THRESH : ((cs < -THRESH) ? -THRESH : cs); }

    __device__ inline void type1a_fb_scalar(curandState *rng, unsigned int *ta_state, const unsigned int *patch) {
        for (int li = 0; li < LITERALS; ++li) {
            unsigned int patch_bit = (patch[li / INT_SIZE] >> (li % INT_SIZE)) & 1u;
            if (patch_bit == 1 && ta_state[li] < MAX_TA_STATE) {
                ta_state[li] += 1;
            } else if (patch_bit == 0 && ta_state[li] > 0 && curand_uniform(rng) <= S_INV) {
                ta_state[li] -= 1;
            }
        }
    }

    __device__ inline void type1a_fb(curandState *rng, unsigned int *ta_state, const unsigned int *patch) {
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
        for (int li = VECTORIZED_LIMIT; li < LITERALS; ++li) {
            unsigned int patch_bit = (patch[li / INT_SIZE] >> (li % INT_SIZE)) & 1u;
            if (patch_bit == 1 && ta_state[li] < MAX_TA_STATE) {
                ta_state[li] += 1;
            } else if (patch_bit == 0 && ta_state[li] > 0 && curand_uniform(rng) <= S_INV) {
                ta_state[li] -= 1;
            }
        }
    }

    __device__ inline void type1b_fb_scalar(curandState *rng, unsigned int *ta_state) {
        for (int li = 0; li < LITERALS; ++li) {
            if (ta_state[li] > 0 && curand_uniform(rng) <= S_INV) {
                ta_state[li] -= 1;
            }
        }
    }

    __device__ inline void type1b_fb(curandState *rng, unsigned int *ta_state) {
        for (int li = 0; li < VECTORIZED_LIMIT; li += 4) {
            uint4 ta_vec = *((uint4 *)&ta_state[li]);

            ta_vec.x -= (ta_vec.x > 0 && curand_uniform(rng) <= S_INV);
            ta_vec.y -= (ta_vec.y > 0 && curand_uniform(rng) <= S_INV);
            ta_vec.z -= (ta_vec.z > 0 && curand_uniform(rng) <= S_INV);
            ta_vec.w -= (ta_vec.w > 0 && curand_uniform(rng) <= S_INV);

            *((uint4 *)&ta_state[li]) = ta_vec;
        }

        for (int li = VECTORIZED_LIMIT; li < LITERALS; ++li) {
            if (ta_state[li] > 0 && curand_uniform(rng) <= S_INV) {
                ta_state[li] -= 1;
            }
        }
    }

    __device__ inline void type2_fb_scalar(unsigned int *ta_state, const unsigned int *patch) {
        for (int li = 0; li < LITERALS; ++li) {
            unsigned int patch_bit = (patch[li / INT_SIZE] >> (li % INT_SIZE)) & 1u;
            if (patch_bit == 0 && ta_state[li] <= HALF_STATE) {
                ta_state[li] += 1;
            }
        }
    }

    __device__ inline void type2_fb(unsigned int *ta_state, const unsigned int *patch) {
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

        for (int li = VECTORIZED_LIMIT; li < LITERALS; ++li) {
            unsigned int patch_bit = (patch[li / INT_SIZE] >> (li % INT_SIZE)) & 1u;
            if (patch_bit == 0) {
                ta_state[li] += 1;
            }
        }
    }

    __device__ inline double update_probability(double v, double y, double mod, double h) {
        double prob;
        // if (y > 0) {
        //     if (v <= y * (2 * h - 1))
        //         prob = (y - v) / (2 * y);
        //     else {
        //         double a = (1 - h);
        //         double b = (y - v) / (2 * y * a);
        //         prob = a * (1 - pow(1 - b, 1.0 / mod));
        //     }
        // } else {
        //     if (v > y * (2 * h - 1))
        //         prob = (y - v) / (2 * y);
        //     else {
        //         double a = (1 - h);
        //         double b = (y - v) / (2 * y * a);
        //         prob = a * (1 - pow(1 - b, 1.0 / mod));
        //     }
        // }
        prob = (y - v) / (2 * y);
        return prob;
    }

    __global__ void clause_update(curandState *rng, unsigned int *global_ta_states, float *clause_weights,
                                  float *bias_weights, const float *class_sums, const int *selected_patch_ids,
                                  const int *num_includes, const double *true_mod, const double *false_mod,
                                  const unsigned int *clause_drop_mask, const unsigned int *X_batch, const int *Y_batch,
                                  const int e, const int focusced_pos_sampling, const int focused_neg_sampling) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;
        curandState localRNG = rng[index];

        // Should this be separate kernel?
        double update_probs[CLASSES];
        double pos_target_sum = 0, neg_target_sum = 0;
        for (int class_id = 0; class_id < CLASSES; ++class_id) {
            float clipped_cs = clip_cs(class_sums[class_id]);
            int y = Y_batch[e * CLASSES + class_id];
            int local_target = 1 - 2 * (clipped_cs > y);
            update_probs[class_id] =
                update_probability((double)clipped_cs, (double)y,
                                   local_target == 1 ? true_mod[class_id] : false_mod[class_id], H[class_id]);
            local_target == 1 ? (pos_target_sum += update_probs[class_id]) : (neg_target_sum += update_probs[class_id]);
        }

        for (int clause = index; clause < CLAUSES; clause += stride) {
            // Skip dropped clauses
            if (clause_drop_mask[clause] == 1) continue;

            unsigned int *ta_state = &global_ta_states[clause * LITERALS];
            int local_clause_output = selected_patch_ids[clause] > -1 ? 1 : 0;
            const unsigned int *X = &X_batch[(ull)e * (ull)(PATCHES * NUM_LITERAL_CHUNKS)];
            const unsigned int *patch =
                selected_patch_ids[clause] > -1 ? &X[selected_patch_ids[clause] * NUM_LITERAL_CHUNKS] : nullptr;

#if COALESCED == 0
            ull class_id = (ull)clause / CLAUSES_PER_BANK;
            {
#else
            for (ull class_id = 0; class_id < CLASSES; ++class_id) {
#endif
                float clipped_cs = clip_cs(class_sums[class_id]);
                int y = Y_batch[e * CLASSES + class_id];
                int local_target = 1 - 2 * (clipped_cs > y);

                if (local_target == -1 && curand_uniform(&localRNG) > Q_PROB) continue;

                double update_prob = update_probs[class_id];
                // if (focusced_pos_sampling && local_target == 1) update_prob = update_prob / pos_target_sum;
                // if (focused_neg_sampling && local_target == -1) update_prob = update_prob / neg_target_sum;

                float *local_weight = &clause_weights[clause * CLASSES + class_id];
                int sign = (*local_weight >= 0) - (*local_weight < 0);

                bool should_update = (curand_uniform(&localRNG) <= update_prob);
                bool type1a =
                    ((local_target * sign) > 0 && local_clause_output && num_includes[clause] <= MAX_INCLUDED_LITERALS);
                bool type1b = ((local_target * sign) > 0 &&
                               !(local_clause_output && num_includes[clause] <= MAX_INCLUDED_LITERALS));
                bool type2 = ((local_target * sign) < 0 && local_clause_output);

                if (should_update) {  // CLause update with prob update_p else skip
                    if (type1a) {
                        (*local_weight) += sign * 1.0f;
#if BIAS
                        bias_weights[class_id] += sign * 1.0f;
#endif
                        type1a_fb(&localRNG, ta_state, patch);
                    } else if (type1b) {
                        type1b_fb(&localRNG, ta_state);
                    } else if (type2) {
                        (*local_weight) -= sign * 1.0f;
#if BIAS
                        bias_weights[class_id] -= sign * 1.0f;
#endif
#if NEGATIVE_CLAUSES == 0
                        if (*local_weight < 1) *local_weight = 1;
                        if (bias_weights[class_id] < 0) bias_weights[class_id] = 0;
#endif
                        type2_fb(ta_state, patch);
                    }
                }
            }
        }
        rng[index] = localRNG;
    }
}
