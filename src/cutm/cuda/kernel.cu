// Ignore this block, it is used to only for neovim clangd lsp.
#ifdef IS_NEOVIM_CLANGD_ENV
    #define CLAUSES 100ULL
    #define THRESH 500
    #define S 10.0
    #define Q 1
    #define DIM0 28ULL
    #define DIM1 28ULL
    #define DIM2 1ULL
    #define PATCH_DIM0 10
    #define PATCH_DIM1 10
    #define PATCHES 361ULL
    #define LITERALS 272ULL
    #define MAX_INCLUDED_LITERALS 272ULL
    #define APPEND_NEGATED 1
    #define NEGATIVE_CLAUSES 1
    #define CLASSES 10
    #define MAX_TA_STATE 255
    #define ENCODE_LOC 1
    #define COALESCED 1
    #define CLAUSE_BANKS 1
    #define WEIGHTED 1
    #define MAX_WEIGHT 3.4e38f
    #define S_NEG_POLARITY S
    #define ALLOW_POLARITY_CHANGE 1
    #define INCLUDE_TA_STATE 128
    #define TYPE1A_FB 1
    #define TYPE1B_FB 1
    #define TYPE2_FB 1
    #define SPLIT_CLASS_SUM 0
__device__ double H[CLASSES] = {0.5};
#endif

#include <curand_kernel.h>

#define CLAUSES_PER_BANK (CLAUSES / CLAUSE_BANKS)
#if ((LITERALS / 2) & 1)        // Ensure that LITERALS/2 is even, because the vectorized code does not work
                                // otherwise.......dont know why....some memory aligment issue
    #define VECTORIZED_LIMIT 0  // odd
#else
    #define VECTORIZED_LIMIT (LITERALS & ~3)  // even
#endif
#define S_INV (1.0f / S)
#define S_NEG_POLARITY_INV (1.0f / S_NEG_POLARITY)
#define Q_PROB (1.0f * Q / max(1, CLASSES - 1))
#define INT_SIZE 32
#define NUM_LITERAL_CHUNKS (((LITERALS - 1) / INT_SIZE) + 1)
#if ((LITERALS % INT_SIZE) != 0)
    #define FILTER (~(0xFFFFFFFF << (LITERALS % INT_SIZE)))
#else
    #define FILTER 0xFFFFFFFF
#endif

#if COALESCED == 0
    #define LOOP_CLASS_ID(class_id, clause) class_id = (ull)clause / CLAUSES_PER_BANK;
#else
    #define LOOP_CLASS_ID(class_id, clause) for (class_id = 0; class_id < CLASSES; ++class_id)
#endif

#define CLIP(val, min, max) ((val < min) ? min : ((val > max) ? max : val))

typedef unsigned long long ull;

extern "C" {
    /***********INPUT ENCODING***********/
    __global__ void encode_batch(const int* X, unsigned int* encoded_X, const int N) {
        // X -> (N * DIM0 * DIM1 * DIM2) array with possible values {-1, 0, 1}.
        // 1 -> feat present, -1 -> feat absent, 0 -> dont care
        // encoded_X -> (N * PATCHES * NUM_LITERAL_CHUNKS)
        ull index = blockIdx.x * blockDim.x + threadIdx.x;
        ull stride = blockDim.x * gridDim.x;

        for (ull e_patch = index; e_patch < (ull)(PATCHES * N); e_patch += stride) {
            ull e = e_patch / PATCHES;
            ull patch_id = e_patch % PATCHES;

            // Calculate the starting point of the patch in the original image
            int patch_coordinate_y = patch_id / (DIM0 - PATCH_DIM0 + 1);
            int patch_coordinate_x = patch_id % (DIM0 - PATCH_DIM0 + 1);

            ull encX_offset = e * (ull)(PATCHES * NUM_LITERAL_CHUNKS) + patch_id * (ull)NUM_LITERAL_CHUNKS;
            unsigned int* patch_output = &encoded_X[encX_offset];

            // Initialization.
            // By default, all values in encoded_X are set to 0 (in python code).
            // So, only need to initialize all negated literals to 1.
#if APPEND_NEGATED
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

            // Iterate over features in a patch, that are either 1 (present) or 0 (dont care). -1(absent) is already
            // taken care of in the initialization.
            for (ull p_y = patch_coordinate_y; p_y < patch_coordinate_y + PATCH_DIM1; ++p_y) {
                for (ull p_x = patch_coordinate_x; p_x < patch_coordinate_x + PATCH_DIM0; ++p_x) {
                    for (int z = 0; z < DIM2; ++z) {
                        unsigned long long dense_idx =
                            e * (ull)(DIM0 * DIM1 * DIM2) + p_y * (ull)(DIM0 * DIM2) + p_x * (ull)DIM2 + z;

                        int rel_y = p_y - patch_coordinate_y;
                        int rel_x = p_x - patch_coordinate_x;
#if ENCODE_LOC
                        int patch_pos =
                            (DIM1 - PATCH_DIM1) + (DIM0 - PATCH_DIM0) + rel_y * PATCH_DIM0 * DIM2 + rel_x * DIM2 + z;
#else
                        int patch_pos = rel_y * PATCH_DIM0 * DIM2 + rel_x * DIM2 + z;
#endif
                        if (X[dense_idx] == 1) {
                            int chunk_nr = patch_pos / INT_SIZE;
                            int chunk_pos = patch_pos % INT_SIZE;
                            patch_output[chunk_nr] |= (1u << chunk_pos);
#if APPEND_NEGATED
                            int neg_chunk_nr = (patch_pos + (LITERALS / 2)) / INT_SIZE;
                            int neg_chunk_pos = (patch_pos + (LITERALS / 2)) % INT_SIZE;
                            patch_output[neg_chunk_nr] &= ~(1u << neg_chunk_pos);
#endif
                        } else if (X[dense_idx] == 0) {
                            // Dont care value. 0 in both positive and negative literals.
                            // positive literal is already 0, only need to set negative literal to 0
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
    __global__ void pack_clauses(const unsigned int* global_ta_states, unsigned int* packed_clauses,
                                 int* num_includes) {
        /*
         * Pack the TA states into chunks of 32 bits. Each chunk represents a set of literals.
         * The number of included literals is also calculated here.
         */
        ull index = blockIdx.x * blockDim.x + threadIdx.x;
        ull stride = blockDim.x * gridDim.x;
        for (ull clause = index; clause < CLAUSES; clause += stride) {
            const unsigned int* ta_state = &global_ta_states[clause * LITERALS];
            unsigned int* packed_clause = &packed_clauses[clause * NUM_LITERAL_CHUNKS];
            int total_count = 0;
            memset(packed_clause, 0, NUM_LITERAL_CHUNKS * sizeof(unsigned int));
            for (int li = 0; li < VECTORIZED_LIMIT; li += 4) {
                uint4 ta_vec = *((uint4*)&ta_state[li]);

                if (ta_vec.x >= INCLUDE_TA_STATE) {
                    packed_clause[li / INT_SIZE] |= (1u << (li % INT_SIZE));
                    total_count++;
                }
                if (ta_vec.y >= INCLUDE_TA_STATE) {
                    packed_clause[(li + 1) / INT_SIZE] |= (1u << ((li + 1) % INT_SIZE));
                    total_count++;
                }
                if (ta_vec.z >= INCLUDE_TA_STATE) {
                    packed_clause[(li + 2) / INT_SIZE] |= (1u << ((li + 2) % INT_SIZE));
                    total_count++;
                }
                if (ta_vec.w >= INCLUDE_TA_STATE) {
                    packed_clause[(li + 3) / INT_SIZE] |= (1u << ((li + 3) % INT_SIZE));
                    total_count++;
                }
            }
            for (int li = VECTORIZED_LIMIT; li < LITERALS; ++li) {
                if (ta_state[li] >= INCLUDE_TA_STATE) {
                    packed_clause[li / INT_SIZE] |= (1u << (li % INT_SIZE));
                    total_count++;
                }
            }
            num_includes[clause] = total_count;
        }
    }

    /***********CLAUSE EVALUATION***********/
    __device__ inline int clause_match(const unsigned int* ta_state, const unsigned int* X,
                                       const unsigned int* literal_mask) {
        for (int chunk = 0; chunk < NUM_LITERAL_CHUNKS - 1; ++chunk)
            if ((ta_state[chunk] & (X[chunk] & literal_mask[chunk])) != (ta_state[chunk] & literal_mask[chunk]))
                return 0;
        if ((ta_state[NUM_LITERAL_CHUNKS - 1] &
             (X[NUM_LITERAL_CHUNKS - 1] & literal_mask[NUM_LITERAL_CHUNKS - 1] & FILTER)) !=
            (ta_state[NUM_LITERAL_CHUNKS - 1] & literal_mask[NUM_LITERAL_CHUNKS - 1] & FILTER))
            return 0;

        return 1;
    }

    /***********FAST EVALUATION KERNELS***********/
    __global__ void fast_eval(const unsigned int* packed_ta_states, const int* num_includes,
                              const unsigned int* clause_drop_mask, const unsigned int* literal_mask,
                              const unsigned int* X_batch, unsigned int* clause_outputs, const int e) {
        // clause_outputs => (N * CLAUSES * PATCHES)
        ull index = blockIdx.x * blockDim.x + threadIdx.x;
        ull stride = blockDim.x * gridDim.x;
        for (ull clause_patch = index; clause_patch < (ull)CLAUSES * (ull)PATCHES; clause_patch += stride) {
            unsigned int* clause_output = &clause_outputs[clause_patch];

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

            *clause_output = clause_match(
                &packed_ta_states[clause * NUM_LITERAL_CHUNKS],
                &X_batch[(ull)e * (ull)(PATCHES * NUM_LITERAL_CHUNKS) + patch_id * NUM_LITERAL_CHUNKS], literal_mask);
        }
    }

    /***********SELECT ACTIVE CLAUSES AND CALCULATE CLASS SUMS***********/
    __global__ void select_active(curandState* rng, const float* clause_weights, const unsigned int* clause_outputs,
                                  int* patch_weights, int* selected_patch_ids, float* positive_evidence,
                                  float* negative_evidence) {
        ull index = blockIdx.x * blockDim.x + threadIdx.x;
        ull stride = blockDim.x * gridDim.x;

        curandState localRNG = rng[index];

        for (ull clause = index; clause < CLAUSES; clause += stride) {
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
                ull class_id;
                LOOP_CLASS_ID(class_id, clause) {
                    if (clause_weights[clause * CLASSES + class_id] >= 0)
                        // Positive polarity clauses
                        atomicAdd(&positive_evidence[class_id], clause_outputs[clause * PATCHES + selected_id] *
                                                                    clause_weights[clause * CLASSES + class_id]);
                    else
                        // Negative polarity clauses
                        atomicAdd(&negative_evidence[class_id], clause_outputs[clause * PATCHES + selected_id] *
                                                                    clause_weights[clause * CLASSES + class_id]);
                }
            }
        }
        rng[index] = localRNG;
    }

    /***********FAST CLASS SUMS CALCULATION FOR INFERENCE***********/
    __global__ void calc_class_sums_infer_batch(const unsigned int* packed_ta_states, const float* clause_weights,
                                                const int* num_includes, const unsigned int* X_batch, const int N,
                                                float* class_sums_batch, const unsigned int* literal_mask) {
        ull index = blockIdx.x * blockDim.x + threadIdx.x;
        ull stride = blockDim.x * gridDim.x;

        for (ull e_clause = index; e_clause < (ull)N * (ull)CLAUSES; e_clause += stride) {
            ull e = e_clause / CLAUSES;
            ull clause = e_clause % CLAUSES;
            if (num_includes[clause] == 0) continue;  // Skip empty clauses
            int clause_output = 0;
            for (int patch_id = 0; patch_id < PATCHES; ++patch_id) {
                if (clause_match(&packed_ta_states[clause * NUM_LITERAL_CHUNKS],
                                 &X_batch[e * (ull)(PATCHES * NUM_LITERAL_CHUNKS) + patch_id * NUM_LITERAL_CHUNKS],
                                 literal_mask)) {
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
    __global__ void transform(const unsigned int* packed_ta_states, const int* num_includes,
                              const unsigned int* X_batch, const int N, unsigned int* clause_outputs,
                              const unsigned int* literal_mask) {
        // clause_outputs => (N * CLAUSES)
        ull index = blockIdx.x * blockDim.x + threadIdx.x;
        ull stride = blockDim.x * gridDim.x;
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
                                 &X_batch[e * (ull)(PATCHES * NUM_LITERAL_CHUNKS) + patch_id * NUM_LITERAL_CHUNKS],
                                 literal_mask)) {
                    clause_output = 1;
                    break;
                }
            }
            clause_outputs[e * CLAUSES + clause] = clause_output;
        }
    }

    __global__ void transform_patchwise(const unsigned int* packed_ta_states, const int* num_includes,
                                        const unsigned int* X_batch, const int N, unsigned int* clause_outputs,
                                        const unsigned int* literal_mask) {
        // clause_outputs => (N * CLAUSES * PATCHES)
        ull index = blockIdx.x * blockDim.x + threadIdx.x;
        ull stride = blockDim.x * gridDim.x;
        for (ull e_clause_patch = index; e_clause_patch < (ull)N * (ull)CLAUSES * (ull)PATCHES;
             e_clause_patch += stride) {
            unsigned int* clause_output = &clause_outputs[e_clause_patch];

            ull e_clause = e_clause_patch / PATCHES;
            ull patch_id = e_clause_patch % PATCHES;

            ull e = e_clause / CLAUSES;
            ull clause = e_clause % CLAUSES;

            if (num_includes[clause] == 0) {
                *clause_output = 1;
                continue;
            }

            *clause_output = clause_match(
                &packed_ta_states[clause * NUM_LITERAL_CHUNKS],
                &X_batch[e * (ull)(PATCHES * NUM_LITERAL_CHUNKS) + patch_id * NUM_LITERAL_CHUNKS], literal_mask);
        }
    }

    /***********CLAUSE UPDATE KERNELS***********/
    __device__ inline void type1a_fb(curandState* rng, unsigned int* ta_state, const unsigned int* patch,
                                     const int sign, const unsigned int* literal_mask) {
        float s_inv = (sign == 1) ? S_INV : S_NEG_POLARITY_INV;
        for (int li = 0; li < VECTORIZED_LIMIT; li += 4) {
            uint4 ta_vec = *((uint4*)&ta_state[li]);
            uint4 patch_vec = {
                (patch[li / INT_SIZE] >> (li % INT_SIZE)) & 1u,
                (patch[(li + 1) / INT_SIZE] >> ((li + 1) % INT_SIZE)) & 1u,
                (patch[(li + 2) / INT_SIZE] >> ((li + 2) % INT_SIZE)) & 1u,
                (patch[(li + 3) / INT_SIZE] >> ((li + 3) % INT_SIZE)) & 1u,
            };

            uint4 lit_up = {
                (literal_mask[li / INT_SIZE] >> (li % INT_SIZE)) & 1u,
                (literal_mask[(li + 1) / INT_SIZE] >> ((li + 1) % INT_SIZE)) & 1u,
                (literal_mask[(li + 2) / INT_SIZE] >> ((li + 2) % INT_SIZE)) & 1u,
                (literal_mask[(li + 3) / INT_SIZE] >> ((li + 3) % INT_SIZE)) & 1u,
            };

            ta_vec.x += (lit_up.x == 1 && patch_vec.x == 1 && ta_vec.x < MAX_TA_STATE);
            ta_vec.y += (lit_up.y == 1 && patch_vec.y == 1 && ta_vec.y < MAX_TA_STATE);
            ta_vec.z += (lit_up.z == 1 && patch_vec.z == 1 && ta_vec.z < MAX_TA_STATE);
            ta_vec.w += (lit_up.w == 1 && patch_vec.w == 1 && ta_vec.w < MAX_TA_STATE);

            ta_vec.x -= (lit_up.x == 1 && patch_vec.x == 0 && ta_vec.x > 0 && curand_uniform(rng) <= s_inv);
            ta_vec.y -= (lit_up.y == 1 && patch_vec.y == 0 && ta_vec.y > 0 && curand_uniform(rng) <= s_inv);
            ta_vec.z -= (lit_up.z == 1 && patch_vec.z == 0 && ta_vec.z > 0 && curand_uniform(rng) <= s_inv);
            ta_vec.w -= (lit_up.w == 1 && patch_vec.w == 0 && ta_vec.w > 0 && curand_uniform(rng) <= s_inv);

            // Write back the vectorized results
            *((uint4*)&ta_state[li]) = ta_vec;
        }

        // Handle remaining literals (when LITERALS % 4 != 0)
        for (int li = VECTORIZED_LIMIT; li < LITERALS; ++li) {
            unsigned int patch_bit = (patch[li / INT_SIZE] >> (li % INT_SIZE)) & 1u;
            unsigned int lit_up = (literal_mask[li / INT_SIZE] >> (li % INT_SIZE)) & 1u;
            if (lit_up == 1 && patch_bit == 1 && ta_state[li] < MAX_TA_STATE) {
                ta_state[li] += 1;
            } else if (lit_up == 1 && patch_bit == 0 && ta_state[li] > 0 && curand_uniform(rng) <= s_inv) {
                ta_state[li] -= 1;
            }
        }
    }

    __device__ inline void type1b_fb(curandState* rng, unsigned int* ta_state, const int sign,
                                     const unsigned int* literal_mask) {
        float s_inv = (sign == 1) ? S_INV : S_NEG_POLARITY_INV;
        for (int li = 0; li < VECTORIZED_LIMIT; li += 4) {
            uint4 ta_vec = *((uint4*)&ta_state[li]);
            uint4 lit_up = {
                (literal_mask[li / INT_SIZE] >> (li % INT_SIZE)) & 1u,
                (literal_mask[(li + 1) / INT_SIZE] >> ((li + 1) % INT_SIZE)) & 1u,
                (literal_mask[(li + 2) / INT_SIZE] >> ((li + 2) % INT_SIZE)) & 1u,
                (literal_mask[(li + 3) / INT_SIZE] >> ((li + 3) % INT_SIZE)) & 1u,
            };

            ta_vec.x -= (lit_up.x == 1 && ta_vec.x > 0 && curand_uniform(rng) <= s_inv);
            ta_vec.y -= (lit_up.y == 1 && ta_vec.y > 0 && curand_uniform(rng) <= s_inv);
            ta_vec.z -= (lit_up.z == 1 && ta_vec.z > 0 && curand_uniform(rng) <= s_inv);
            ta_vec.w -= (lit_up.w == 1 && ta_vec.w > 0 && curand_uniform(rng) <= s_inv);

            *((uint4*)&ta_state[li]) = ta_vec;
        }

        for (int li = VECTORIZED_LIMIT; li < LITERALS; ++li) {
            unsigned int lit_up = (literal_mask[li / INT_SIZE] >> (li % INT_SIZE)) & 1u;
            if (lit_up == 1 && ta_state[li] > 0 && curand_uniform(rng) <= s_inv) {
                ta_state[li] -= 1;
            }
        }
    }

    __device__ inline void type2_fb(unsigned int* ta_state, const unsigned int* patch,
                                    const unsigned int* literal_mask) {
        for (int li = 0; li < VECTORIZED_LIMIT; li += 4) {
            uint4 ta_vec = *((uint4*)&ta_state[li]);
            uint4 patch_vec = {
                (patch[li / INT_SIZE] >> (li % INT_SIZE)) & 1u,
                (patch[(li + 1) / INT_SIZE] >> ((li + 1) % INT_SIZE)) & 1u,
                (patch[(li + 2) / INT_SIZE] >> ((li + 2) % INT_SIZE)) & 1u,
                (patch[(li + 3) / INT_SIZE] >> ((li + 3) % INT_SIZE)) & 1u,
            };

            uint4 lit_up = {
                (literal_mask[li / INT_SIZE] >> (li % INT_SIZE)) & 1u,
                (literal_mask[(li + 1) / INT_SIZE] >> ((li + 1) % INT_SIZE)) & 1u,
                (literal_mask[(li + 2) / INT_SIZE] >> ((li + 2) % INT_SIZE)) & 1u,
                (literal_mask[(li + 3) / INT_SIZE] >> ((li + 3) % INT_SIZE)) & 1u,
            };

            // Increment ta_state elements where patch is 0 and ta_state < INCLUDE_TA_STATE
            ta_vec.x += (lit_up.x == 1 && patch_vec.x == 0 && ta_vec.x < INCLUDE_TA_STATE);
            ta_vec.y += (lit_up.y == 1 && patch_vec.y == 0 && ta_vec.y < INCLUDE_TA_STATE);
            ta_vec.z += (lit_up.z == 1 && patch_vec.z == 0 && ta_vec.z < INCLUDE_TA_STATE);
            ta_vec.w += (lit_up.w == 1 && patch_vec.w == 0 && ta_vec.w < INCLUDE_TA_STATE);

            *((uint4*)&ta_state[li]) = ta_vec;
        }

        for (int li = VECTORIZED_LIMIT; li < LITERALS; ++li) {
            unsigned int patch_bit = (patch[li / INT_SIZE] >> (li % INT_SIZE)) & 1u;
            unsigned int lit_up = (literal_mask[li / INT_SIZE] >> (li % INT_SIZE)) & 1u;
            if (lit_up == 1 && patch_bit == 0 && ta_state[li] < INCLUDE_TA_STATE) {
                ta_state[li] += 1;
            }
        }
    }

    __device__ inline double uprob_fun(double v, double y, double h, double g) {
        int tcs = (2 * y * h) - y;
        double prob = (y - v) / (2 * y);

        // Formula:
        // For y > 0:
        //    if v <= tcs:
        //      prob = 1.0 - (h^(1-g) * (1-prob)^g)
        //    else:
        //      prob = (1-h)^(1-g) * prob^g
        //  For y < 0:
        //    if v <= tcs:
        //      prob = (1-h)^(1-g) * prob^g
        //    else:
        //      prob = 1.0 - (h^(1-g) * (1-prob)^g)

        if ((y > 0) == (v > tcs)) {
            prob = pow((1.0 - h), (1.0 - g)) * pow(prob, g);
        } else {
            prob = 1.0 - (pow(h, 1.0 - g) * pow((1.0 - prob), g));
        }

        return prob;
    }

    __global__ void evidence_to_update_prob(const float* positive_evidence, const float* negative_evidence,
                                            const int* targets, const double* g_pos, const double* g_neg, const int e,
                                            double* pprob, double* nprob) {
        ull index = blockIdx.x * blockDim.x + threadIdx.x;
        ull stride = blockDim.x * gridDim.x;
        for (ull class_id = index; class_id < CLASSES; class_id += stride) {
            int local_target = targets[e * CLASSES + class_id];
            if (local_target == 0) {
                pprob[class_id] = 0.0;
                nprob[class_id] = 0.0;
                continue;
            }

#if SPLIT_CLASS_SUM == 1
            double pos_ev = (double)CLIP(positive_evidence[class_id], 0, THRESH);
            double neg_ev = (double)CLIP(negative_evidence[class_id], -THRESH, 0);
            // Special case when using split class sums. Not integrated with g and h yet.
            if (local_target == 1) {
                // We want to learn the positive evidence for this class, and supress the negative evidence.
                // Which, mean that the positive evidence should reach THRESH, and negative evidence should reach 0.
                // So, pprob will be (THRESH - pos_ev) / THRESH, This will produce values from 1 to 0 as pos_ev goes
                // from 0 to THRESH. And, nprob will be (0 - neg_ev) / THRESH, This will produce values from 1 to 0 as
                // neg_ev goes from -THRESH to 0.
                pprob[class_id] = (THRESH - pos_ev) / THRESH;
                nprob[class_id] = (0.0 - neg_ev) / THRESH;
            } else if (local_target == -1) {
                // We want to learn the negative evidence for this class, and supress the positive evidence.
                // Which, mean that the negative evidence should reach -THRESH, and positive evidence should reach 0.
                // So, nprob will be (-THRESH - neg_ev) / -THRESH, This will produce values from 1 to 0 as neg_ev goes
                // from 0 to -THRESH. And, pprob will be (0 - pos_ev) / -THRESH, This will produce values from 1 to 0 as
                // pos_ev goes from THRESH to 0.
                pprob[class_id] = (0.0 - pos_ev) / -THRESH;
                nprob[class_id] = (-THRESH - neg_ev) / -THRESH;
            }
#else
            // Normal case.
            double y = (double)THRESH * (double)local_target;
            double g = (local_target == 1) ? g_pos[class_id] : g_neg[class_id];
            double h = (local_target == 1) ? H[class_id] : (1.0 - H[class_id]);
            if (h == 1.0) h = 0.999999;
            if (h == 0.0) h = 0.000001;
            double class_sum = (double)CLIP(positive_evidence[class_id] + negative_evidence[class_id], -THRESH, THRESH);
            pprob[class_id] = uprob_fun(class_sum, y, h, g);
            nprob[class_id] = pprob[class_id];
#endif
        }
    }

    __global__ void clause_update(curandState* rng, unsigned int* global_ta_states, float* clause_weights,
                                  const int* selected_patch_ids, const int* num_includes,
                                  const unsigned int* clause_drop_mask, const unsigned int* literal_mask,
                                  const unsigned int* X_batch, const int* targets, const double* pprob,
                                  const double* nprob, const int e) {
        ull index = blockIdx.x * blockDim.x + threadIdx.x;
        ull stride = blockDim.x * gridDim.x;
        curandState localRNG = rng[index];

        for (ull clause = index; clause < CLAUSES; clause += stride) {
            // Skip dropped clauses
            if (clause_drop_mask[clause] == 1) continue;

            unsigned int* ta_state = &global_ta_states[clause * LITERALS];
            int local_clause_output = selected_patch_ids[clause] > -1 ? 1 : 0;
            const unsigned int* X = &X_batch[(ull)e * (ull)(PATCHES * NUM_LITERAL_CHUNKS)];
            const unsigned int* patch =
                selected_patch_ids[clause] > -1 ? &X[selected_patch_ids[clause] * NUM_LITERAL_CHUNKS] : nullptr;

            ull class_id;
            LOOP_CLASS_ID(class_id, clause) {
                int local_target = targets[e * CLASSES + class_id];
                if (local_target == 0) continue;

                float* local_weight = &clause_weights[clause * CLASSES + class_id];
                int sign = (*local_weight >= 0) - (*local_weight < 0);

                double update_prob = (sign == 1) ? pprob[class_id] : nprob[class_id];
                bool should_update = (curand_uniform(&localRNG) <= update_prob);
                bool clause_has_space = (num_includes[clause] <= MAX_INCLUDED_LITERALS);
                bool t1 = (local_target * sign) > 0;

#if TYPE1A_FB
                // Type 1a feedback - TP - if the clause is active and has the correct polarity for the target class, and has
                // space
                if (should_update && t1 && local_clause_output && clause_has_space) {
                    type1a_fb(&localRNG, ta_state, patch, sign, literal_mask);
    #if WEIGHTED
                    if (fabs(*local_weight) < MAX_WEIGHT) (*local_weight) += sign * 1.0f;
    #endif
                }
#endif

#if TYPE1B_FB
                // Type 1b feedback - FN - If clause is inactive, but should have been active (has correct polarity for target), OR
                // if the clause is not overflowing
                if (should_update && t1 && !(local_clause_output && clause_has_space)) {
                    type1b_fb(&localRNG, ta_state, sign, literal_mask);
                }
#endif

#if TYPE2_FB
                // Type 2 feedback - FP - if the clause is active, but has the wrong polarity for the target class
                if (should_update && (local_target * sign) < 0 && local_clause_output) {
                    type2_fb(ta_state, patch, literal_mask);
    #if WEIGHTED
                    if (fabs(*local_weight) < MAX_WEIGHT) (*local_weight) -= sign * 1.0f;
        #if ALLOW_POLARITY_CHANGE == 0
                    if (sign == 1 && *local_weight < 0) *local_weight = 1;
                    if (sign == -1 && *local_weight >= 0) *local_weight = -1;
        #endif
    #endif
    #if NEGATIVE_CLAUSES == 0
                    if (*local_weight < 1) *local_weight = 1;
    #endif
                }
#endif
            }
        }
        rng[index] = localRNG;
    }
}
