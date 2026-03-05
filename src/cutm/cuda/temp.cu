
    /***********CLAUSE EVALUATION---SLOWER***********/
    // __global__ void clause_eval(curandState *rng, const unsigned int *packed_ta_states, const float *clause_weights,
    //                             int *patch_weights, const unsigned int *X_batch, int *selected_patch_ids,
    //                             float *class_sums, const int e) {
    //     ull index = blockIdx.x * blockDim.x + threadIdx.x;
    //     ull stride = blockDim.x * gridDim.x;
    //
    //     curandState localRNG = rng[index];
    //
    //     for (ull clause = index; clause < CLAUSES; clause += stride) {
    //         int active_patches[PATCHES];
    //         int active_count = 0;
    //
    //         for (ull patch_id = 0; patch_id < PATCHES; ++patch_id) {
    //             int patch_matched = clause_match(
    //                 &packed_ta_states[clause * NUM_LITERAL_CHUNKS],
    //                 &X_batch[(ull)e * (ull)(PATCHES * NUM_LITERAL_CHUNKS) + patch_id * NUM_LITERAL_CHUNKS]);
    //             if (patch_matched) {
    //                 active_patches[active_count] = patch_id;
    //                 active_count++;
    //             }
    //         }
    //         if (active_count > 0) {
    //             int random_index = curand(&localRNG) % active_count;
    //             selected_patch_ids[clause] = active_patches[random_index];
    //             patch_weights[clause * PATCHES + active_patches[random_index]] = 1;
    //             for (int class_id = 0; class_id < CLASSES; ++class_id) {
    //                 atomicAdd(&class_sums[0 * CLASSES + class_id], clause_weights[clause * CLASSES + class_id]);
    //             }
    //         } else {
    //             selected_patch_ids[clause] = -1;
    //         }
    //     }
    //     rng[index] = localRNG;
    // }
    //
    // __device__ inline void type1a_fb_scalar(curandState *rng, unsigned int *ta_state, const unsigned int *patch,
    // const unsigned int *literal_mask) {
    //     for (int li = 0; li < LITERALS; ++li) {
    //         unsigned int patch_bit = (patch[li / INT_SIZE] >> (li % INT_SIZE)) & 1u;
    //         if (patch_bit == 1 && ta_state[li] < MAX_TA_STATE) {
    //             ta_state[li] += 1;
    //         } else if (patch_bit == 0 && ta_state[li] > 0 && curand_uniform(rng) <= S_INV) {
    //             ta_state[li] -= 1;
    //         }
    //     }
    // }
    //
    // __device__ inline void type1b_fb_scalar(curandState *rng, unsigned int *ta_state, const int sign, const unsigned
    // int *literal_mask) {
    //     float s_inv = (sign == 1) ? S_INV : S_NEG_POLARITY_INV;
    //     for (int li = 0; li < LITERALS; ++li) {
    //         if (ta_state[li] > 0 && curand_uniform(rng) <= s_inv) {
    //             ta_state[li] -= 1;
    //         }
    //     }
    // }
    //
    // __device__ inline void type2_fb_scalar(unsigned int *ta_state, const unsigned int *patch, const unsigned int
    // *literal_mask) {
    //     for (int li = 0; li < LITERALS; ++li) {
    //         unsigned int patch_bit = (patch[li / INT_SIZE] >> (li % INT_SIZE)) & 1u;
    //         if (patch_bit == 0 && ta_state[li] < INCLUDE_TA_STATE) {
    //             ta_state[li] += 1;
    //         }
    //     }
    // }
