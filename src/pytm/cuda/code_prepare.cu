#include <curand_kernel.h>
extern "C" {
__global__ void prepare(curandState *state, unsigned int *global_ta_state, int *clause_weights) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    curandState localState = state[index];

    for (int clause = index; clause < CLAUSES; clause += stride) {
        unsigned int *ta_state = &global_ta_state[clause * LA_CHUNKS * STATE_BITS];

        for (int la_chunk = 0; la_chunk < LA_CHUNKS; ++la_chunk) {
            for (int b = 0; b < STATE_BITS - 1; ++b) {
                ta_state[la_chunk * STATE_BITS + b] = ~0;
            }
            ta_state[la_chunk * STATE_BITS + STATE_BITS - 1] = 0;
        }
    }

    for (int clause = 0; clause < CLAUSES; clause++) {
        for (int class_id = 0; class_id < CLASSES; ++class_id) {
            if (NEGATIVE_CLAUSES)
                clause_weights[class_id * CLAUSES + clause] = 1 - 2 * (curand(&localState) % 2);
            else
                clause_weights[class_id * CLAUSES + clause] = 1;
        }
    }

    state[index] = localState;
}

__global__ void prepare_packed(curandState *state, unsigned int *global_ta_state, unsigned int *included_literals,
                               unsigned int *included_literals_length) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    curandState localState = state[index];

    for (int clause = index; clause < CLAUSES; clause += stride) {
        unsigned int *ta_state = &global_ta_state[clause * LA_CHUNKS * STATE_BITS];

        included_literals_length[clause] = 0;
        for (int literal = 0; literal < FEATURES; ++literal) {
            int chunk = literal / INT_SIZE;
            int pos = literal % INT_SIZE;

            if ((ta_state[chunk * STATE_BITS + STATE_BITS - 1] & (1U << pos)) > 0) {
                included_literals[clause * FEATURES * 2 + included_literals_length[clause] * 2] = literal;
                included_literals_length[clause]++;
            }
        }
    }
    state[index] = localState;
}
}
