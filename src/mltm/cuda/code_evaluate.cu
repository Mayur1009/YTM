#include <curand_kernel.h>
extern "C" {
// Evaluate examples
__global__ void evaluate(unsigned int *global_ta_state, int *clause_weights, int *class_sum, int *X) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int clause = index; clause < CLAUSES; clause += stride) {
        unsigned int *ta_state = &global_ta_state[clause * LA_CHUNKS * STATE_BITS];

        int all_exclude = 1;
        for (int la_chunk = 0; la_chunk < LA_CHUNKS - 1; ++la_chunk) {
            if (ta_state[la_chunk * STATE_BITS + STATE_BITS - 1] > 0) {
                all_exclude = 0;
                break;
            }
        }

        if ((ta_state[(LA_CHUNKS - 1) * STATE_BITS + STATE_BITS - 1] & FILTER) > 0) {
            all_exclude = 0;
        }

        if (all_exclude) {
            continue;
        }

        int clause_output;
        for (int patch = 0; patch < PATCHES; ++patch) {
            clause_output = 1;
            for (int la_chunk = 0; la_chunk < LA_CHUNKS - 1; ++la_chunk) {
                if ((ta_state[la_chunk * STATE_BITS + STATE_BITS - 1] & X[patch * LA_CHUNKS + la_chunk]) !=
                    ta_state[la_chunk * STATE_BITS + STATE_BITS - 1]) {
                    clause_output = 0;
                    break;
                }
            }

            if ((ta_state[(LA_CHUNKS - 1) * STATE_BITS + STATE_BITS - 1] & X[patch * LA_CHUNKS + LA_CHUNKS - 1] &
                 FILTER) != (ta_state[(LA_CHUNKS - 1) * STATE_BITS + STATE_BITS - 1] & FILTER)) {
                clause_output = 0;
            }

            if (clause_output) {
                break;
            }
        }

        if (clause_output) {
            for (int class_id = 0; class_id < CLASSES; ++class_id) {
                int clause_weight = clause_weights[class_id * CLAUSES + clause];
                atomicAdd(&class_sum[class_id], clause_weight);
            }
        }
    }
}

// Evaluate examples
__global__ void evaluate_packed(unsigned int *included_literals, unsigned int *included_literals_length,
                                int *clause_weights, int *class_sum, int *X) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = 0; i < CLASSES; i++) {
        class_sum[i] = 0;
    }

    for (int clause = index; clause < CLAUSES; clause += stride) {
        if (included_literals_length[clause] == 0) {
            continue;
        }

        unsigned int clause_output = 0;
        for (int patch_chunk = 0; patch_chunk < PATCH_CHUNKS - 1; ++patch_chunk) {
            clause_output = (~(0U));
            for (int literal = 0; literal < included_literals_length[clause]; ++literal) {
                clause_output &= X[patch_chunk * FEATURES + included_literals[clause * FEATURES * 2 + literal * 2]];
            }

            if (clause_output) {
                break;
            }
        }

        if (!clause_output) {
            clause_output = PATCH_FILTER;
            for (int literal = 0; literal < included_literals_length[clause]; ++literal) {
                clause_output &=
                    X[(PATCH_CHUNKS - 1) * FEATURES + included_literals[clause * FEATURES * 2 + literal * 2]];
            }
        }

        if (clause_output) {
            for (int class_id = 0; class_id < CLASSES; ++class_id) {
                int clause_weight = clause_weights[class_id * CLAUSES + clause];
                atomicAdd(&class_sum[class_id], clause_weight);
            }
        }
    }
}
}
