#include <curand_kernel.h>
extern "C" {
// Transform examples

__global__ void transform(unsigned int* included_literals, unsigned int* included_literals_length, int* X,
                          int* transformed_X) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int clause = index; clause < CLAUSES; clause += stride) {
        if (included_literals_length[clause] == 0) {
            transformed_X[clause] = 0;
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

        transformed_X[clause] = clause_output;
    }
}

__global__ void transform_patchwise(unsigned int* included_literals, unsigned int* included_literals_length, int* X,
                                    int* transformed_X) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int clause = index; clause < CLAUSES; clause += stride) {
        if (included_literals_length[clause] == 0) {
            for (int patch = 0; patch < PATCHES; patch++) {
                transformed_X[clause * PATCHES + patch] = 0;
            }
            continue;
        }
        unsigned int clause_output = 0;
        for (int patch_chunk = 0; patch_chunk < PATCH_CHUNKS - 1; ++patch_chunk) {
            clause_output = (~(0U));
            for (int literal = 0; literal < included_literals_length[clause]; ++literal) {
                clause_output &= X[patch_chunk * FEATURES + included_literals[clause * FEATURES * 2 + literal * 2]];
            }

            for (int bit = 0; bit < INT_SIZE; bit++) {
                transformed_X[clause * PATCHES + patch_chunk * INT_SIZE + bit] = (clause_output & (1 << bit)) > 0;
            }
        }

        clause_output = PATCH_FILTER;
        for (int literal = 0; literal < included_literals_length[clause]; ++literal) {
            clause_output &= X[(PATCH_CHUNKS - 1) * FEATURES + included_literals[clause * FEATURES * 2 + literal * 2]];
        }

        for (int bit = 0; bit < INT_SIZE; bit++) {
            transformed_X[clause * PATCHES + (PATCH_CHUNKS - 1) * INT_SIZE + bit] = clause_output & (1 << bit) > 0;
        }
    }
}
}
