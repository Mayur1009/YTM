#include <curand_kernel.h>

extern "C" {
__global__ void encode(unsigned int *X_indptr, unsigned int *X_indices, unsigned int *encoded_X, int e,
                       int class_features) {
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
            int patch_pos =
                class_features + (DIM1 - PATCH_DIM1) + (DIM0 - PATCH_DIM0) + p_y * PATCH_DIM0 * DIM2 + p_x * DIM2 + z;
#else
            int patch_pos = class_features + p_y * PATCH_DIM0 * DIM2 + p_x * DIM2 + z;
#endif

            int chunk_nr = patch_pos / 32;
            int chunk_pos = patch_pos % 32;
            encoded_X[patch * LA_CHUNKS + chunk_nr] |= (1U << chunk_pos);

#if APPEND_NEGATED
            chunk_nr = (patch_pos + (FEATURES / 2)) / 32;
            chunk_pos = (patch_pos + (FEATURES / 2)) % 32;
            encoded_X[patch * LA_CHUNKS + chunk_nr] &= ~(1U << chunk_pos);
#endif
        }
    }
}

__global__ void encode_packed(unsigned int *X_indptr, unsigned int *X_indices, unsigned int *encoded_X,
                              int e, int class_features) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    unsigned int *indices = &X_indices[X_indptr[e]];
    int number_of_indices = X_indptr[e + 1] - X_indptr[e];

    // Looping over all pixels that are 1
    for (int k = 0; k < number_of_indices; ++k) {
        // Coordinate of the pixel
        int y = indices[k] / (DIM0 * DIM2);
        int x = (indices[k] % (DIM0 * DIM2)) / DIM2;
        int z = (indices[k] % (DIM0 * DIM2)) % DIM2;

        // Looping over each patch
        for (int patch = index; patch < PATCHES; patch += stride) {
            // Coordinate of the patch
            int patch_coordinate_y = patch / (DIM0 - PATCH_DIM0 + 1);
            int patch_coordinate_x = patch % (DIM0 - PATCH_DIM0 + 1);

            // Ignore patch if the pixel is not inside this patch
            if ((y < patch_coordinate_y) || (y >= patch_coordinate_y + PATCH_DIM1) || (x < patch_coordinate_x) ||
                (x >= patch_coordinate_x + PATCH_DIM0)) {
                continue;
            }

            int chunk = patch / 32;
            int pos = patch % 32;

            // Coordinate of this pixel relative to this patch, meaning location inside the patch
            int p_y = y - patch_coordinate_y;
            int p_x = x - patch_coordinate_x;

            // Location of this pixel, when all features are layed out in 1d format
#if ENCODE_LOC
            int patch_pos =
                class_features + (DIM1 - PATCH_DIM1) + (DIM0 - PATCH_DIM0) + p_y * PATCH_DIM0 * DIM2 + p_x * DIM2 + z;
#else
            int patch_pos = class_features + p_y * PATCH_DIM0 * DIM2 + p_x * DIM2 + z;
#endif

            encoded_X[chunk * FEATURES + patch_pos] |= (1U << pos);

#if APPEND_NEGATED
            encoded_X[chunk * FEATURES + patch_pos + FEATURES / 2] &= ~(1U << pos);
#endif
        }
    }
}

__global__ void produce_autoencoder_example(curandState *state, unsigned int *active_output,
                                            int number_of_active_outputs, unsigned int *indptr_row,
                                            unsigned int *indices_row, int number_of_rows, unsigned int *indptr_col,
                                            unsigned int *indices_col, int number_of_cols, unsigned int *X,
                                            unsigned int *encoded_Y, int target, int accumulation, int T,
                                            int append_negated) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    // int stride = blockDim.x * gridDim.x;

    if (index != 0) {
        return;
    }

    /* Copy state to local memory for efficiency */
    curandState localState = state[index];

    int row;

    int number_of_features = number_of_cols;
    int number_of_literals = 2 * number_of_features;

    // unsigned int number_of_literal_chunks = (number_of_literals - 1) / 32 + 1;

    // Initialize example vector X

    for (int k = 0; k < number_of_features; ++k) {
        int chunk_nr = k / 32;
        int chunk_pos = k % 32;
        X[chunk_nr] &= ~(1U << chunk_pos);
    }

    if (append_negated) {
        for (int k = number_of_features; k < number_of_literals; ++k) {
            int chunk_nr = k / 32;
            int chunk_pos = k % 32;
            X[chunk_nr] |= (1U << chunk_pos);
        }
    }

    if ((indptr_col[active_output[target] + 1] - indptr_col[active_output[target]] == 0) ||
        (indptr_col[active_output[target] + 1] - indptr_col[active_output[target]] == number_of_rows)) {
        // If no positive/negative examples, produce a random example
        for (int a = 0; a < accumulation; ++a) {
            row = curand(&localState) % number_of_rows;
            for (int k = indptr_row[row]; k < indptr_row[row + 1]; ++k) {
                int chunk_nr = indices_row[k] / 32;
                int chunk_pos = indices_row[k] % 32;
                X[chunk_nr] |= (1U << chunk_pos);

                if (append_negated) {
                    chunk_nr = (indices_row[k] + number_of_features) / 32;
                    chunk_pos = (indices_row[k] + number_of_features) % 32;
                    X[chunk_nr] &= ~(1U << chunk_pos);
                }
            }
        }

        for (int i = 0; i < number_of_active_outputs; ++i) {
            if (i == target) {
                // int chunk_nr = active_output[i] / 32;
                // int chunk_pos = active_output[i] % 32;
                // X[chunk_nr] &= ~(1U << chunk_pos);

                encoded_Y[i] = T;
            } else {
                encoded_Y[i] = -T;
            }
        }

        state[index] = localState;

        return;
    }

    for (int a = 0; a < accumulation; ++a) {
        // Pick example randomly among positive examples
        int random_index =
            indptr_col[active_output[target]] +
            (curand(&localState) % (indptr_col[active_output[target] + 1] - indptr_col[active_output[target]]));
        row = indices_col[random_index];

        for (int k = indptr_row[row]; k < indptr_row[row + 1]; ++k) {
            int chunk_nr = indices_row[k] / 32;
            int chunk_pos = indices_row[k] % 32;
            X[chunk_nr] |= (1U << chunk_pos);

            if (append_negated) {
                chunk_nr = (indices_row[k] + number_of_features) / 32;
                chunk_pos = (indices_row[k] + number_of_features) % 32;
                X[chunk_nr] &= ~(1U << chunk_pos);
            }
        }
    }

    for (int i = 0; i < number_of_active_outputs; ++i) {
        if (i == target) {
            // int chunk_nr = active_output[i] / 32;
            // int chunk_pos = active_output[i] % 32;
            // X[chunk_nr] &= ~(1U << chunk_pos);

            encoded_Y[i] = T;
        } else {
            encoded_Y[i] = -T;
        }
    }

    state[index] = localState;
}
}
