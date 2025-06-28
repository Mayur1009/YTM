#ifdef IS_NEOVIM_CLANGD_ENV
#define DIM0 28
#define DIM1 28
#define DIM2 1
#define PATCH_DIM0 10
#define PATCH_DIM1 10
#define PATCHES 361
#define LITERALS 272
#define APPEND_NEGATED 1
#define ENCODE_LOC 1
#endif

#define INT_SIZE 32
#define NUM_LITERAL_CHUNKS (((LITERALS - 1) / INT_SIZE) + 1)

extern "C" {
__global__ void encode_batch(const unsigned int *X_indptr, const unsigned int *X_indices, unsigned int *encoded_X,
                             const int N) {
    /*
     * Encode the input data X into patches. Each patch has a set of literals, which are packed into chunks of 32 bits.
     */
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (unsigned long long e_patch = index; e_patch < (unsigned long long)PATCHES * (unsigned long long)N;
         e_patch += stride) {
        int e = e_patch / PATCHES;
        int patch = e_patch % PATCHES;

        const unsigned int *indices = &X_indices[X_indptr[e]];
        int number_of_indices = X_indptr[e + 1] - X_indptr[e];

        // Pre-calculate patch boundaries once per thread
        int patch_coordinate_y = patch / (DIM0 - PATCH_DIM0 + 1);
        int patch_coordinate_x = patch % (DIM0 - PATCH_DIM0 + 1);

        // unsigned int *patch_output = &encoded_X[patch * NUM_LITERAL_CHUNKS];
        unsigned int *patch_output = &encoded_X[e * PATCHES * NUM_LITERAL_CHUNKS + patch * NUM_LITERAL_CHUNKS];

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
}
