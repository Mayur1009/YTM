// Ignore this block, it is used to only for neovim clangd lsp.
#ifdef IS_NEOVIM_CLANGD_ENV
    #define CLAUSES_PER_CLASS 100ULL
    #define THRESH 500
    #define S 10.0
    #define DIM0 28ULL
    #define DIM1 28ULL
    #define DIM2 1ULL
    #define CLASSES 10
    #define Q 1
    #define PATCH_DIM0 10
    #define PATCH_DIM1 10
    #define WEIGHTED 1
    #define COALESCED 1
    #define ENCODE_LOC 1
    #define MAX_INCLUDED_LITERALS 272ULL
    #define NEGATED_LITERALS 1
    #define NEGATIVE_POLARITY 1
    #define ALLOW_POLARITY_CHANGE 1
    #define MAX_WEIGHT 3.4e38f
    #define MAX_TA_STATE 255
    #define INCLUDE_TA_STATE 128
    #define TYPE1A_FB 1
    #define TYPE1B_FB 1
    #define TYPE2_FB 1
    #define TOTAL_CLAUSES 100ULL
    #define PATCHES 361ULL
    #define LITERALS 272ULL
#endif

#if ((LITERALS / 2) & 1)        // Ensure that LITERALS/2 is even, because the vectorized code does not work
                                // otherwise.......dont know why....some memory aligment issue
    #define VECTORIZED_LIMIT 0  // odd
#else
    #define VECTORIZED_LIMIT (LITERALS & ~3)  // even
#endif
#define S_INV (1.0f / S)
#define Q_PROB (1.0f * Q / max(1, CLASSES - 1))
#define INT_SIZE 32
#define NUM_LITERAL_CHUNKS (((LITERALS - 1) / INT_SIZE) + 1)
#if ((LITERALS % INT_SIZE) != 0)
    #define FILTER (~(0xFFFFFFFF << (LITERALS % INT_SIZE)))
#else
    #define FILTER 0xFFFFFFFF
#endif

#if COALESCED == 0
    #define LOOP_CLASS_ID(class_id, clause) class_id = (ull)clause / CLAUSES_PER_CLASS;
#else
    #define LOOP_CLASS_ID(class_id, clause) for (class_id = 0; class_id < CLASSES; ++class_id)
#endif

#define CLIP(val, min, max) ((val < min) ? min : ((val > max) ? max : val))

typedef unsigned long long ull;

void encode_batch(const int* X, unsigned int* encoded_X, const int N) {
    // X -> (N * DIM0 * DIM1 * DIM2) array with possible values {-1, 0, 1}.
    // 1 -> feat present, -1 -> feat absent, 0 -> dont care
    // encoded_X -> (N * PATCHES * NUM_LITERAL_CHUNKS)

    for (ull e = 0; e < N; ++e) {
        for (ull patch_id = 0; patch_id < PATCHES; ++patch_id) {
            // Calculate the starting point of the patch in the original image
            int patch_coordinate_y = patch_id / (DIM0 - PATCH_DIM0 + 1);
            int patch_coordinate_x = patch_id % (DIM0 - PATCH_DIM0 + 1);

            ull encX_offset = e * (ull)(PATCHES * NUM_LITERAL_CHUNKS) + patch_id * (ull)NUM_LITERAL_CHUNKS;
            unsigned int* patch_output = &encoded_X[encX_offset];

            // Initialization.
            // By default, all values in encoded_X are set to 0 (in python code).
            // So, only need to initialize all negated literals to 1.
#if NEGATED_LITERALS
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
#if NEGATED_LITERALS
                int neg_chunk_nr = (lit + (LITERALS / 2)) / INT_SIZE;
                int neg_chunk_pos = (lit + (LITERALS / 2)) % INT_SIZE;
                patch_output[neg_chunk_nr] &= ~(1u << neg_chunk_pos);
#endif
            }

            for (int lit = 0; lit < patch_coordinate_x; ++lit) {
                int chunk_nr = (DIM1 - PATCH_DIM1 + lit) / INT_SIZE;
                int chunk_pos = (DIM1 - PATCH_DIM1 + lit) % INT_SIZE;
                patch_output[chunk_nr] |= (1u << chunk_pos);
#if NEGATED_LITERALS
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
#if NEGATED_LITERALS
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
}
