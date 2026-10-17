#ifdef IS_NEOVIM_CLANGD_ENV
#include "common.h"
#include "cuda.h"
#endif

/*
 * Type 1b feedback. Parallelized across all the literals
 *
 * Arguments:
 *   keys              RNG key of each sample in the batch
 *   e_dev             index of the current sample
 *   t1b_n             T1b list length
 *   t1b_cid           T1b list clause ids
 *   t1b_cnt           T1b count of each entry
 *   global_ta_states  TA states of all clauses
 *   is_clause_synced  Flag to indicate synced and stale clauses.
 */
extern "C" __global__ void update_t1b(const ull* keys, const int* e_dev, const uint* t1b_n, const uint* t1b_cid,
                                      const NCLASSES_T* t1b_cnt, TA_STATE_T* global_ta_states,
                                      int8_t* is_clause_synced) {
    const int e = *e_dev;
    const ull seed = keys[e];
    auto [warp, lane, warp_id, total_warps] = warp_grid();
    const uint n = *t1b_n;
    if (n == 0)
        return;

    // Calculates tiles to split the work.
    const uint max_tiles = (N_LITERALS + T1B_MIN_TILE - 1) / T1B_MIN_TILE;
    ull tiles = (total_warps + n - 1) / n;
    tiles = tiles > max_tiles ? max_tiles : tiles;
    const int tile_len = (N_LITERALS + (int)tiles - 1) / (int)tiles;

    WARP_STRIDE_LOOP(t, (ull)n * tiles) {
        ull i = t / tiles;
        int tile = (int)(t % tiles);
        ull clause = t1b_cid[i];
        TA_STATE_T* ta_states = &global_ta_states[ta_offset(clause, 0)];

        int start = tile * tile_len;
        int end = (start + tile_len < N_LITERALS) ? start + tile_len : N_LITERALS;

        ull rng_k = rng_hash(seed, clause, (0x71B71B71ULL << 32) | ((ull)tile << 5) | (ull)lane);
        uint rng_counter = 0;

        bool crossed =
            dec_range(rng_k, &rng_counter, binom_params((int)t1b_cnt[i], S_INV), start, end, 0, LANE_COUNT, ta_states);
        if (warp.any(crossed) && lane == 0)
            is_clause_synced[clause] = 0;
    }
}

/*
 * Moves a tile cut to the next feature boundary.
 *
 * update_t1a_t2 splits a clause's patch literals into tiles of about equal size, one warp per tile. A
 * thermometer feature must stay whole in one tile, since its value decides where its 1s end and 0s begin and
 * one lane handles all its literals. So a cut wanted at patch literal x moves to the first feature that starts
 * at or after x, and that feature begins the next tile. Returns N_RAW_PATCH_FEATS when no feature starts at or
 * after x. Binary search over literal_offsets; with all binary features, feature f is literal f.
 */
__device__ inline int first_feat_from(const NLITS_T* literal_offsets, int x) {
    if (ALL_BINARY_FEATS)
        return x < N_RAW_PATCH_FEATS ? x : N_RAW_PATCH_FEATS;
    int lo = 0, hi = N_RAW_PATCH_FEATS;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if ((int)literal_offsets[mid] < x)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo;
}
/*
 * Type 1a and Type 2 feedback. Parallelized across the feature tiles.
 *
 * Arguments:
 *   keys                RNG key of each sample in the batch
 *   e_dev               index of the current sample
 *   selected_patch_ids  patch each clause fired on
 *   X                   features of the batch
 *   literal_offsets     first literal of each patch feature
 *   t1a_t2_n            T1a/T2 list length
 *   t1a_t2_cid          T1a/T2 list clause ids
 *   t1a_cnt             T1a count of each entry
 *   t2_cnt              T2 count of each entry
 *   global_ta_states    TA states of all clauses
 *   is_clause_synced    Flag to indicate synced and stale clauses.
 */
extern "C" __global__ void update_t1a_t2(const ull* keys, const int* e_dev, const NPATCHES_T* selected_patch_ids,
                                         const FBOUND_T* X, const NLITS_T* literal_offsets, const uint* t1a_t2_n,
                                         const uint* t1a_t2_cid, const NCLASSES_T* t1a_cnt, const NCLASSES_T* t2_cnt,
                                         TA_STATE_T* global_ta_states, int8_t* is_clause_synced) {
    const int e = *e_dev;
    const ull seed = keys[e];
    auto [warp, lane, warp_id, total_warps] = warp_grid();

    const FBOUND_T* Xe = &X[sample_offset(e)];
    const uint n = *t1a_t2_n;
    if (n == 0)
        return;

#if ALL_BINARY_FEATS
    const int n_lits = N_RAW_PATCH_FEATS;
#else
    const int n_lits = (int)literal_offsets[N_RAW_PATCH_FEATS];
#endif
    ull max_tiles = ((ull)n_lits + T1A_T2_MIN_TILE - 1) / T1A_T2_MIN_TILE;
    max_tiles = max_tiles < 1 ? 1 : max_tiles;
    ull tiles = (total_warps + n - 1) / n;
    tiles = tiles > max_tiles ? max_tiles : tiles;
    const int tile_len = (n_lits + (int)tiles - 1) / (int)tiles;

    WARP_STRIDE_LOOP(t, (ull)n * tiles) {
        ull i = t / tiles;
        int tile = (int)(t % tiles);
        ull clause = t1a_t2_cid[i];
        TA_STATE_T* ta_states = &global_ta_states[ta_offset(clause, 0)];

        int patch_idx_y = 0, patch_idx_x = 0;
        if (N_PATCHES > 1) {
            int patch_id = (int)selected_patch_ids[clause];
            patch_idx_y = patch_id / N_PATCHES_X;
            patch_idx_x = patch_id % N_PATCHES_X;
        }

        int fid_start = first_feat_from(literal_offsets, tile * tile_len);
        int fid_end =
            (tile + 1 == (int)tiles) ? N_RAW_PATCH_FEATS : first_feat_from(literal_offsets, (tile + 1) * tile_len);

        ull rng_k = rng_hash(seed, clause, (0x71A2712AULL << 32) | ((ull)tile << 5) | (ull)lane);
        uint rng_counter = 0;

        bool crossed = t1a_t2_fb(rng_k, &rng_counter, (int)t1a_cnt[i], (int)t2_cnt[i], Xe, patch_idx_y, patch_idx_x,
                                 literal_offsets, ta_states, lane, fid_start, fid_end, tile == 0);
        if (warp.any(crossed) && lane == 0)
            is_clause_synced[clause] = 0;
    }
}
