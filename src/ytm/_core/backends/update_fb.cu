#ifdef IS_NEOVIM_CLANGD_ENV
#include "common.h"
#include "cuda.h"
#include "feedback.h"
#include "rng.h"
#endif

// Applying the feedback lists built by each model's decide_feedback. T1b first, then T1a and T2, on one stream.

// Tiles below this width cost more to set up than they save.
#ifndef T1B_MIN_TILE
#define T1B_MIN_TILE 256
#endif

// T1b for every clause in the T1b list, all its classes merged into one pass. Each clause's literals are
// split into tiles so that few clauses still fill every resident warp, and a whole warp samples one tile
// together. Restarting the geometric skip at a tile edge is exact. The tiling follows the device's warp count,
// so the same seed gives different, equally valid, draws on different GPUs.
extern "C" __global__ void update_t1b(const ull* keys, const int* e_dev, const uint* t1b_n, const uint* t1b_cid,
                                      const NCLASSES_T* t1b_cnt, TA_STATE_T* global_ta_states,
                                      int8_t* is_clause_synced) {
    const int e = *e_dev;
    const ull seed = keys[e];
#if TYPE1B_FB
    auto [warp, lane, warp_id, total_warps] = warp_grid();
    const uint n = *t1b_n;
    if (n == 0)
        return;

    const uint max_tiles = (N_LITERALS + T1B_MIN_TILE - 1) / T1B_MIN_TILE;
    ull tiles = (total_warps + n - 1) / n;
    tiles = tiles > max_tiles ? max_tiles : (tiles < 1 ? 1 : tiles);
    const int tile_len = (N_LITERALS + (int)tiles - 1) / (int)tiles;

    WARP_STRIDE_LOOP(t, (ull)n * tiles) {
        ull i = t / tiles;
        int tile = (int)(t % tiles);
        ull clause = t1b_cid[i];
        TA_STATE_T* ta_states = &global_ta_states[ta_offset(clause, 0)];

        int start = tile * tile_len;
        int end = (start + tile_len < N_LITERALS) ? start + tile_len : N_LITERALS;

        ull rng_k = rng_hash(rng_hash(seed, clause, (ull)tile), (ull)lane, 0x71B71B71ULL);
        uint rng_counter = 0;

        // Repack only when some literal crossed the include line, nothing else changes the packed clause.
        bool crossed = geom_dec_range(rng_k, &rng_counter, binom_params((int)t1b_cnt[i], S_INV), start, end, ta_states);
        if (warp.any(crossed) && lane == 0)
            is_clause_synced[clause] = 0;
    }
#endif
}

#ifndef T1A_T2_MIN_TILE
#define T1A_T2_MIN_TILE 256
#endif

// T1a and T2 for every clause in the T1a/T2 list, all its classes merged into one pass. These clauses
// all fired, so the selected patch is always valid. Each clause's patch features are split into tiles of
// whole features, balanced by literal count, so few clauses still fill every resident warp and no
// thermometer is cut. Inside a tile, lanes stride over its features.
extern "C" __global__ void update_t1a_t2(const ull* keys, const int* e_dev, const NPATCHES_T* selected_patch_ids, const FBOUND_T* X,
                                         const NLITS_T* literal_offsets, const uint* t1a_t2_n,
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
    tiles = tiles > max_tiles ? max_tiles : (tiles < 1 ? 1 : tiles);
    const int tile_len = (n_lits + (int)tiles - 1) / (int)tiles;

    WARP_STRIDE_LOOP(t, (ull)n * tiles) {
        ull i = t / tiles;
        int tile = (int)(t % tiles);
        ull clause = t1a_t2_cid[i];
        TA_STATE_T* ta_states = &global_ta_states[ta_offset(clause, 0)];

#if (N_PATCHES > 1)
        int patch_id = (int)selected_patch_ids[clause];
        int patch_idx_y = patch_id / N_PATCHES_X;
        int patch_idx_x = patch_id % N_PATCHES_X;
#else
        int patch_idx_y = 0, patch_idx_x = 0;
#endif

        int fid_start = first_feat_from(literal_offsets, tile * tile_len);
        int fid_end = (tile + 1 == (int)tiles) ? N_RAW_PATCH_FEATS : first_feat_from(literal_offsets, (tile + 1) * tile_len);

        ull rng_k = rng_hash(rng_hash(seed, clause, (ull)tile), (ull)lane, 0x71A2712AULL);
        uint rng_counter = 0;

        bool crossed = t1a_t2_fb(rng_k, &rng_counter, (int)t1a_cnt[i], (int)t2_cnt[i], Xe, patch_idx_y, patch_idx_x,
                  literal_offsets, ta_states, lane, fid_start, fid_end, tile == 0);
        if (warp.any(crossed) && lane == 0)
            is_clause_synced[clause] = 0;
    }
}

// First node of each sample: moves the sample index on and empties the feedback lists this sample will fill.
extern "C" __global__ void prep_iter(int* e_dev, uint* t1b_n, uint* t1a_t2_n) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *e_dev += 1;
        *t1b_n = 0;
        *t1a_t2_n = 0;
    }
}
