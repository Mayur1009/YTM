#ifdef IS_NEOVIM_CLANGD_ENV
#include "common.h"
#include "cuda.h"
#endif

/*
 * Type 1b feedback. One warp per T1b list entry, lanes split the clause's literals.
 *
 * Arguments:
 *   seed              RNG key of the current sample
 *   t1b_n             T1b list length
 *   t1b_cid           T1b list clause ids
 *   t1b_cnt           T1b count of each entry
 *   global_ta_states  TA states of all clauses
 *   is_clause_synced  Flag to indicate synced and stale clauses.
 */
extern "C" __global__ void update_t1b(const ull seed, const uint* t1b_n, const uint* t1b_cid,
                                      const NCLASSES_T* t1b_cnt, TA_STATE_T* global_ta_states,
                                      int8_t* is_clause_synced) {
    auto [warp, lane, warp_id, total_warps] = warp_grid();
    const uint n = *t1b_n;

    WARP_STRIDE_LOOP(i, (ull)n) {
        ull clause = t1b_cid[i];
        TA_STATE_T* ta_states = &global_ta_states[ta_offset(clause, 0)];

        ull rng_k = rng_hash(seed, clause, (0x71B71B71ULL << 32) | (ull)lane);
        uint rng_counter = 0;

        bool crossed = dec_range(rng_k, &rng_counter, binom_params((int)t1b_cnt[i], S_INV), 0, N_LITERALS, 0,
                                 LANE_COUNT, ta_states);
        if (warp.any(crossed) && lane == 0)
            is_clause_synced[clause] = 0;
    }
}

/*
 * Type 1a and Type 2 feedback. One warp per T1a/T2 list entry, lanes split the clause's patch features.
 *
 * Arguments:
 *   seed                RNG key of the current sample
 *   e                   index of the current sample
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
extern "C" __global__ void update_t1a_t2(const ull seed, const int e, const NPATCHES_T* selected_patch_ids,
                                         const FBOUND_T* X, const NLITS_T* literal_offsets, const uint* t1a_t2_n,
                                         const uint* t1a_t2_cid, const NCLASSES_T* t1a_cnt, const NCLASSES_T* t2_cnt,
                                         TA_STATE_T* global_ta_states, int8_t* is_clause_synced) {
    auto [warp, lane, warp_id, total_warps] = warp_grid();
    const FBOUND_T* Xe = &X[sample_offset(e)];
    const uint n = *t1a_t2_n;

    WARP_STRIDE_LOOP(i, (ull)n) {
        ull clause = t1a_t2_cid[i];
        TA_STATE_T* ta_states = &global_ta_states[ta_offset(clause, 0)];

        int patch_idx_y = 0, patch_idx_x = 0;
        if (N_PATCHES > 1) {
            int patch_id = (int)selected_patch_ids[clause];
            patch_idx_y = patch_id / N_PATCHES_X;
            patch_idx_x = patch_id % N_PATCHES_X;
        }

        ull rng_k = rng_hash(seed, clause, (0x71A2712AULL << 32) | (ull)lane);
        uint rng_counter = 0;

        bool crossed = t1a_t2_fb(rng_k, &rng_counter, (int)t1a_cnt[i], (int)t2_cnt[i], Xe, patch_idx_y, patch_idx_x,
                                 literal_offsets, ta_states, lane, 0, N_RAW_PATCH_FEATS, true);
        if (warp.any(crossed) && lane == 0)
            is_clause_synced[clause] = 0;
    }
}
