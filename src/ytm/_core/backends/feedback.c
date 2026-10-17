#ifdef IS_NEOVIM_CLANGD_ENV
#include "common.h"
#include "cpu.h"
#include "feedback.h"
#include "rng.h"
#endif

/*
 * Type 1b feedback. Parallelized across the T1b list entries.
 *
 * Arguments:
 *   seed              RNG key of the current sample
 *   t1b_n             T1b list length
 *   t1b_cid           T1b list clause ids
 *   t1b_cnt           T1b count of each entry
 *   global_ta_states  TA states of all clauses
 *   is_clause_synced  Flag to indicate synced and stale clauses.
 */
void update_t1b(const ull seed, const uint* restrict t1b_n, const uint* restrict t1b_cid,
                const NCLASSES_T* restrict t1b_cnt, TA_STATE_T* restrict global_ta_states,
                int8_t* restrict is_clause_synced) {
    const uint n = *t1b_n;
    if (n == 0)
        return;

#pragma omp parallel for schedule(dynamic) num_threads(ytm_n_threads)
    for (ull i = 0; i < n; i++) {
        ull clause = t1b_cid[i];
        TA_STATE_T* ta_states = &global_ta_states[ta_offset(clause, 0)];

        ull rng_k = rng_hash(seed, clause, 0x71B71B71ULL);
        uint rng_counter = 0;

        if (dec_range(rng_k, &rng_counter, binom_params((int)t1b_cnt[i], S_INV), 0, N_LITERALS, 0, LANE_COUNT,
                      ta_states))
            is_clause_synced[clause] = 0;
    }
}

/*
 * Type 1a and Type 2 feedback. Parallelized across the T1a/T2 list entries.
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
void update_t1a_t2(const ull seed, const int e, const NPATCHES_T* restrict selected_patch_ids,
                   const FBOUND_T* restrict X, const NLITS_T* restrict literal_offsets, const uint* restrict t1a_t2_n,
                   const uint* restrict t1a_t2_cid, const NCLASSES_T* restrict t1a_cnt,
                   const NCLASSES_T* restrict t2_cnt, TA_STATE_T* restrict global_ta_states,
                   int8_t* restrict is_clause_synced) {
    const FBOUND_T* Xe = &X[sample_offset(e)];
    const uint n = *t1a_t2_n;
    if (n == 0)
        return;

#pragma omp parallel for schedule(dynamic) num_threads(ytm_n_threads)
    for (ull i = 0; i < n; i++) {
        ull clause = t1a_t2_cid[i];
        TA_STATE_T* ta_states = &global_ta_states[ta_offset(clause, 0)];

        int patch_idx_y = 0, patch_idx_x = 0;
        if (N_PATCHES > 1) {
            int patch_id = (int)selected_patch_ids[clause];
            patch_idx_y = patch_id / N_PATCHES_X;
            patch_idx_x = patch_id % N_PATCHES_X;
        }

        ull rng_k = rng_hash(seed, clause, 0x71A2712AULL);
        uint rng_counter = 0;

        if (t1a_t2_fb(rng_k, &rng_counter, (int)t1a_cnt[i], (int)t2_cnt[i], Xe, patch_idx_y, patch_idx_x,
                      literal_offsets, ta_states, 0, 0, N_RAW_PATCH_FEATS, true))
            is_clause_synced[clause] = 0;
    }
}
