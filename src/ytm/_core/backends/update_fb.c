#ifdef IS_NEOVIM_CLANGD_ENV
#include "common.h"
#include "cpu.h"
#include "feedback.h"
#include "rng.h"
#endif

// Applying the feedback lists built by each model's decide_feedback. T1b first, then T1a and T2.

// T1b for every clause in the T1b list, all its classes merged into one pass.
void update_t1b(const ull seed, const uint* restrict t1b_n, const uint* restrict t1b_cid,
                const NCLASSES_T* restrict t1b_cnt, TA_STATE_T* restrict global_ta_states,
                int8_t* restrict is_clause_synced) {
#if TYPE1B_FB
    const uint n = *t1b_n;
    if (n == 0)
        return;

#pragma omp parallel for schedule(dynamic) num_threads(ytm_n_threads)
    for (ull i = 0; i < n; i++) {
        ull clause = t1b_cid[i];
        TA_STATE_T* ta_states = &global_ta_states[ta_offset(clause, 0)];

        ull rng_k = rng_hash(seed, clause, 0x71B71B71ULL);
        uint rng_counter = 0;

        // Repack only when some literal crossed the include line, nothing else changes the packed clause.
        if (geom_dec_range(rng_k, &rng_counter, binom_params((int)t1b_cnt[i], S_INV), 0, N_LITERALS, ta_states))
            is_clause_synced[clause] = 0;
    }
#endif
}

// T1a and T2 for every clause in the T1a/T2 list, all its classes merged into one pass. These clauses
// all fired, so the selected patch is always valid.
void update_t1a_t2(const ull seed, const NPATCHES_T* restrict selected_patch_ids, const FBOUND_T* restrict X,
                   const int e, const NLITS_T* restrict literal_offsets, const uint* restrict t1a_t2_n,
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

#if (N_PATCHES > 1)
        int patch_id = (int)selected_patch_ids[clause];
        int patch_idx_y = patch_id / N_PATCHES_X;
        int patch_idx_x = patch_id % N_PATCHES_X;
#else
        int patch_idx_y = 0, patch_idx_x = 0;
#endif

        ull rng_k = rng_hash(seed, clause, 0x71A2712AULL);
        uint rng_counter = 0;

        if (t1a_t2_fb(rng_k, &rng_counter, (int)t1a_cnt[i], (int)t2_cnt[i], Xe, patch_idx_y, patch_idx_x,
                  literal_offsets, ta_states, 0, 0, N_RAW_PATCH_FEATS, true))
            is_clause_synced[clause] = 0;
    }
}
