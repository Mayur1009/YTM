#ifdef IS_NEOVIM_CLANGD_ENV
#include "common.h"
#include "cuda.h"
#include "feedback.h"
#include "rng.h"

#define FB_NONE 0
#define FB_T1A 1
#define FB_T1B 2
#define FB_T2 3
#endif

#pragma once

__device__ inline void type1a_fb(ull rng_key, uint* rng_counter, const FBOUND_T* Xe, int patch_idx_y, int patch_idx_x,
                                 const NLITS_T* literal_offsets, TA_STATE_T* ta_states, int lane) {
#if TYPE1A_FB
#if POSITION_LITERALS
    if (lane == 0) {
        t1a_incs(rng_key, rng_counter, 0, patch_idx_y, 0, ta_states);
        t1a_incs(rng_key, rng_counter, N_POSITION_FEATS_Y, N_POSITION_FEATS_Y + patch_idx_x, 0, ta_states);

        t1a_decs(rng_key, rng_counter, patch_idx_y, N_POSITION_FEATS_Y, 0, ta_states);
        t1a_decs(rng_key, rng_counter, N_POSITION_FEATS_Y + patch_idx_x, N_POSITION_FEATS, 0, ta_states);
#if NEGATED_LITERALS
        t1a_incs(rng_key, rng_counter, patch_idx_y, N_POSITION_FEATS_Y, N_LITERALS / 2, ta_states);
        t1a_incs(rng_key, rng_counter, N_POSITION_FEATS_Y + patch_idx_x, N_POSITION_FEATS, N_LITERALS / 2, ta_states);

        t1a_decs(rng_key, rng_counter, 0, patch_idx_y, N_LITERALS / 2, ta_states);
        t1a_decs(rng_key, rng_counter, N_POSITION_FEATS_Y, N_POSITION_FEATS_Y + patch_idx_x, N_LITERALS / 2, ta_states);
#endif
    }
#endif

    for (int fid = lane; fid < N_RAW_PATCH_FEATS; fid += WARP_SIZE) {
        int lit_start = N_POSITION_FEATS + (int)literal_offsets[fid];
        int lit_end = N_POSITION_FEATS + (int)literal_offsets[fid + 1];
        int shifted_val = (int)get_feature_value(Xe, patch_idx_y, patch_idx_x, fid);

        t1a_incs(rng_key, rng_counter, lit_start, lit_start + shifted_val, 0, ta_states);
        t1a_decs(rng_key, rng_counter, lit_start + shifted_val, lit_end, 0, ta_states);
#if NEGATED_LITERALS
        t1a_incs(rng_key, rng_counter, lit_start + shifted_val, lit_end, N_LITERALS / 2, ta_states);
        t1a_decs(rng_key, rng_counter, lit_start, lit_start + shifted_val, N_LITERALS / 2, ta_states);
#endif
    }
#endif
}

// The lane walks its own stripe of the literals, geometrically skipping within it, so a literal is
// still decremented with probability S_INV overall.
__device__ inline void type1b_fb(ull rng_key, uint* rng_counter, TA_STATE_T* ta_states, int lane) {
#if TYPE1B_FB
    float suc = geom_sample(rng_key, rng_counter, S_INV) - 1.0f;
    while (suc * (float)WARP_SIZE + (float)lane < (float)N_LITERALS) {
        int idx = (int)suc * WARP_SIZE + lane;
        if (ta_states[idx] > 0)
            ta_states[idx] -= 1;
        suc += geom_sample(rng_key, rng_counter, S_INV);
    }
#endif
}

__device__ inline void type2_fb(const FBOUND_T* Xe, int patch_idx_y, int patch_idx_x,
                                const NLITS_T* literal_offsets, TA_STATE_T* ta_states, int lane) {
#if TYPE2_FB
#if POSITION_LITERALS
    if (lane == 0) {
        inc_lits(patch_idx_y, N_POSITION_FEATS_Y, 0, ta_states);
        inc_lits(N_POSITION_FEATS_Y + patch_idx_x, N_POSITION_FEATS, 0, ta_states);

#if NEGATED_LITERALS
        inc_lits(0, patch_idx_y, N_LITERALS / 2, ta_states);
        inc_lits(N_POSITION_FEATS_Y, N_POSITION_FEATS_Y + patch_idx_x, N_LITERALS / 2, ta_states);
#endif
    }
#endif

    for (int fid = lane; fid < N_RAW_PATCH_FEATS; fid += WARP_SIZE) {
        int lit_start = N_POSITION_FEATS + (int)literal_offsets[fid];
        int lit_end = N_POSITION_FEATS + (int)literal_offsets[fid + 1];
        int shifted_val = (int)get_feature_value(Xe, patch_idx_y, patch_idx_x, fid);

        inc_lits(lit_start + shifted_val, lit_end, 0, ta_states);
#if NEGATED_LITERALS
        inc_lits(lit_start, lit_start + shifted_val, N_LITERALS / 2, ta_states);
#endif
    }
#endif
}

__device__ inline void apply_feedback(ull rng_key, uint* rng_counter, uint8_t fb, const FBOUND_T* Xe, int patch_idx_y,
                                      int patch_idx_x, const NLITS_T* literal_offsets,
                                      TA_STATE_T* ta_states, int lane) {
    if (fb == FB_T1A) {
        type1a_fb(rng_key, rng_counter, Xe, patch_idx_y, patch_idx_x, literal_offsets, ta_states, lane);
    } else if (fb == FB_T1B) {
        type1b_fb(rng_key, rng_counter, ta_states, lane);
    } else {
        type2_fb(Xe, patch_idx_y, patch_idx_x, literal_offsets, ta_states, lane);
    }
}
