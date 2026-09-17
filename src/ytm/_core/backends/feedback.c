#ifdef IS_NEOVIM_CLANGD_ENV
#include "feedback.h"
#include "common.h"
#include "cpu.h"
#include "rng.h"

#define FB_NONE 0
#define FB_T1A 1
#define FB_T1B 2
#define FB_T2 3
#endif

#pragma once

static inline void type1a_fb(ull rng_key, uint* restrict rng_counter, const int* restrict Xe, int patch_idx_y,
                             int patch_idx_x, const int* restrict feat_mins, const int* restrict literal_offsets,
                             TA_STATE_T* restrict ta_states) {
#if TYPE1A_FB
#if POSITION_LITERALS
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
#endif

    for (int fid = 0; fid < N_RAW_PATCH_FEATS; fid++) {
        int lit_start = N_POSITION_FEATS + literal_offsets[fid];
        int lit_end = N_POSITION_FEATS + literal_offsets[fid + 1];
        int shifted_val = get_feature_value(Xe, patch_idx_y, patch_idx_x, fid) - feat_mins[fid];

        t1a_incs(rng_key, rng_counter, lit_start, lit_start + shifted_val, 0, ta_states);
        t1a_decs(rng_key, rng_counter, lit_start + shifted_val, lit_end, 0, ta_states);
#if NEGATED_LITERALS
        t1a_incs(rng_key, rng_counter, lit_start + shifted_val, lit_end, N_LITERALS / 2, ta_states);
        t1a_decs(rng_key, rng_counter, lit_start, lit_start + shifted_val, N_LITERALS / 2, ta_states);
#endif
    }
#endif
}

static inline void type1b_fb(ull rng_key, uint* restrict rng_counter, TA_STATE_T* restrict ta_states) {
#if TYPE1B_FB
    prob_dec_lits(rng_key, rng_counter, S_INV, 0, N_LITERALS, 0, ta_states);
#endif
}

static inline void type2_fb(const int* restrict Xe, int patch_idx_y, int patch_idx_x, const int* restrict feat_mins,
                            const int* restrict literal_offsets, TA_STATE_T* restrict ta_states) {
#if TYPE2_FB
#if POSITION_LITERALS
    inc_lits(patch_idx_y, N_POSITION_FEATS_Y, 0, ta_states);
    inc_lits(N_POSITION_FEATS_Y + patch_idx_x, N_POSITION_FEATS, 0, ta_states);

#if NEGATED_LITERALS
    inc_lits(0, patch_idx_y, N_LITERALS / 2, ta_states);
    inc_lits(N_POSITION_FEATS_Y, N_POSITION_FEATS_Y + patch_idx_x, N_LITERALS / 2, ta_states);
#endif
#endif

    for (int fid = 0; fid < N_RAW_PATCH_FEATS; fid++) {
        int lit_start = N_POSITION_FEATS + literal_offsets[fid];
        int lit_end = N_POSITION_FEATS + literal_offsets[fid + 1];
        int shifted_val = get_feature_value(Xe, patch_idx_y, patch_idx_x, fid) - feat_mins[fid];

        inc_lits(lit_start + shifted_val, lit_end, 0, ta_states);
#if NEGATED_LITERALS
        inc_lits(lit_start, lit_start + shifted_val, N_LITERALS / 2, ta_states);
#endif
    }
#endif
}

static inline void apply_feedback(ull rng_key, uint* restrict rng_counter, uint8_t fb, const int* restrict Xe,
                                  int patch_idx_y, int patch_idx_x, const int* restrict feat_mins,
                                  const int* restrict literal_offsets, TA_STATE_T* restrict ta_states) {
    if (fb == FB_T1A) {
        type1a_fb(rng_key, rng_counter, Xe, patch_idx_y, patch_idx_x, feat_mins, literal_offsets, ta_states);
    } else if (fb == FB_T1B) {
        type1b_fb(rng_key, rng_counter, ta_states);
    } else {
        type2_fb(Xe, patch_idx_y, patch_idx_x, feat_mins, literal_offsets, ta_states);
    }
}
