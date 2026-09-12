#ifdef IS_NEOVIM_CLANGD_ENV
#include "cpu.h"
#include "common.h"
#include "rng.h"
#endif

#pragma once

INLINE_FN void inc_lits(int start, int end, int offset, uint* ta_states) {
    for (int li = start; li < end; ++li)
        if (ta_states[li + offset] < MAX_TA_STATE)
            ta_states[li + offset] += 1;
}

INLINE_FN void dec_lits(int start, int end, int offset, uint* ta_states) {
    for (int li = start; li < end; ++li)
        if (ta_states[li + offset] > 0)
            ta_states[li + offset] -= 1;
}

INLINE_FN void prob_inc_lits(ull rng_key, uint* rng_counter, float prob, int start, int end, int offset,
                             uint* ta_states) {
    float li = (float)start + geom_sample(rng_key, rng_counter, prob) - 1.0f;
    while (li < (float)end) {
        int idx = (int)li + offset;
        if (ta_states[idx] < MAX_TA_STATE)
            ta_states[idx] += 1;
        li += geom_sample(rng_key, rng_counter, prob);
    }
}

INLINE_FN void prob_dec_lits(ull rng_key, uint* rng_counter, float prob, int start, int end, int offset,
                             uint* ta_states) {
    float li = (float)start + geom_sample(rng_key, rng_counter, prob) - 1.0f;
    while (li < (float)end) {
        int idx = (int)li + offset;
        if (ta_states[idx] > 0)
            ta_states[idx] -= 1;
        li += geom_sample(rng_key, rng_counter, prob);
    }
}

INLINE_FN void t1a_incs(ull rng_key, uint* rng_counter, int start, int end, int offset, uint* ta_states) {
#if BOOST_TP_INC
    (void)rng_key;
    (void)rng_counter;
    inc_lits(start, end, offset, ta_states);
#else
    prob_inc_lits(rng_key, rng_counter, 1.0f - S_INV, start, end, offset, ta_states);
#endif
}

INLINE_FN void t1a_decs(ull rng_key, uint* rng_counter, int start, int end, int offset, uint* ta_states) {
#if BOOST_TP_DEC
    (void)rng_key;
    (void)rng_counter;
    dec_lits(start, end, offset, ta_states);
#else
    prob_dec_lits(rng_key, rng_counter, S_INV, start, end, offset, ta_states);
#endif
}

INLINE_FN void type1a_fb(ull rng_key, uint* rng_counter, const int* Xe, int patch_idx_y, int patch_idx_x,
                         const int* feat_mins, const int* literal_offsets, uint* ta_states) {
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
}

INLINE_FN void type1b_fb(ull rng_key, uint* rng_counter, uint* ta_states) {
    prob_dec_lits(rng_key, rng_counter, S_INV, 0, N_LITERALS, 0, ta_states);
}

INLINE_FN void type2_fb(const int* Xe, int patch_idx_y, int patch_idx_x, const int* feat_mins,
                        const int* literal_offsets, uint* ta_states) {
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
}
