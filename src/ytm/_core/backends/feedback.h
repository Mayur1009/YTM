#ifdef IS_NEOVIM_CLANGD_ENV
#include "common.h"
#include "cpu.h"
#include "rng.h"
#define FB_NONE 0
#define FB_T1A 1
#define FB_T1B 2
#define FB_T2 3
#endif

#pragma once

INLINE_FN void safe_lit_inc(int li, TA_STATE_T* ta_states) {
    if (ta_states[li] < MAX_TA_STATE)
        ta_states[li] += 1;
}

INLINE_FN void safe_lit_dec(int li, TA_STATE_T* ta_states) {
    if (ta_states[li] > 0)
        ta_states[li] -= 1;
}

INLINE_FN void inc_literals_in_range(int start, int end, int offset, TA_STATE_T* ta_states) {
    for (int li = start; li < end; ++li)
        safe_lit_inc(li + offset, ta_states);
}

INLINE_FN void dec_literals_in_range(int start, int end, int offset, TA_STATE_T* ta_states) {
    for (int li = start; li < end; ++li)
        safe_lit_dec(li + offset, ta_states);
}

INLINE_FN void prob_inc_literals_in_range(ull rng_key, uint* RESTRICT rng_counter, float prob, int start, int end,
                                          int offset, TA_STATE_T* RESTRICT ta_states) {
    float li = (float)start + geom_sample(rng_key, rng_counter, prob) - 1.0f;
    while (li < (float)end) {
        safe_lit_inc((int)li + offset, ta_states);
        li += geom_sample(rng_key, rng_counter, prob);
    }
}

INLINE_FN void prob_dec_literals_in_range(ull rng_key, uint* RESTRICT rng_counter, float prob, int start, int end,
                                          int offset, TA_STATE_T* RESTRICT ta_states) {
    float li = (float)start + geom_sample(rng_key, rng_counter, prob) - 1.0f;
    while (li < (float)end) {
        safe_lit_dec((int)li + offset, ta_states);
        li += geom_sample(rng_key, rng_counter, prob);
    }
}

INLINE_FN void t2_inc_literals_in_range(int start, int end, int offset, TA_STATE_T* RESTRICT ta_states) {
    for (int li = start; li < end; ++li)
        ta_states[li + offset] += 1;
}

INLINE_FN void t1a_inc_literals_in_range(ull rng_key, uint* RESTRICT rng_counter, int start, int end, int offset,
                                         TA_STATE_T* RESTRICT ta_states) {
#if BOOST_TP_INC
    inc_literals_in_range(start, end, offset, ta_states);
#else
    prob_inc_literals_in_range(rng_key, rng_counter, 1.0f - S_INV, start, end, offset, ta_states);
#endif
}

INLINE_FN void t1a_dec_literals_in_range(ull rng_key, uint* RESTRICT rng_counter, int start, int end, int offset,
                                         TA_STATE_T* RESTRICT ta_states) {
#if BOOST_TP_DEC
    dec_literals_in_range(start, end, offset, ta_states);
#else
    prob_dec_literals_in_range(rng_key, rng_counter, S_INV, start, end, offset, ta_states);
#endif
}

INLINE_FN void t1a_therm(ull rng_key, uint* RESTRICT rng_counter, int lit_start, int lit_end, int val,
                         TA_STATE_T* RESTRICT ta_states) {
    t1a_inc_literals_in_range(rng_key, rng_counter, lit_start, lit_start + val, 0, ta_states);
    t1a_dec_literals_in_range(rng_key, rng_counter, lit_start + val, lit_end, 0, ta_states);
#if NEGATED_LITERALS
    t1a_inc_literals_in_range(rng_key, rng_counter, lit_start + val, lit_end, N_LITERALS / 2, ta_states);
    t1a_dec_literals_in_range(rng_key, rng_counter, lit_start, lit_start + val, N_LITERALS / 2, ta_states);
#endif
}

INLINE_FN void t1a_binary(ull rng_key, uint* RESTRICT rng_counter, int lit, int val, TA_STATE_T* RESTRICT ta_states) {
    if (val) {
        // Increment lit
        if (BOOST_TP_INC || rand_uniform(rng_key, rng_counter) < 1.0f - S_INV)
            safe_lit_inc(lit, ta_states);

#if NEGATED_LITERALS
        // Decrement neg_lit
        if (BOOST_TP_DEC || rand_uniform(rng_key, rng_counter) < S_INV)
            safe_lit_dec(lit + N_LITERALS / 2, ta_states);
#endif

    } else {
        // Decrement lit
        if (BOOST_TP_DEC || rand_uniform(rng_key, rng_counter) < S_INV)
            safe_lit_dec(lit, ta_states);

#if NEGATED_LITERALS
        // Increment neg_lit
        if (BOOST_TP_INC || rand_uniform(rng_key, rng_counter) < 1.0f - S_INV)
            safe_lit_inc(lit + N_LITERALS / 2, ta_states);
#endif
    }
}

INLINE_FN void t2_therm(int lit_start, int lit_end, int val, TA_STATE_T* RESTRICT ta_states) {
    t2_inc_literals_in_range(lit_start + val, lit_end, 0, ta_states);
#if NEGATED_LITERALS
    t2_inc_literals_in_range(lit_start, lit_start + val, N_LITERALS / 2, ta_states);
#endif
}

INLINE_FN void t2_binary(int lit, int val, TA_STATE_T* RESTRICT ta_states) {
    if (!val)
        ta_states[lit] += 1;
#if NEGATED_LITERALS
    else
        ta_states[lit + N_LITERALS / 2] += 1;
#endif
}

INLINE_FN void t1a_feature(ull rng_key, uint* RESTRICT rng_counter, const NLITS_T* RESTRICT literal_offsets, int fid,
                           int val, TA_STATE_T* RESTRICT ta_states) {
#if ALL_BINARY_FEATS
    t1a_binary(rng_key, rng_counter, N_POSITION_FEATS + fid, val, ta_states);
#else
    t1a_therm(rng_key, rng_counter, N_POSITION_FEATS + (int)literal_offsets[fid],
              N_POSITION_FEATS + (int)literal_offsets[fid + 1], val, ta_states);
#endif
}

INLINE_FN void t2_feature(const NLITS_T* RESTRICT literal_offsets, int fid, int val, TA_STATE_T* RESTRICT ta_states) {
#if ALL_BINARY_FEATS
    t2_binary(N_POSITION_FEATS + fid, val, ta_states);
#else
    t2_therm(N_POSITION_FEATS + (int)literal_offsets[fid], N_POSITION_FEATS + (int)literal_offsets[fid + 1], val,
             ta_states);
#endif
}

INLINE_FN void type1a_fb(ull rng_key, uint* RESTRICT rng_counter, const FBOUND_T* RESTRICT Xe, int patch_idx_y,
                         int patch_idx_x, const NLITS_T* RESTRICT literal_offsets, TA_STATE_T* RESTRICT ta_states,
                         int lane) {
#if TYPE1A_FB
#if POSITION_LITERALS
    if (lane == 0) {
        t1a_therm(rng_key, rng_counter, 0, N_POSITION_FEATS_Y, patch_idx_y, ta_states);
        t1a_therm(rng_key, rng_counter, N_POSITION_FEATS_Y, N_POSITION_FEATS, patch_idx_x, ta_states);
    }
#endif

    for (int fid = lane; fid < N_RAW_PATCH_FEATS; fid += LANE_COUNT)
        t1a_feature(rng_key, rng_counter, literal_offsets, fid,
                    (int)get_feature_value(Xe, patch_idx_y, patch_idx_x, fid), ta_states);
#endif
}

INLINE_FN void type2_fb(const FBOUND_T* RESTRICT Xe, int patch_idx_y, int patch_idx_x,
                        const NLITS_T* RESTRICT literal_offsets, TA_STATE_T* RESTRICT ta_states, int lane) {
#if TYPE2_FB
#if POSITION_LITERALS
    if (lane == 0) {
        t2_therm(0, N_POSITION_FEATS_Y, patch_idx_y, ta_states);
        t2_therm(N_POSITION_FEATS_Y, N_POSITION_FEATS, patch_idx_x, ta_states);
    }
#endif

    for (int fid = lane; fid < N_RAW_PATCH_FEATS; fid += LANE_COUNT)
        t2_feature(literal_offsets, fid, (int)get_feature_value(Xe, patch_idx_y, patch_idx_x, fid), ta_states);

#endif
}

INLINE_FN void type1b_fb(ull rng_key, uint* RESTRICT rng_counter, TA_STATE_T* RESTRICT ta_states, int lane) {
#if TYPE1B_FB
    float suc = geom_sample(rng_key, rng_counter, S_INV) - 1.0f;
    while (suc * (float)LANE_COUNT + (float)lane < (float)N_LITERALS) {
        safe_lit_dec((int)suc * LANE_COUNT + lane, ta_states);
        suc += geom_sample(rng_key, rng_counter, S_INV);
    }
#endif
}

INLINE_FN void apply_feedback(ull rng_key, uint* RESTRICT rng_counter, uint8_t fb, const FBOUND_T* RESTRICT Xe,
                              int patch_idx_y, int patch_idx_x, const NLITS_T* RESTRICT literal_offsets,
                              TA_STATE_T* RESTRICT ta_states, int lane) {
    if (fb == FB_T1A) {
        type1a_fb(rng_key, rng_counter, Xe, patch_idx_y, patch_idx_x, literal_offsets, ta_states, lane);
    } else if (fb == FB_T1B) {
        type1b_fb(rng_key, rng_counter, ta_states, lane);
    } else {
        type2_fb(Xe, patch_idx_y, patch_idx_x, literal_offsets, ta_states, lane);
    }
}
