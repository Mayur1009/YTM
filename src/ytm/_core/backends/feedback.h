#ifdef IS_NEOVIM_CLANGD_ENV
#include "common.h"
#include "cpu.h"
#include "rng.h"
#endif

#pragma once

INLINE_FN void inc_lits(int start, int end, int offset, TA_STATE_T* ta_states) {
    for (int li = start; li < end; ++li)
        if (ta_states[li + offset] < MAX_TA_STATE)
            ta_states[li + offset] += 1;
}

INLINE_FN void dec_lits(int start, int end, int offset, TA_STATE_T* ta_states) {
    for (int li = start; li < end; ++li)
        if (ta_states[li + offset] > 0)
            ta_states[li + offset] -= 1;
}

INLINE_FN void prob_inc_lits(ull rng_key, uint* rng_counter, float prob, int start, int end, int offset,
                             TA_STATE_T* ta_states) {
    float li = (float)start + geom_sample(rng_key, rng_counter, prob) - 1.0f;
    while (li < (float)end) {
        int idx = (int)li + offset;
        if (ta_states[idx] < MAX_TA_STATE)
            ta_states[idx] += 1;
        li += geom_sample(rng_key, rng_counter, prob);
    }
}

INLINE_FN void prob_dec_lits(ull rng_key, uint* rng_counter, float prob, int start, int end, int offset,
                             TA_STATE_T* ta_states) {
    float li = (float)start + geom_sample(rng_key, rng_counter, prob) - 1.0f;
    while (li < (float)end) {
        int idx = (int)li + offset;
        if (ta_states[idx] > 0)
            ta_states[idx] -= 1;
        li += geom_sample(rng_key, rng_counter, prob);
    }
}

INLINE_FN void t1a_incs(ull rng_key, uint* rng_counter, int start, int end, int offset, TA_STATE_T* ta_states) {
#if BOOST_TP_INC
    inc_lits(start, end, offset, ta_states);
#else
    prob_inc_lits(rng_key, rng_counter, 1.0f - S_INV, start, end, offset, ta_states);
#endif
}

INLINE_FN void t1a_decs(ull rng_key, uint* rng_counter, int start, int end, int offset, TA_STATE_T* ta_states) {
#if BOOST_TP_DEC
    dec_lits(start, end, offset, ta_states);
#else
    prob_dec_lits(rng_key, rng_counter, S_INV, start, end, offset, ta_states);
#endif
}
