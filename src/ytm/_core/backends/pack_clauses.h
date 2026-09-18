#ifdef IS_NEOVIM_CLANGD_ENV
#include "common.h"
#endif

#pragma once

typedef struct {
    int pos0, pos1, pos2, pos3;
    uint includes;
    bool valid;
} PositionResult;

typedef struct {
    int n_bounded_feats;
    uint includes;
    bool all_valid;
} FeatureResult;

typedef struct {
    FBOUND_T lb, ub;
    uint includes;
    bool is_bounded;
} FeatScan;

// One position literal and its negation, narrowing [lo, hi] inward.
INLINE_FN void update_pos_bounds(const TA_STATE_T* ta_state, int lit, int lit_off, int* lo, int* hi, uint* includes) {
    if (is_included(ta_state[lit_off + lit])) {
        if (lit + 1 > *lo)
            *lo = lit + 1;
        (*includes)++;
    }
#if NEGATED_LITERALS
    if (is_included(ta_state[lit_off + lit + N_LITERALS / 2])) {
        if (lit < *hi)
            *hi = lit;
        (*includes)++;
    }
#endif
}

// Every thermometer literal of one raw patch feature, collapsed to the interval [lb, ub].
INLINE_FN FeatScan calc_single_feat_bounds(const TA_STATE_T* ta_state, const FBOUND_T* therm_bits,
                                           const NLITS_T* literal_offsets, int fid) {
    FeatScan r;

#if ALL_BINARY_FEATS
    // One bit per feature.
    int lstart = N_POSITION_FEATS + fid;

    bool inc = is_included(ta_state[lstart]);
    r.lb = inc;
    r.includes = inc;
#if NEGATED_LITERALS
    bool neg = is_included(ta_state[lstart + N_LITERALS / 2]);
    r.ub = !neg;
    r.includes += neg;
#else
    r.ub = 1;
#endif
    r.is_bounded = (r.includes > 0);

#else

    FBOUND_T n_bits = therm_bits[fid];
    int lstart = N_POSITION_FEATS + literal_offsets[fid];
    r.lb = 0;
    r.ub = n_bits;
    r.includes = 0;
    r.is_bounded = false;

    for (FBOUND_T bit = 0; bit < n_bits; ++bit) {
        if (is_included(ta_state[lstart + bit])) {
            if (bit + 1 > r.lb)
                r.lb = bit + 1;
            r.includes++;
            r.is_bounded = true;
        }
#if NEGATED_LITERALS
        if (is_included(ta_state[lstart + bit + N_LITERALS / 2])) {
            if (bit < r.ub)
                r.ub = bit;
            r.includes++;
            r.is_bounded = true;
        }
#endif
    }
#endif
    return r;
}
