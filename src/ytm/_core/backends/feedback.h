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

/*
 * Append a clause to feedback list along with count.
 *
 *   n1b > 0       -> (clause, n1b)      on the T1b list
 *   n1a + n2 > 0  -> (clause, n1a, n2)  on the T1a/T2 list
 * A list is a length counter plus parallel arrays: ids in *_cid, counts in *_cnt at the same index.
 * The final list is unordered.
 *
 * Arguments:
 *   clause      clause id
 *   n1a         number of classes that gave T1a
 *   n1b         number of classes that gave T1b
 *   n2          number of classes that gave T2
 *   t1b_n       T1b list length, incremented atomically
 *   t1b_cid     T1b list clause ids
 *   t1b_cnt     T1b count of each entry
 *   t1a_t2_n    T1a/T2 list length, incremented atomically
 *   t1a_t2_cid  T1a/T2 list clause ids
 *   t1a_cnt     T1a count of each entry
 *   t2_cnt      T2 count of each entry
 */
INLINE_FN void append_fb_lists(ull clause, int n1a, int n1b, int n2, uint* RESTRICT t1b_n, uint* RESTRICT t1b_cid,
                               NCLASSES_T* RESTRICT t1b_cnt, uint* RESTRICT t1a_t2_n, uint* RESTRICT t1a_t2_cid,
                               NCLASSES_T* RESTRICT t1a_cnt, NCLASSES_T* RESTRICT t2_cnt) {
    if (!TYPE1A_FB)
        n1a = 0;
    if (!TYPE1B_FB)
        n1b = 0;
    if (!TYPE2_FB)
        n2 = 0;

    if (n1b > 0)
        t1b_cnt[list_push(t1b_n, t1b_cid, clause)] = (NCLASSES_T)n1b;

    if (n1a + n2 > 0) {
        uint i = list_push(t1a_t2_n, t1a_t2_cid, clause);
        t1a_cnt[i] = (NCLASSES_T)n1a;
        t2_cnt[i] = (NCLASSES_T)n2;
    }
}


// Increment the TA state `ta` by `n`, saturating at `cap`. Returns true when a literal crosses the include line.
INLINE_FN bool safe_lit_inc(TA_STATE_T* ta, int n, int cap) {
    if (n == 0)
        return false;
    TA_STATE_T old = *ta;
    int v = (int)old + n;
    *ta = (TA_STATE_T)(v > cap ? cap : v);
    return is_included(old) != is_included(*ta);
}

// Decrement the TA state `ta` by `n`, saturating at 0. Returns true when a literal crosses the include line.
INLINE_FN bool safe_lit_dec(TA_STATE_T* ta, int n) {
    TA_STATE_T old = *ta;
    int v = (int)old - n;
    *ta = (TA_STATE_T)(v < 0 ? 0 : v);
    return is_included(old) != is_included(*ta);
}

// Increment literals in [start, end). Returns true if a literal crosses the include line.
INLINE_FN bool inc_range(int start, int end, int offset, int n, int cap, TA_STATE_T* RESTRICT ta_states) {
    bool crossed = false;
    for (int li = start; li < end; ++li)
        crossed |= safe_lit_inc(&ta_states[li + offset], n, cap);
    return crossed;
}

// Decrement literals in range [start, end), each with prob dec.p_any using geometric sampling
INLINE_FN bool dec_range_geom(ull rng_key, uint* RESTRICT rng_counter, binom_t dec, int start, int end, int offset,
                              int n_lanes, TA_STATE_T* RESTRICT ta_states) {
    bool crossed = false;
    float base = (float)start - 1.0f;
    while (base < (float)end) {
        float gap = geom_sample(rng_key, rng_counter, dec.p_any);
        float li = base + (n_lanes > 1 ? lane_scan_add(gap) : gap);
        if (li < (float)end)
            crossed |= safe_lit_dec(&ta_states[(int)li + offset], binom_at_least_one(rng_key, rng_counter, dec));
        base = n_lanes > 1 ? lane_last(li) : li;
    }
    return crossed;
}

// Dense decrement literals in range [start, end), each with prob dec.p_any
INLINE_FN bool dec_range_dense(ull rng_key, uint* RESTRICT rng_counter, binom_t dec, int start, int end, int offset,
                               int n_lanes, TA_STATE_T* RESTRICT ta_states) {
    const int lane = n_lanes > 1 ? lane_id() : 0;
    bool crossed = false;
    for (int li = start + lane; li < end; li += n_lanes) {
        crossed |= safe_lit_dec(&ta_states[li + offset], binomial(rng_key, rng_counter, dec));
    }
    return crossed;
}

/*
 * Decrement literals in [start, end). Switches between geometric samling and dense, depending on the probavility of
 * decrement. Returns true if a literal crosses the include line.
 */
INLINE_FN bool dec_range(ull rng_key, uint* RESTRICT rng_counter, binom_t dec, int start, int end, int offset,
                         int n_lanes, TA_STATE_T* RESTRICT ta_states) {
    if (start >= end || dec.p_any <= 0.0f)
        return false;
    if (end - start == 1 || dec.p_any >= P_DENSE_MIN)
        return dec_range_dense(rng_key, rng_counter, dec, start, end, offset, n_lanes, ta_states);
    return dec_range_geom(rng_key, rng_counter, dec, start, end, offset, n_lanes, ta_states);
}

INLINE_FN bool inc_range_fails(ull rng_key, uint* RESTRICT rng_counter, binom_t fails, int n, int start, int end,
                               int offset, int cap, TA_STATE_T* RESTRICT ta_states) {
    if (start >= end || fails.p_any <= 0.0f)
        return inc_range(start, end, offset, n, cap, ta_states);

    bool crossed = false;
    if (end - start == 1 || fails.p_any >= P_DENSE_MIN) {
        for (int li = start; li < end; ++li) {
            crossed |= safe_lit_inc(&ta_states[li + offset], n - binomial(rng_key, rng_counter, fails), cap);
        }
        return crossed;
    }

    float base = (float)start - 1.0f;
    while (base < (float)end) {
        float li = base + geom_sample(rng_key, rng_counter, fails.p_any);
        int stop = li < (float)end ? (int)li : end;
        crossed |= inc_range((int)base + 1, stop, offset, n, cap, ta_states);
        if (li < (float)end)
            crossed |= safe_lit_inc(&ta_states[(int)li + offset], n - binom_at_least_one(rng_key, rng_counter, fails),
                                    cap);
        base = li;
    }
    return crossed;
}

// Stores the counts and distribution for t1a and t2 feedbacks.
typedef struct {
    int t1a_cnt;           // number of classes that gave t1a.
    int t2_cnt;            // number of classes that gave t2.
    binom_t t1a_inc_fails; // T1a increment failures = Binomial(t1a_cnt, 1/s). When boosting, always 0.
    binom_t t1a_dec;       // T1a dec = Binomial(t1a_cnt, 1/s). When BOOST_TP_DEC, always t1a_cnt.
} t1a_t2_t;

// Create the t1a_t2_t struct which stores the counts and distributions needed for the updates.
INLINE_FN t1a_t2_t t1a_t2_params(int t1a_cnt, int t2_cnt) {
    t1a_t2_t fb;
    fb.t1a_cnt = t1a_cnt;
    fb.t2_cnt = t2_cnt;
    fb.t1a_inc_fails = binom_params(t1a_cnt, BOOST_TP_INC ? 0.0f : S_INV);
    fb.t1a_dec = binom_params(t1a_cnt, BOOST_TP_DEC ? 1.0f : S_INV);
    return fb;
}

// Apply aggregated T1a and T2 feedbacks.
INLINE_FN bool apply_t1a_t2_fb(ull rng_key, uint* RESTRICT rng_counter, t1a_t2_t fb, int start, int end, int offset,
                               bool lit_val, TA_STATE_T* RESTRICT ta_states) {
    bool crossed;
    if (lit_val) {
        crossed = inc_range_fails(rng_key, rng_counter, fb.t1a_inc_fails, fb.t1a_cnt, start, end, offset, MAX_TA_STATE,
                                  ta_states);
    } else {
        crossed = dec_range(rng_key, rng_counter, fb.t1a_dec, start, end, offset, 1, ta_states);
        crossed |= inc_range(start, end, offset, fb.t2_cnt, INCLUDE_STATE, ta_states);
    }
    return crossed;
}

/*
 * Apply t1a and t2 feedbacks to one thermometer features' literals in range [start, end).
 * Returns true when a literal crosses the include line.
 */
INLINE_FN bool apply_t1a_t2_therm_feat(ull rng_key, uint* RESTRICT rng_counter, t1a_t2_t fb, int start, int end,
                                       int val, TA_STATE_T* RESTRICT ta_states) {
    int split = start + val;
    bool crossed = apply_t1a_t2_fb(rng_key, rng_counter, fb, start, split, 0, true, ta_states);
    crossed |= apply_t1a_t2_fb(rng_key, rng_counter, fb, split, end, 0, false, ta_states);
    if (NEGATED_LITERALS) {
        crossed |= apply_t1a_t2_fb(rng_key, rng_counter, fb, split, end, N_LITERALS / 2, true, ta_states);
        crossed |= apply_t1a_t2_fb(rng_key, rng_counter, fb, start, split, N_LITERALS / 2, false, ta_states);
    }
    return crossed;
}

INLINE_FN bool t1a_t2_binary_lit(ull rng_key, uint* RESTRICT rng_counter, t1a_t2_t fb, TA_STATE_T* RESTRICT ta,
                                 bool val) {
    if (val) {
        int inc = fb.t1a_cnt;
        if (!BOOST_TP_INC)
            inc -= binomial(rng_key, rng_counter, fb.t1a_inc_fails);
        return safe_lit_inc(ta, inc, MAX_TA_STATE);
    }
    int dec = BOOST_TP_DEC ? fb.t1a_cnt : binomial(rng_key, rng_counter, fb.t1a_dec);
    bool crossed = safe_lit_dec(ta, dec);
    return crossed | safe_lit_inc(ta, fb.t2_cnt, INCLUDE_STATE);
}

/*
 * Apply t1a and t2 feedbacks to multiple binary features.
 * Returns true when a literal crosses the include line.
 */
INLINE_FN bool apply_t1a_t2_binary_feats(ull rng_key, uint* RESTRICT rng_counter, t1a_t2_t fb,
                                         const FBOUND_T* RESTRICT Xe, int patch_idx_y, int patch_idx_x,
                                         TA_STATE_T* RESTRICT ta_states, int lane, int fid_start, int fid_end) {
    const int neg = N_LITERALS / 2;
    bool crossed = false;

    for (int fid = fid_start + lane; fid < fid_end; fid += LANE_COUNT) {
        int li = N_POSITION_FEATS + fid;
        bool val = get_feature_value(Xe, patch_idx_y, patch_idx_x, fid);
        crossed |= t1a_t2_binary_lit(rng_key, rng_counter, fb, &ta_states[li], val);
        if (NEGATED_LITERALS)
            crossed |= t1a_t2_binary_lit(rng_key, rng_counter, fb, &ta_states[li + neg], !val);
    }
    return crossed;
}

/*
 * Apply one clause's T1a and T2 feedback, on one tile of its patch features.
 *
 * Arguments:
 *   rng_key, rng_counter      RNG stream of this clause, tile and lane
 *   t1a_cnt, t2_cnt           number of classes that gave T1a and T2
 *   Xe                        features of the current sample
 *   patch_idx_y, patch_idx_x  patch the clause fired on
 *   literal_offsets           first literal of each patch feature
 *   ta_states                 TA states of the clause
 *   lane                      this lane, 0 on CPU
 *   fid_start, fid_end        patch features of this tile
 *   with_position             this tile also handles the position literals
 *
 * All lanes must call it together. Returns true when any literal crossed the include line.
 */
INLINE_FN bool t1a_t2_fb(ull rng_key, uint* RESTRICT rng_counter, int t1a_cnt, int t2_cnt, const FBOUND_T* RESTRICT Xe,
                         int patch_idx_y, int patch_idx_x, const NLITS_T* RESTRICT literal_offsets,
                         TA_STATE_T* RESTRICT ta_states, int lane, int fid_start, int fid_end, bool with_position) {
    if (t1a_cnt == 0 && t2_cnt == 0)
        return false;
    t1a_t2_t fb = t1a_t2_params(t1a_cnt, t2_cnt);
    bool crossed = false;

    if (POSITION_LITERALS && with_position && lane == 0) {
        crossed |= apply_t1a_t2_therm_feat(rng_key, rng_counter, fb, 0, N_POSITION_FEATS_Y, patch_idx_y, ta_states);
        crossed |= apply_t1a_t2_therm_feat(rng_key, rng_counter, fb, N_POSITION_FEATS_Y, N_POSITION_FEATS, patch_idx_x,
                                           ta_states);
    }

    if (ALL_BINARY_FEATS) {
        crossed |= apply_t1a_t2_binary_feats(rng_key, rng_counter, fb, Xe, patch_idx_y, patch_idx_x, ta_states, lane,
                                             fid_start, fid_end);
    } else {
        for (int fid = fid_start + lane; fid < fid_end; fid += LANE_COUNT) {
            int val = (int)get_feature_value(Xe, patch_idx_y, patch_idx_x, fid);
            int start = N_POSITION_FEATS + (int)literal_offsets[fid];
            int end = N_POSITION_FEATS + (int)literal_offsets[fid + 1];
            crossed |= apply_t1a_t2_therm_feat(rng_key, rng_counter, fb, start, end, val, ta_states);
        }
    }
    return crossed;
}
