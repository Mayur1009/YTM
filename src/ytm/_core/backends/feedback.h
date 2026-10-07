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

// Adds a clause to the feedback lists with its counts over classes. T1b goes to its own list, T1a and T2
// share one, since they are applied together.
INLINE_FN void fb_lists_append(ull clause, int n1a, int n1b, int n2, uint* RESTRICT t1b_n, uint* RESTRICT t1b_cid,
                               NCLASSES_T* RESTRICT t1b_cnt, uint* RESTRICT t1a_t2_n, uint* RESTRICT t1a_t2_cid,
                               NCLASSES_T* RESTRICT t1a_cnt, NCLASSES_T* RESTRICT t2_cnt) {
    if (n1b > 0)
        t1b_cnt[fb_list_append(t1b_n, t1b_cid, clause)] = (NCLASSES_T)n1b;
    if (n1a + n2 > 0) {
        uint i = fb_list_append(t1a_t2_n, t1a_t2_cid, clause);
        t1a_cnt[i] = (NCLASSES_T)n1a;
        t2_cnt[i] = (NCLASSES_T)n2;
    }
}

INLINE_FN void safe_lit_inc(int li, TA_STATE_T* ta_states) {
    if (ta_states[li] < MAX_TA_STATE)
        ta_states[li] += 1;
}

INLINE_FN void safe_lit_dec(int li, TA_STATE_T* ta_states) {
    if (ta_states[li] > 0)
        ta_states[li] -= 1;
}

// Returns true when the literal crossed the include line, the only change that needs the clause repacked.
INLINE_FN bool safe_lit_dec_by(int li, int n, TA_STATE_T* ta_states) {
    TA_STATE_T d = (TA_STATE_T)n;
    TA_STATE_T old = ta_states[li];
    ta_states[li] = (old > d) ? old - d : 0;
    return is_included(old) != is_included(ta_states[li]);
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

// Above this hit probability a uniform draw per literal is cheaper than geometric skipping, which pays a log per
// hit. Measured on GPU: the warp cooperative geometric path stays as fast up to about 0.35 to 0.5, since its log is
// shared by 32 hits per round.
#ifndef P_DENSE_MIN
#define P_DENSE_MIN 0.5f
#endif

// Dense twin of the geometric sampling: every literal in [start, end) is hit with dec.p_any, decided by one
// uniform draw, and a hit literal loses binom_at_least_one(dec) states. `n_lanes` lanes starting at `lane` stride
// over the literals. Returns true when any literal crossed the include line.
INLINE_FN bool bits_dec_range(ull rng_key, uint* RESTRICT rng_counter, binom_t dec, int start, int end, int offset,
                              int lane, int n_lanes, TA_STATE_T* RESTRICT ta_states) {
    bool crossed = false;
    for (int li = start + lane; li < end; li += n_lanes) {
        if (rand_uniform(rng_key, rng_counter) < dec.p_any)
            crossed |= safe_lit_dec_by(li + offset, binom_at_least_one(rng_key, rng_counter, dec), ta_states);
    }
    return crossed;
}

// Warp cooperative geometric sampling over [start, end): each lane draws one gap per round and an
// inclusive scan turns the gaps into consecutive hit positions. All lanes must call it together.
// Merges dec.n_trials feedbacks that each decrement a literal with dec.p_success: skip straight to
// literals hit at least once, then decrement by how many of the feedbacks hit it.
INLINE_FN bool geom_dec_range(ull rng_key, uint* RESTRICT rng_counter, binom_t dec, int start, int end,
                              TA_STATE_T* RESTRICT ta_states) {
    if (dec.p_any >= P_DENSE_MIN)
        return bits_dec_range(rng_key, rng_counter, dec, start, end, 0, lane_id(), LANE_COUNT, ta_states);

    bool crossed = false;
    float base = (float)start - 1.0f;
    while (base < (float)end) {
        float li = base + lane_scan_add(geom_sample(rng_key, rng_counter, dec.p_any));
        if (li < (float)end)
            crossed |= safe_lit_dec_by((int)li, binom_at_least_one(rng_key, rng_counter, dec), ta_states);
        base = lane_last(li);
    }
    return crossed;
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
    geom_dec_range(rng_key, rng_counter, binom_params(1, S_INV), 0, N_LITERALS, ta_states);
#endif
}

// Returns true when the literal crossed the include line.
INLINE_FN bool clamp_store(int li, int v, int cap, TA_STATE_T* RESTRICT ta_states) {
    TA_STATE_T old = ta_states[li];
    ta_states[li] = (TA_STATE_T)(v < 0 ? 0 : (v > cap ? cap : v));
    return is_included(old) != is_included(ta_states[li]);
}

// Merged T1a and T2 for a firing clause. Every literal in [start, end) gets `add`, minus a
// Binomial(dec.n_trials, dec.p_success) drawn sparsely, summed first and then clamped to [0, cap]. Returns true
// when any literal crossed the include line.
INLINE_FN bool merged_range_update(ull rng_key, uint* RESTRICT rng_counter, int add, binom_t dec, int start, int end,
                                   int offset, int cap, TA_STATE_T* RESTRICT ta_states) {
    if (start >= end)
        return false;

    // A single literal, binary features and short thermometers: one uniform draw instead of a geometric gap.
    if (end - start == 1) {
        int sub = (dec.n_trials > 0 && rand_uniform(rng_key, rng_counter) < dec.p_any)
                      ? binom_at_least_one(rng_key, rng_counter, dec)
                      : 0;
        if (add != 0 || sub != 0)
            return clamp_store(start + offset, (int)ta_states[start + offset] + add - sub, cap, ta_states);
        return false;
    }

    bool crossed = false;
    const bool dense = dec.n_trials > 0 && dec.p_any >= P_DENSE_MIN;

    // Nothing to add, only the hit literals change.
    if (add == 0) {
        if (dense)
            return bits_dec_range(rng_key, rng_counter, dec, start, end, offset, 0, 1, ta_states);
        float hit = (float)start - 1.0f + geom_sample(rng_key, rng_counter, dec.p_any);
        while (hit < (float)end) {
            crossed |= safe_lit_dec_by((int)hit + offset, binom_at_least_one(rng_key, rng_counter, dec), ta_states);
            hit += geom_sample(rng_key, rng_counter, dec.p_any);
        }
        return crossed;
    }

    // Every literal is visited anyway, so a dense hit probability is decided per literal with one uniform draw.
    if (dense) {
        for (int li = start; li < end; ++li) {
            int v = (int)ta_states[li + offset] + add;
            if (rand_uniform(rng_key, rng_counter) < dec.p_any)
                v -= binom_at_least_one(rng_key, rng_counter, dec);
            crossed |= clamp_store(li + offset, v, cap, ta_states);
        }
        return crossed;
    }

    float hit = (float)start - 1.0f + geom_sample(rng_key, rng_counter, dec.p_any);
    for (int li = start; li < end; ++li) {
        int v = (int)ta_states[li + offset] + add;
        if ((float)li == hit) {
            v -= binom_at_least_one(rng_key, rng_counter, dec);
            hit += geom_sample(rng_key, rng_counter, dec.p_any);
        }
        crossed |= clamp_store(li + offset, v, cap, ta_states);
    }
    return crossed;
}

// Literals that are 1: T1a increments, each with prob 1 - 1/s unless boosted. `misses` are the T1a
// trials that did not increment, sparse with prob 1/s.
INLINE_FN bool t1a_t2_ones(ull rng_key, uint* RESTRICT rng_counter, int t1a_cnt, binom_t misses, int start, int end,
                           int offset, TA_STATE_T* RESTRICT ta_states) {
    return merged_range_update(rng_key, rng_counter, t1a_cnt, misses, start, end, offset, MAX_TA_STATE, ta_states);
}

// Literals that are 0: T1a decrements with prob 1/s unless boosted, T2 increments up to the include line.
INLINE_FN bool t1a_t2_zeros(ull rng_key, uint* RESTRICT rng_counter, int add, binom_t decs, int t2_cnt, int start,
                            int end, int offset, TA_STATE_T* RESTRICT ta_states) {
    int cap = (t2_cnt > 0) ? INCLUDE_STATE : MAX_TA_STATE;
    return merged_range_update(rng_key, rng_counter, add, decs, start, end, offset, cap, ta_states);
}

// The merged counts of one clause, with the per clause sampling parameters computed once.
typedef struct {
    int t1a_cnt;
    int t2_cnt;
    int zeros_add;   // net deterministic change on literals that are 0
    binom_t misses;  // sparse T1a misses on literals that are 1
    binom_t decs;    // sparse T1a decrements on literals that are 0
} t1a_t2_t;

INLINE_FN t1a_t2_t t1a_t2_params(int t1a_cnt, int t2_cnt) {
    t1a_t2_t m;
    m.t1a_cnt = t1a_cnt;
    m.t2_cnt = t2_cnt;
    binom_t none = binom_params(0, 0.0f);
    binom_t sparse = binom_params(t1a_cnt, S_INV);
#if BOOST_TP_INC
    m.misses = none;
#else
    m.misses = sparse;
#endif
#if BOOST_TP_DEC
    m.zeros_add = t2_cnt - t1a_cnt;
    m.decs = none;
#else
    m.zeros_add = t2_cnt;
    m.decs = sparse;
#endif
    return m;
}

// One thermometer feature, literals [start, start + val) are 1 and [start + val, end) are 0. Negated
// literals are the opposite. A binary feature is the width 1 case.
INLINE_FN bool t1a_t2_range(ull rng_key, uint* RESTRICT rng_counter, t1a_t2_t m, int start, int end, int val,
                            TA_STATE_T* RESTRICT ta_states) {
    int split = start + val;
    bool crossed = t1a_t2_ones(rng_key, rng_counter, m.t1a_cnt, m.misses, start, split, 0, ta_states);
    crossed |= t1a_t2_zeros(rng_key, rng_counter, m.zeros_add, m.decs, m.t2_cnt, split, end, 0, ta_states);
#if NEGATED_LITERALS
    crossed |= t1a_t2_ones(rng_key, rng_counter, m.t1a_cnt, m.misses, split, end, N_LITERALS / 2, ta_states);
    crossed |= t1a_t2_zeros(rng_key, rng_counter, m.zeros_add, m.decs, m.t2_cnt, start, split, N_LITERALS / 2, ta_states);
#endif
    return crossed;
}

#if ALL_BINARY_FEATS
// One literal of a binary feature: `add`, minus a merged count when its uniform draw hits, then clamped to [0, cap].
// Returns true when the literal crossed the include line.
INLINE_FN bool binary_lit_update(int li, int add, binom_t dec, int cap, ull rng_key, uint* RESTRICT rng_counter,
                                 TA_STATE_T* RESTRICT ta_states) {
    int sub = 0;
    if (dec.n_trials > 0 && rand_uniform(rng_key, rng_counter) < dec.p_any)
        sub = binom_at_least_one(rng_key, rng_counter, dec);
    if (add == 0 && sub == 0)
        return false;
    return clamp_store(li, (int)ta_states[li] + add - sub, cap, ta_states);
}

// T1a and T2 on binary patch features [fid_start, fid_end). A feature is one literal and its negation, so each
// literal needs at most one decision, one uniform draw instead of a geometric gap. Lanes
// keep striding over features so neighbouring lanes touch neighbouring literals.
INLINE_FN bool t1a_t2_binary_feats(ull rng_key, uint* RESTRICT rng_counter, t1a_t2_t m, const FBOUND_T* RESTRICT Xe,
                                   int patch_idx_y, int patch_idx_x, TA_STATE_T* RESTRICT ta_states, int lane,
                                   int fid_start, int fid_end) {
    const int zero_cap = (m.t2_cnt > 0) ? INCLUDE_STATE : MAX_TA_STATE;
    bool crossed = false;

    for (int fid = fid_start + lane; fid < fid_end; fid += LANE_COUNT) {
        int li = N_POSITION_FEATS + fid;
        bool val = get_feature_value(Xe, patch_idx_y, patch_idx_x, fid) != 0;
        // Literal li is 1 when the feature is, its negation the opposite.
        int li_one = val ? li : li + N_LITERALS / 2;
        int li_zero = val ? li + N_LITERALS / 2 : li;
#if NEGATED_LITERALS
        crossed |= binary_lit_update(li_one, m.t1a_cnt, m.misses, MAX_TA_STATE, rng_key, rng_counter, ta_states);
        crossed |= binary_lit_update(li_zero, m.zeros_add, m.decs, zero_cap, rng_key, rng_counter, ta_states);
#else
        if (val)
            crossed |= binary_lit_update(li, m.t1a_cnt, m.misses, MAX_TA_STATE, rng_key, rng_counter, ta_states);
        else
            crossed |= binary_lit_update(li, m.zeros_add, m.decs, zero_cap, rng_key, rng_counter, ta_states);
#endif
    }
    return crossed;
}
#endif

// First patch feature whose literals start at or after patch literal x, so tiles cut between features.
INLINE_FN int first_feat_from(const NLITS_T* RESTRICT literal_offsets, int x) {
#if ALL_BINARY_FEATS
    return x < N_RAW_PATCH_FEATS ? x : N_RAW_PATCH_FEATS;
#else
    int lo = 0, hi = N_RAW_PATCH_FEATS;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if ((int)literal_offsets[mid] < x)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo;
#endif
}

// T1a and T2 on patch features [fid_start, fid_end), whole features only, lanes striding over them. The
// position literals go with the tile that asks for them. Returns true when any literal crossed the include line.
INLINE_FN bool t1a_t2_fb(ull rng_key, uint* RESTRICT rng_counter, int t1a_cnt, int t2_cnt, const FBOUND_T* RESTRICT Xe,
                         int patch_idx_y, int patch_idx_x, const NLITS_T* RESTRICT literal_offsets,
                         TA_STATE_T* RESTRICT ta_states, int lane, int fid_start, int fid_end, bool with_position) {
#if !TYPE1A_FB
    t1a_cnt = 0;
#endif
#if !TYPE2_FB
    t2_cnt = 0;
#endif
    if (t1a_cnt == 0 && t2_cnt == 0)
        return false;
    t1a_t2_t m = t1a_t2_params(t1a_cnt, t2_cnt);
    bool crossed = false;

#if POSITION_LITERALS
    if (with_position && lane == 0) {
        crossed |= t1a_t2_range(rng_key, rng_counter, m, 0, N_POSITION_FEATS_Y, patch_idx_y, ta_states);
        crossed |= t1a_t2_range(rng_key, rng_counter, m, N_POSITION_FEATS_Y, N_POSITION_FEATS, patch_idx_x, ta_states);
    }
#endif

#if ALL_BINARY_FEATS
    crossed |= t1a_t2_binary_feats(rng_key, rng_counter, m, Xe, patch_idx_y, patch_idx_x, ta_states, lane, fid_start,
                                   fid_end);
#else
    for (int fid = fid_start + lane; fid < fid_end; fid += LANE_COUNT) {
        int val = (int)get_feature_value(Xe, patch_idx_y, patch_idx_x, fid);
        int start = N_POSITION_FEATS + (int)literal_offsets[fid];
        int end = N_POSITION_FEATS + (int)literal_offsets[fid + 1];
        crossed |= t1a_t2_range(rng_key, rng_counter, m, start, end, val, ta_states);
    }
#endif
    return crossed;
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
