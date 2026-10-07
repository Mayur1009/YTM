// Stateless Splitmix RNG

#pragma once
#ifdef IS_NEOVIM_CLANGD_ENV
#include "common.h"
#endif

INLINE_FN ull mix64(ull x) {
    // SplitMix64 finalizer: scrambles all 64 bits, so nearby inputs give unrelated outputs.
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

// Mix b into a.
INLINE_FN ull hash_combine(ull a, ull b) { return mix64(a ^ mix64(b + 0x9e3779b97f4a7c15ULL)); }

// Mix 3 values to get a hash.
INLINE_FN ull rng_hash(ull a, ull b, ull c) { return hash_combine(hash_combine(a, b), c); }

// Random 64bit int. advances the counter after use.
INLINE_FN ull rand_u64(ull key, uint* counter) { return mix64(key ^ (ull)((*counter)++)); }

// Random uniform float.
INLINE_FN float rand_uniform(ull key, uint* counter) {
    ull x = rand_u64(key, counter);
    return (float)(x >> 40) * 0x1p-24f; // 24 bits are enough for float32 rands.
}

typedef struct {
    float r1, r2;
} uniform2_t;

// Make 2 uniform random numbers from the hash.
INLINE_FN uniform2_t rand_uniform2(ull key, uint* counter) {
    ull x = rand_u64(key, counter);
    uniform2_t n;
    n.r1 = (float)(x >> 40) * 0x1p-24f;                 // top 24 bits
    n.r2 = (float)((x >> 16) & 0xFFFFFFull) * 0x1p-24f; // next 24 bits
    return n;
}

// Geometric sampling. number of bernoulli(p) trials up to and including the first success.
INLINE_FN float geom_sample(ull rng_key, uint* rng_counter, float p) {
    if (p >= 1.0f)
        return 1.0f;
    if (!(p > 0.0f))
        return INFINITY;
    float u = rand_uniform(rng_key, rng_counter);
    float u_clamp = clip(u, 1e-7f, 1.0f - 1e-7f);
    float log_u = log1pf(-u_clamp);
    float log_p = log1pf(-p);
    return ceilf(log_u / log_p);
}

// Given n_trials with prob p_success, what is the probability of at least one success.
INLINE_FN float prob_at_least_one(int n_trials, float p_success) {
    if (n_trials <= 0)
        return 0.0f;
    if (n_trials == 1)
        return p_success;
    return 1.0f - powf(1.0f - p_success, (float)n_trials);
}

// n_trials each with p_success, plus what binom_at_least_one needs, computed once so sampling needs no powf.
typedef struct {
    int n_trials;
    float p_success;
    float p_any; // P(at least one success)
    float p_one; // P(exactly one success)
    float odds;  // p_success / (1 - p_success), steps P(exactly j) to P(exactly j+1)
} binom_t;

INLINE_FN binom_t binom_params(int n_trials, float p_success) {
    binom_t b;
    b.n_trials = n_trials;
    b.p_success = p_success;
    b.p_any = prob_at_least_one(n_trials, p_success);
    b.p_one = (n_trials > 1) ? (float)n_trials * p_success * powf(1.0f - p_success, (float)(n_trials - 1)) : b.p_any;
    b.odds = (p_success < 1.0f) ? p_success / (1.0f - p_success) : 0.0f;
    return b;
}

// Given n_trials each with p_success, and given at least one succeeded, return how many successes.
INLINE_FN int binom_at_least_one(ull key, uint* counter, binom_t b) {
    // Only one trial, and we condition on at least one success.
    // OR, if p_success = 0, at least one success is impossible, so fall back to the minimum.
    if (b.n_trials <= 1 || b.p_success <= 0.0f)
        return 1;

    // If p_success = 1, all trials succeed.
    if (b.p_success >= 1.0f)
        return b.n_trials;

    float target = rand_uniform(key, counter) * b.p_any;

    int n_success = 1;
    float p_exact = b.p_one; // P(exactly 1)
    float p_upto = p_exact;
    while (p_upto < target && n_success < b.n_trials) {
        // P(exactly n_success+1)
        p_exact *= ((float)(b.n_trials - n_success) / (float)(n_success + 1)) * b.odds;
        p_upto += p_exact;
        ++n_success;
    }
    return n_success;
}
