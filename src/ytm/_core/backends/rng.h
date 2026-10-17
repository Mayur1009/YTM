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

// Random 32-bit unsigned int in the range [0, high).
INLINE_FN uint rand_int(ull key, uint* counter, uint high) {
    return (uint)(((rand_u64(key, counter) >> 32) * high) >> 32);
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
