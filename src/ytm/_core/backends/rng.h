#ifdef IS_NEOVIM_CLANGD_ENV
#include "common.h"
#endif

// Counter based RNG, shared by the C and CUDA builds.
//
// A key is derived once per work item from the seed and a few identifiers, then drawn from with a
// counter. Nothing is stored between calls, so the same key and counter always give the same value
// and threads never share state.

#pragma once

INLINE_FN ull mix64(ull x) {
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

INLINE_FN ull hash_combine(ull a, ull b) { return mix64(a ^ mix64(b + 0x9e3779b97f4a7c15ULL)); }
INLINE_FN ull rng_hash(ull seed, ull a, ull b) { return hash_combine(hash_combine(seed, a), b); }

INLINE_FN float rand_uniform(ull key, uint* counter) {
    ull x = key ^ (ull)((*counter)++);
    x = mix64(x);
    return (float)(x >> 40) * 0x1p-24f;
}

// Geometric sampling for getting the number of trials after which there will be success.
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
