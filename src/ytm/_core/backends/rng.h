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

INLINE_FN ull rng_hash(ull seed, ull a, ull b, ull c) {
    ull k = hash_combine(seed, a);
    k = hash_combine(k, b);
    k = hash_combine(k, c);
    return k;
}

INLINE_FN float rand_uniform(ull key, uint* counter) {
    ull x = key ^ (ull)((*counter)++);
    x = mix64(x);
    return (float)(x >> 32) * 0x1p-32f;
}

// Geometric sampling for getting the next trail number which will result in success, when all the trials have a
// propability p.
INLINE_FN int geom_sample(ull rng_key, uint* rng_counter, float p) {
    float u = rand_uniform(rng_key, rng_counter);
    double u_clamp = clip(u, 1e-7f, 1.0f - 1e-7f);
    double log_u = log1p(-u_clamp);
    double log_p = log1p(-p);
    int sample = (int)(log_u / log_p) + 1;
    return sample;
}
