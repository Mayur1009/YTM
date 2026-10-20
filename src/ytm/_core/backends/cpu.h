#pragma once

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef unsigned int uint;
typedef unsigned long long ull;

#define INLINE_FN static inline
#define RESTRICT restrict
#define LANE_COUNT 1

typedef struct {
    float x, y;
} float2;

typedef struct {
    int x, y;
} int2;

INLINE_FN float2 make_float2(float x, float y) {
    float2 r = {x, y};
    return r;
}

INLINE_FN int2 make_int2(int x, int y) {
    int2 r = {x, y};
    return r;
}

static int ytm_n_threads = 1;
void set_num_threads(int n) { ytm_n_threads = n < 1 ? 1 : n; }

#if _OPENMP
#include <omp.h>
#define GET_THREAD_ID omp_get_thread_num()
#else
#define GET_THREAD_ID 0
#endif

// Lane primitives, a single lane on CPU so both are the identity.
INLINE_FN float lane_scan_add(float v) { return v; }

INLINE_FN float lane_last(float v) { return v; }

INLINE_FN int lane_id(void) { return 0; }

INLINE_FN uint list_push(uint* RESTRICT n, uint* RESTRICT ids, ull id) {
    uint i;
#pragma omp atomic capture
    i = (*n)++;
    ids[i] = (uint)id;
    return i;
}
