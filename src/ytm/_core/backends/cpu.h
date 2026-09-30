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

static int ytm_n_threads = 1;
void set_num_threads(int n) { ytm_n_threads = n < 1 ? 1 : n; }

#if _OPENMP
#include <omp.h>
#define GET_THREAD_ID omp_get_thread_num()
#else
#define GET_THREAD_ID 0
#endif

INLINE_FN void fb_list_append(uint* RESTRICT fb_count, uint* RESTRICT fb_ids, ull clause) {
    uint i;
#pragma omp atomic capture
    i = (*fb_count)++;
    fb_ids[i] = (uint)clause;
}
