// Platform layer for the C build. Concatenated after the generated header, before common.h.
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

#if _OPENMP
#include <omp.h>
#define GET_THREAD_ID omp_get_thread_num()
void set_num_threads(int n) { omp_set_num_threads(n); }
#else
#define GET_THREAD_ID 0
void set_num_threads(int n) {}
#endif
