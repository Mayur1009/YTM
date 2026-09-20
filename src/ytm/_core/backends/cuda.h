#pragma once

#define WARP_SIZE 32

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;
using warp_t = cg::thread_block_tile<WARP_SIZE>;

typedef unsigned int uint;
typedef unsigned long long ull;
typedef signed char int8_t;
typedef unsigned char uint8_t;

#ifndef INFINITY
#define INFINITY __int_as_float(0x7f800000)
#endif

#define INLINE_FN __device__ inline
#define RESTRICT __restrict__
#define LANE_COUNT WARP_SIZE
