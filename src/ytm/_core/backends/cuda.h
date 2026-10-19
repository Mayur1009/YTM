#pragma once

#define WARP_SIZE 32

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>
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

#define WARP_STRIDE_LOOP(var, count) for (ull var = warp_id; var < (ull)(count); var += total_warps)

#define GRID_STRIDE_LOOP(var, count)                                                                                   \
    for (ull var = blockIdx.x * (ull)blockDim.x + threadIdx.x; var < (ull)(count); var += (ull)blockDim.x * gridDim.x)

struct WarpGrid {
    warp_t warp;
    int lane;
    ull warp_id;
    ull total_warps;
};

__device__ inline WarpGrid warp_grid() {
    auto warp = cg::tiled_partition<WARP_SIZE>(cg::this_thread_block());
    auto grid = cg::this_grid();
    return {warp, (int)warp.thread_rank(), grid.thread_rank() / warp.size(), grid.size() / warp.size()};
}

// Lane primitives for warp cooperative loops. All 32 lanes must call them together.
INLINE_FN float lane_scan_add(float v) {
    auto warp = cg::tiled_partition<WARP_SIZE>(cg::this_thread_block());
    return cg::inclusive_scan(warp, v);
}

INLINE_FN float lane_last(float v) { return __shfl_sync(0xffffffffu, v, WARP_SIZE - 1); }

INLINE_FN int lane_id() { return (int)(threadIdx.x & (WARP_SIZE - 1)); }

INLINE_FN uint list_push(uint* RESTRICT n, uint* RESTRICT ids, ull id) {
    auto g = cg::coalesced_threads();
    uint base;
    if (g.thread_rank() == 0)
        base = atomicAdd(n, g.size());
    base = g.shfl(base, 0);
    uint i = base + g.thread_rank();
    ids[i] = (uint)id;
    return i;
}
