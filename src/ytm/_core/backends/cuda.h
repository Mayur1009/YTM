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

INLINE_FN int warp_compact_slot(const warp_t& warp, bool keep, int* offset) {
    uint mask = warp.ballot(keep);
    int slot = *offset + __popc(mask & ((1u << warp.thread_rank()) - 1));
    *offset += __popc(mask);
    return slot;
}

