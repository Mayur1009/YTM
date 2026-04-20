#ifdef IS_NEOVIM_CLANGD_ENV
#define TOTAL_CLAUSES 1000
#define CLASSES 10
#define HEIGHT 28
#define WIDTH 28
#define DEPTH 1
#define PATCH_HEIGHT 10
#define PATCH_WIDTH 10
#define STRIDE_Y 1
#define STRIDE_X 1
#define NEGATED_LITERALS 1
#define POSITION_LITERALS 1
#define COALESCED 0
#define INCLUDE_STATE 128
#define N_RAW_PATCH_FEATS 100
#define N_PATCH_FEATS 100
#define N_POSITION_FEATS 36
#define N_PATCHES_Y 19
#define N_PATCHES_X 19
#define N_PATCHES 361
#define N_LITERALS 272
#define WARP_SIZE 32
#endif

#define N_POSITION_FEATS_Y (N_PATCHES_Y - 1)
#define N_POSITION_FEATS_X (N_PATCHES_X - 1)
#define MAX_FIDS_PER_LANE ((N_RAW_PATCH_FEATS + WARP_SIZE - 1) / WARP_SIZE)

typedef signed char int8_t;
typedef unsigned long long ull;
typedef unsigned int uint;

extern "C" {
__device__ inline uint warp_reduce_sum(uint val) {
    for (int d = warpSize / 2; d > 0; d >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, d);
    return val;
}

__device__ inline void warp_reduce_bounds(int* vmax1, int* vmin1, int* vmax2, int* vmin2) {
    for (int d = warpSize / 2; d > 0; d >>= 1) {
        *vmax1 = max(*vmax1, __shfl_xor_sync(0xFFFFFFFF, *vmax1, d));
        *vmin1 = min(*vmin1, __shfl_xor_sync(0xFFFFFFFF, *vmin1, d));
        *vmax2 = max(*vmax2, __shfl_xor_sync(0xFFFFFFFF, *vmax2, d));
        *vmin2 = min(*vmin2, __shfl_xor_sync(0xFFFFFFFF, *vmin2, d));
    }
}

__device__ inline int warp_exclusive_prefix_sum(int val) {
    int lane = threadIdx.x % warpSize;
    for (int d = 1; d < warpSize; d <<= 1) {
        int n = __shfl_up_sync(0xFFFFFFFF, val, d);
        if (lane >= d)
            val += n;
    }
    int exclusive = __shfl_up_sync(0xFFFFFFFF, val, 1);
    return (lane == 0) ? 0 : exclusive;
}

__device__ inline bool is_included(uint ta_state) { return ta_state >= INCLUDE_STATE; }

struct PositionResult {
    int pos0, pos1, pos2, pos3;
    uint includes;
    bool valid;
};

__device__ inline PositionResult scan_position_literals(const uint* __restrict__ ta_state, int lane) {
#if POSITION_LITERALS
    int pos0 = 0, pos1 = N_PATCHES_Y - 1;
    int pos2 = 0, pos3 = N_PATCHES_X - 1;
    uint includes = 0;

    for (int lit = lane; lit < N_POSITION_FEATS_Y; lit += warpSize) {
        if (is_included(ta_state[lit])) {
            pos0 = max(pos0, lit + 1);
            includes++;
        }
#if NEGATED_LITERALS
        if (is_included(ta_state[lit + N_LITERALS / 2])) {
            pos1 = min(pos1, lit);
            includes++;
        }
#endif
    }

    for (int lit = lane; lit < N_POSITION_FEATS_X; lit += warpSize) {
        if (is_included(ta_state[N_POSITION_FEATS_Y + lit])) {
            pos2 = max(pos2, lit + 1);
            includes++;
        }
#if NEGATED_LITERALS
        if (is_included(ta_state[N_POSITION_FEATS_Y + lit + N_LITERALS / 2])) {
            pos3 = min(pos3, lit);
            includes++;
        }
#endif
    }

    // Warp reductions
    warp_reduce_bounds(&pos0, &pos1, &pos2, &pos3);
    return {pos0, pos1, pos2, pos3, includes, (pos0 <= pos1 && pos2 <= pos3)};
#else
    // Non-convolution
    return {0, N_PATCHES_Y - 1, 0, N_PATCHES_X - 1, 0, true};
#endif
}

struct FeatureResult {
    int n_constrained;
    uint includes;
    bool all_valid;
};

__device__ inline FeatureResult scan_feature_literals(const uint* __restrict__ ta_state,
                                                      const int* __restrict__ feat_mins,
                                                      const int* __restrict__ feat_maxs,
                                                      const int* __restrict__ literal_offsets, int* __restrict__ cfb,
                                                      int* __restrict__ cfids, int lane) {
    int my_cfids[MAX_FIDS_PER_LANE];
    int my_n_constrained = 0;
    uint my_includes = 0;
    bool my_valid = true;

    for (int fid = lane; fid < N_RAW_PATCH_FEATS; fid += warpSize) {
        int n_bits = literal_offsets[fid + 1] - literal_offsets[fid];
        int lstart = N_POSITION_FEATS + literal_offsets[fid];
        bool has_constraint = false;

        int b0 = feat_mins[fid];
        int b1 = feat_maxs[fid];

        for (int bit = 0; bit < n_bits; ++bit) {
            if (is_included(ta_state[lstart + bit])) {
                b0 = max(b0, feat_mins[fid] + bit + 1);
                my_includes++;
                has_constraint = true;
            }
#if NEGATED_LITERALS
            if (is_included(ta_state[lstart + bit + N_LITERALS / 2])) {
                b1 = min(b1, feat_mins[fid] + bit);
                my_includes++;
                has_constraint = true;
            }
#endif
        }

        if (b0 > b1)
            my_valid = false;
        if (has_constraint) {
            cfb[fid * 2 + 0] = b0;
            cfb[fid * 2 + 1] = b1;
            my_cfids[my_n_constrained++] = fid;
        }
    }

    // Compact constrained_fids via warp prefix sum
    int write_offset = warp_exclusive_prefix_sum(my_n_constrained);
    for (int i = 0; i < my_n_constrained; i++)
        cfids[write_offset + i] = my_cfids[i];

    uint total_nc = warp_reduce_sum((uint)my_n_constrained);
    bool all_valid = (bool)__all_sync(0xFFFFFFFF, my_valid);

    return {(int)total_nc, my_includes, all_valid};
}

__global__ void pack_clauses(const uint* __restrict__ global_ta_states, const int* __restrict__ feat_mins,
                             const int* __restrict__ feat_maxs, const int* __restrict__ literal_offsets,
                             int* __restrict__ clause_position_bounds, int* __restrict__ clause_feat_bounds,
                             int* __restrict__ constrained_fids, int* __restrict__ n_constrained,
                             uint* __restrict__ num_includes, int8_t* __restrict__ is_clause_valid,
                             int8_t* __restrict__ is_clause_synced) {
    int lane = threadIdx.x % warpSize;
    ull warp_id = (ull)(blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    ull total_warps = (ull)(blockDim.x * gridDim.x) / warpSize;

    for (ull clause = warp_id; clause < (ull)TOTAL_CLAUSES; clause += total_warps) {
        if (is_clause_synced[clause])
            continue;

        const uint* ta_state = &global_ta_states[clause * (ull)N_LITERALS];
        int* pos = &clause_position_bounds[clause * 4];

        // Position literals
        PositionResult pr = scan_position_literals(ta_state, lane);

        if (lane == 0) {
            pos[0] = pr.pos0;
            pos[1] = pr.pos1;
            pos[2] = pr.pos2;
            pos[3] = pr.pos3;
        }

        if (!pr.valid) {
            uint pos_includes = warp_reduce_sum(pr.includes);
            if (lane == 0) {
                is_clause_valid[clause] = 0;
                is_clause_synced[clause] = 1;
                num_includes[clause] = pos_includes;
            }
            continue;
        }

        // Feature literals
        int* cfb = &clause_feat_bounds[clause * (ull)N_RAW_PATCH_FEATS * 2];
        int* cfids = &constrained_fids[clause * (ull)N_RAW_PATCH_FEATS];

        FeatureResult fr = scan_feature_literals(ta_state, feat_mins, feat_maxs, literal_offsets, cfb, cfids, lane);

        uint total_inc = warp_reduce_sum(pr.includes + fr.includes);

        if (lane == 0) {
            n_constrained[clause] = fr.n_constrained;
            num_includes[clause] = total_inc;
            is_clause_valid[clause] = fr.all_valid ? 1 : 0;
            is_clause_synced[clause] = 1;
        }
    }
}

} // extern "C"
