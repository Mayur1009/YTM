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
#define TRACK_PATCH_WEIGHTS 1
#define N_RAW_PATCH_FEATS 100
#define N_PATCH_FEATS 100
#define N_POSITION_FEATS 36
#define N_PATCHES_Y 19
#define N_PATCHES_X 19
#define N_PATCHES 361
#define N_LITERALS 272
#define WARP_SIZE 32
#endif

#include <curand_kernel.h>

typedef signed char int8_t;
typedef unsigned long long ull;
typedef unsigned int uint;

extern "C" {

__device__ inline int get_feature_value(const int* X, int patch_idx_y, int patch_idx_x, int fid) {
    int rel_y = fid / (PATCH_WIDTH * DEPTH);
    int rel_x = (fid / DEPTH) % PATCH_WIDTH;
    int z = fid % DEPTH;
    int abs_y = patch_idx_y * STRIDE_Y + rel_y;
    int abs_x = patch_idx_x * STRIDE_X + rel_x;
    return X[abs_y * (WIDTH * DEPTH) + abs_x * DEPTH + z];
}

__device__ inline bool match_patch(const int* X, int patch_idx_y, int patch_idx_x,
                                   const int* cfb, const int* cfids, int n_cfids) {
    for (int i = 0; i < n_cfids; ++i) {
        int fid = cfids[i];
        int val = get_feature_value(X, patch_idx_y, patch_idx_x, fid);
        if (val < cfb[fid * 2] || val > cfb[fid * 2 + 1])
            return false;
    }
    return true;
}

// --- Convolution: 1 warp per clause, 32 lanes evaluate patches in parallel ---

__global__ void evaluate_conv(const int* X, const int e, const int8_t* clause_drop_mask,
                         const int* clause_position_bounds, const int* clause_feat_bounds,
                         const int* constrained_fids, const int* n_constrained,
                         const uint* num_includes, const int8_t* is_clause_valid,
                         curandState* rng, int* selected_patch_ids, int* patch_weights) {
    int lane = threadIdx.x % warpSize;
    ull warp_id = (ull)(blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    ull total_warps = (ull)(blockDim.x * gridDim.x) / warpSize;

    const int* Xe = &X[(ull)e * HEIGHT * WIDTH * DEPTH];

    for (ull clause = warp_id; clause < (ull)TOTAL_CLAUSES; clause += total_warps) {
        if (clause_drop_mask[clause] == 1 || is_clause_valid[clause] == 0) {
            if (lane == 0) selected_patch_ids[clause] = -1;
            continue;
        }

        bool empty = (num_includes[clause] == 0);

        const int* pos = &clause_position_bounds[clause * 4];
        int pos0 = pos[0], pos1 = pos[1], pos2 = pos[2], pos3 = pos[3];
        const int* cfb = &clause_feat_bounds[clause * (ull)N_RAW_PATCH_FEATS * 2];
        const int* cfids = &constrained_fids[clause * (ull)N_RAW_PATCH_FEATS];
        int n_cfids = n_constrained[clause];

        curandState local_rng;
        if (lane == 0) local_rng = rng[clause];

        int selected_id = -1;
        int count = 0;

        for (int base = 0; base < N_PATCHES; base += warpSize) {
            int patch = base + lane;
            bool match = false;

            if (patch < N_PATCHES) {
                if (empty) {
                    match = true;
                } else {
                    int py = patch / N_PATCHES_X;
                    int px = patch % N_PATCHES_X;
                    if (py >= pos0 && py <= pos1 && px >= pos2 && px <= pos3)
                        match = match_patch(Xe, py, px, cfb, cfids, n_cfids);
                }
            }

            uint ballot = __ballot_sync(0xFFFFFFFF, match);

            if (lane == 0) {
                uint mask = ballot;
                while (mask) {
                    int bit = __ffs(mask) - 1;
                    mask &= mask - 1;
                    count++;
                    if (curand_uniform(&local_rng) < 1.0f / count)
                        selected_id = base + bit;
                }
            }
        }

        if (lane == 0) {
            selected_patch_ids[clause] = selected_id;
            rng[clause] = local_rng;

#if TRACK_PATCH_WEIGHTS
            if (selected_id >= 0)
                patch_weights[clause * (ull)N_PATCHES + selected_id]++;
#endif
        }
    }
}

// --- Non-convolution: 1 warp per clause, 32 lanes split features via __all_sync ---

__global__ void evaluate_noconv(const int* X, const int e, const int8_t* clause_drop_mask,
                                const int* clause_position_bounds, const int* clause_feat_bounds,
                                const int* constrained_fids, const int* n_constrained,
                                const uint* num_includes, const int8_t* is_clause_valid,
                                curandState* rng, int* selected_patch_ids, int* patch_weights) {
    int lane = threadIdx.x % warpSize;
    ull warp_id = (ull)(blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    ull total_warps = (ull)(blockDim.x * gridDim.x) / warpSize;

    const int* Xe = &X[(ull)e * HEIGHT * WIDTH * DEPTH];

    for (ull clause = warp_id; clause < (ull)TOTAL_CLAUSES; clause += total_warps) {
        if (clause_drop_mask[clause] == 1 || is_clause_valid[clause] == 0) {
            if (lane == 0) selected_patch_ids[clause] = -1;
            continue;
        }

        if (num_includes[clause] == 0) {
            if (lane == 0) selected_patch_ids[clause] = 0;
            continue;
        }

        const int* cfb = &clause_feat_bounds[clause * (ull)N_RAW_PATCH_FEATS * 2];
        const int* cfids = &constrained_fids[clause * (ull)N_RAW_PATCH_FEATS];
        int n_cfids = n_constrained[clause];

        // 32 lanes split constrained features
        bool my_match = true;
        for (int i = lane; i < n_cfids; i += warpSize) {
            int fid = cfids[i];
            int val = get_feature_value(Xe, 0, 0, fid);
            if (val < cfb[fid * 2] || val > cfb[fid * 2 + 1]) {
                my_match = false;
                break;
            }
        }
        bool matched = __all_sync(0xFFFFFFFF, my_match);

        if (lane == 0)
            selected_patch_ids[clause] = matched ? 0 : -1;
    }
}

} // extern "C"
