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

#if COALESCED == 0
#define CLAUSES_PER_CLASS (TOTAL_CLAUSES / CLASSES)
#define LOOP_CLASS_ID(class_id, clause) class_id = (ull)clause / (CLAUSES_PER_CLASS);
#else
#define CLAUSES_PER_CLASS TOTAL_CLAUSES
#define LOOP_CLASS_ID(class_id, clause) for (class_id = 0; class_id < CLASSES; ++class_id)
#endif

typedef unsigned long long ull;
typedef unsigned int uint;
typedef signed char int8_t;

extern "C" {

__device__ inline int get_feature_value(const int* __restrict__ X, int patch_idx_y, int patch_idx_x, int fid) {
    int rel_y = fid / (PATCH_WIDTH * DEPTH);
    int rel_x = (fid / DEPTH) % PATCH_WIDTH;
    int z = fid % DEPTH;
    int abs_y = patch_idx_y * STRIDE_Y + rel_y;
    int abs_x = patch_idx_x * STRIDE_X + rel_x;
    return X[abs_y * (WIDTH * DEPTH) + abs_x * DEPTH + z];
}

__device__ inline bool match_patch(const int* __restrict__ X, int patch_idx_y, int patch_idx_x,
                                   const int* __restrict__ cfb, const int* __restrict__ cfids, int n_cfids) {
    for (int i = 0; i < n_cfids; ++i) {
        int fid = cfids[i];
        int val = get_feature_value(X, patch_idx_y, patch_idx_x, fid);
        if (val < cfb[fid * 2] || val > cfb[fid * 2 + 1])
            return false;
    }
    return true;
}

__global__ void infer_clauses_conv(const int* __restrict__ X, int8_t* __restrict__ clause_outputs, const int N,
                             const int* __restrict__ clause_position_bounds, const int* __restrict__ clause_feat_bounds,
                             const int* __restrict__ constrained_fids, const int* __restrict__ n_constrained,
                             const uint* __restrict__ num_includes, const int8_t* __restrict__ is_clause_valid) {
    int lane = threadIdx.x % warpSize;
    ull warp_id = (ull)(blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    ull total_warps = (ull)(blockDim.x * gridDim.x) / warpSize;
    ull total_work = (ull)N * TOTAL_CLAUSES;

    for (ull idx = warp_id; idx < total_work; idx += total_warps) {
        ull e = idx / (ull)TOTAL_CLAUSES;
        ull clause = idx % (ull)TOTAL_CLAUSES;

        if (num_includes[clause] == 0) {
            if (lane == 0)
                clause_outputs[idx] = 1;
            continue;
        }
        if (is_clause_valid[clause] == 0) {
            if (lane == 0)
                clause_outputs[idx] = 0;
            continue;
        }

        const int* Xe = &X[e * (ull)HEIGHT * WIDTH * DEPTH];
        const int* pos = &clause_position_bounds[clause * 4];
        int pos0 = pos[0], pos1 = pos[1], pos2 = pos[2], pos3 = pos[3];
        const int* cfb = &clause_feat_bounds[clause * (ull)N_RAW_PATCH_FEATS * 2];
        const int* cfids = &constrained_fids[clause * (ull)N_RAW_PATCH_FEATS];
        int n_cfids = n_constrained[clause];

        bool found = false;
        for (int base = 0; base < N_PATCHES && !found; base += warpSize) {
            int patch = base + lane;
            bool match = false;

            if (patch < N_PATCHES) {
                int py = patch / N_PATCHES_X;
                int px = patch % N_PATCHES_X;
                if (py >= pos0 && py <= pos1 && px >= pos2 && px <= pos3)
                    match = match_patch(Xe, py, px, cfb, cfids, n_cfids);
            }

            if (__ballot_sync(0xFFFFFFFF, match))
                found = true;
        }

        if (lane == 0)
            clause_outputs[idx] = found ? 1 : 0;
    }
}

__global__ void infer_clauses_noconv(const int* __restrict__ X, int8_t* __restrict__ clause_outputs, const int N,
                                    const int* __restrict__ clause_position_bounds, const int* __restrict__ clause_feat_bounds,
                                    const int* __restrict__ constrained_fids, const int* __restrict__ n_constrained,
                                    const uint* __restrict__ num_includes, const int8_t* __restrict__ is_clause_valid) {
    int lane = threadIdx.x % warpSize;
    ull warp_id = (ull)(blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    ull total_warps = (ull)(blockDim.x * gridDim.x) / warpSize;
    ull total_work = (ull)N * TOTAL_CLAUSES;

    for (ull idx = warp_id; idx < total_work; idx += total_warps) {
        ull e = idx / (ull)TOTAL_CLAUSES;
        ull clause = idx % (ull)TOTAL_CLAUSES;

        if (num_includes[clause] == 0) {
            if (lane == 0) clause_outputs[idx] = 1;
            continue;
        }
        if (is_clause_valid[clause] == 0) {
            if (lane == 0) clause_outputs[idx] = 0;
            continue;
        }

        const int* Xe = &X[e * (ull)HEIGHT * WIDTH * DEPTH];
        const int* cfb = &clause_feat_bounds[clause * (ull)N_RAW_PATCH_FEATS * 2];
        const int* cfids = &constrained_fids[clause * (ull)N_RAW_PATCH_FEATS];
        int n_cfids = n_constrained[clause];

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
            clause_outputs[idx] = matched ? 1 : 0;
    }
}

__global__ void sum_votes(const int8_t* __restrict__ clause_outputs, const float* __restrict__ clause_weights,
                          float* __restrict__ class_sums, const int N) {
    int lane = threadIdx.x % warpSize;
    ull warp_id = (ull)(blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    ull total_warps = (ull)(blockDim.x * gridDim.x) / warpSize;
    ull total_work = (ull)N * CLASSES;

    for (ull idx = warp_id; idx < total_work; idx += total_warps) {
        ull e = idx / (ull)CLASSES;
        ull class_id = idx % (ull)CLASSES;

        const int8_t* co = &clause_outputs[e * (ull)TOTAL_CLAUSES];
        const float* cw = &clause_weights[class_id * (ull)CLAUSES_PER_CLASS];
        ull clause_base = class_id * (ull)CLAUSES_PER_CLASS;

        float partial = 0.0f;
        for (int c = lane; c < CLAUSES_PER_CLASS; c += warpSize) {
#if COALESCED == 0
            if (co[clause_base + c])
                partial += cw[c];
#else
            if (co[c])
                partial += cw[c];
#endif
        }

        // Warp reduction
        for (int d = warpSize / 2; d > 0; d >>= 1)
            partial += __shfl_xor_sync(0xFFFFFFFF, partial, d);

        if (lane == 0)
            class_sums[e * (ull)CLASSES + class_id] = partial;
    }
}

__global__ void infer_clauses_patchwise(const int* __restrict__ X, int8_t* __restrict__ patch_output, const int N,
                                       const int* __restrict__ clause_position_bounds, const int* __restrict__ clause_feat_bounds,
                                       const int* __restrict__ constrained_fids, const int* __restrict__ n_constrained,
                                       const uint* __restrict__ num_includes, const int8_t* __restrict__ is_clause_valid) {
    ull tid = threadIdx.x + blockIdx.x * blockDim.x;
    ull stride = blockDim.x * gridDim.x;

    for (ull idx = tid; idx < (ull)N * TOTAL_CLAUSES * N_PATCHES; idx += stride) {
        ull e = idx / ((ull)TOTAL_CLAUSES * N_PATCHES);
        ull clause_patch = idx % ((ull)TOTAL_CLAUSES * N_PATCHES);
        ull clause = clause_patch / (ull)N_PATCHES;
        int patch = clause_patch % N_PATCHES;

        int8_t* output = &patch_output[e * (ull)TOTAL_CLAUSES * N_PATCHES + clause * (ull)N_PATCHES + patch];

        // Empty clause matches all patches
        if (num_includes[clause] == 0) {
            *output = 1;
            continue;
        }

        // Skip invalid clauses (contradictions)
        if (is_clause_valid[clause] == 0) {
            *output = 0;
            continue;
        }

        // Check position bounds
        const int* pos = &clause_position_bounds[clause * 4];
        int py = patch / N_PATCHES_X;
        int px = patch % N_PATCHES_X;

        // Closed interval [pos[0], pos[1]]
        if (py < pos[0] || py > pos[1] || px < pos[2] || px > pos[3]) {
            *output = 0;
            continue;
        }

        // Use match_patch with sparse range representation
        const int* Xe = &X[e * (ull)HEIGHT * WIDTH * DEPTH];
        const int* cfb = &clause_feat_bounds[clause * (ull)N_RAW_PATCH_FEATS * 2];
        const int* cfids = &constrained_fids[clause * (ull)N_RAW_PATCH_FEATS];
        int n_cfids = n_constrained[clause];

        *output = match_patch(Xe, py, px, cfb, cfids, n_cfids) ? 1 : 0;
    }
}

} // extern "C"
