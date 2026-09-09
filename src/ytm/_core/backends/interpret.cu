#ifdef IS_NEOVIM_CLANGD_ENV
#include "cuda.h"
#include "common.h"
#endif

extern "C" __global__ void wic(int class_id, int polarity, const float* clause_weights, const int* clause_feat_bounds,
                               const int* clause_position_bounds, const int* clause_density,
                               const float* patch_weights_norm, const int* feat_min, const int* feat_max, float pw_th,
                               float* output) {
    ull tid = threadIdx.x + blockIdx.x * blockDim.x;
    ull stride = blockDim.x * gridDim.x;

    for (ull idx = tid; idx < (ull)TOTAL_CLAUSES * N_PATCHES; idx += stride) {
        ull clause_id = idx / N_PATCHES;
        int p = idx % N_PATCHES;

#if COALESCED == 0
        if ((ull)class_id != clause_id / (ull)CLAUSES_PER_CLASS)
            continue;
#endif

        if (clause_density[clause_id] == -1)
            continue;

        ull rel_clause = clause_id % (ull)CLAUSES_PER_CLASS;
        float w = clause_weights[class_id * (ull)CLAUSES_PER_CLASS + rel_clause];
        if ((polarity > 0 && w <= 0.0f) || (polarity < 0 && w >= 0.0f))
            continue;

#if (N_PATCHES > 1)
        const int* pos = &clause_position_bounds[clause_id * 4];
        int py = p / N_PATCHES_X, px = p % N_PATCHES_X;
        if (py < pos[0] || py > pos[1] || px < pos[2] || px > pos[3])
            continue;

        float pwv = patch_weights_norm[clause_id * (ull)N_PATCHES + p];
        if (pwv <= pw_th)
            continue;

        int y0 = py * STRIDE_Y, x0 = px * STRIDE_X;
#else
        float pwv = 1.0f;
#endif

        float wm = fabsf(w) * pwv;
        const int* cfb = &clause_feat_bounds[clause_id * (ull)N_RAW_PATCH_FEATS * 2];

        for (int k = 0; k < N_RAW_PATCH_FEATS; ++k) {
            float cp = (float)(cfb[k * 2] + cfb[k * 2 + 1] - feat_min[k] - feat_max[k]);

#if (N_PATCHES > 1)
            int rel_y = k / (PATCH_WIDTH * DEPTH);
            int rel_x = (k / DEPTH) % PATCH_WIDTH;
            int d = k % DEPTH;
            int out_idx = (y0 + rel_y) * (WIDTH * DEPTH) + (x0 + rel_x) * DEPTH + d;
#else
            int out_idx = k;
#endif
            atomicAdd(&output[out_idx], cp * wm);
        }
    }
}

extern "C" __global__ void wac(const int* target_classes, int polarity, int N, const float* clause_weights,
                               const int* clause_feat_bounds, const int8_t* patch_output, const int* clause_density,
                               const int* feat_min, const int* feat_max, float* output) {
    ull tid = threadIdx.x + blockIdx.x * blockDim.x;
    ull stride = blockDim.x * gridDim.x;

    for (ull idx = tid; idx < (ull)N * TOTAL_CLAUSES * N_PATCHES; idx += stride) {
        ull e = idx / ((ull)TOTAL_CLAUSES * N_PATCHES);
        ull clause_patch = idx % ((ull)TOTAL_CLAUSES * N_PATCHES);
        ull clause_id = clause_patch / (ull)N_PATCHES;
        int p = clause_patch % N_PATCHES;

        int class_id = target_classes[e];

#if COALESCED == 0
        if ((ull)class_id != clause_id / (ull)CLAUSES_PER_CLASS)
            continue;
#endif

        if (clause_density[clause_id] == -1)
            continue;

        ull rel_clause = clause_id % (ull)CLAUSES_PER_CLASS;
        float w = clause_weights[class_id * (ull)CLAUSES_PER_CLASS + rel_clause];
        if ((polarity > 0 && w <= 0.0f) || (polarity < 0 && w >= 0.0f))
            continue;

        ull act_idx = e * (ull)TOTAL_CLAUSES * N_PATCHES + clause_id * (ull)N_PATCHES + p;
        if (patch_output[act_idx] <= 0)
            continue;

        float wm = fabsf(w);
        const int* cfb = &clause_feat_bounds[clause_id * (ull)N_RAW_PATCH_FEATS * 2];
        float* out_e = &output[e * (ull)HEIGHT * WIDTH * DEPTH];

#if (N_PATCHES > 1)
        int py = p / N_PATCHES_X, px = p % N_PATCHES_X;
        int y0 = py * STRIDE_Y, x0 = px * STRIDE_X;
#endif

        for (int k = 0; k < N_RAW_PATCH_FEATS; ++k) {
            float cp = (float)(cfb[k * 2] + cfb[k * 2 + 1] - feat_min[k] - feat_max[k]);

#if (N_PATCHES > 1)
            int rel_y = k / (PATCH_WIDTH * DEPTH);
            int rel_x = (k / DEPTH) % PATCH_WIDTH;
            int d = k % DEPTH;
            int out_idx = (y0 + rel_y) * (WIDTH * DEPTH) + (x0 + rel_x) * DEPTH + d;
#else
            int out_idx = k;
#endif
            atomicAdd(&out_e[out_idx], cp * wm);
        }
    }
}
