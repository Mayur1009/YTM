#ifdef IS_NEOVIM_CLANGD_ENV
#include "cuda.h"
#include "common.h"
#endif

extern "C" __global__ void wic(int class_id, int polarity, const float* clause_weights, const FBOUND_T* clause_feat_bounds,
                               const NFEAT_T* clause_feat_ids, const NFEAT_T* clause_n_feats,
                               const PBOUND_T* clause_position_bounds, const int8_t* has_contra,
                               const float* patch_weights_norm, const FBOUND_T* therm_bits, float pw_th,
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

        if (has_contra[clause_id])
            continue;

        ull rel_clause = clause_id % (ull)CLAUSES_PER_CLASS;
        float w = clause_weights[class_id * (ull)CLAUSES_PER_CLASS + rel_clause];
        if ((polarity > 0 && w <= 0.0f) || (polarity < 0 && w >= 0.0f))
            continue;

#if (N_PATCHES > 1)
        const PBOUND_T* pos = &clause_position_bounds[clause_id * 4];
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
        const FBOUND_T* cfb = &clause_feat_bounds[clause_id * (ull)N_RAW_PATCH_FEATS * 2];
        const NFEAT_T* cfids = &clause_feat_ids[clause_id * (ull)N_RAW_PATCH_FEATS];
        const int n_feats = (int)clause_n_feats[clause_id];

        for (int i = 0; i < n_feats; ++i) {
            int k = (int)cfids[i];
            float cp = (float)((int)cfb[i * 2] + (int)cfb[i * 2 + 1] - (int)therm_bits[k]);

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
                               const FBOUND_T* clause_feat_bounds, const NFEAT_T* clause_feat_ids, const NFEAT_T* clause_n_feats,
                               const int8_t* patch_output, const int8_t* has_contra,
                               const FBOUND_T* therm_bits, float* output) {
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

        if (has_contra[clause_id])
            continue;

        ull rel_clause = clause_id % (ull)CLAUSES_PER_CLASS;
        float w = clause_weights[class_id * (ull)CLAUSES_PER_CLASS + rel_clause];
        if ((polarity > 0 && w <= 0.0f) || (polarity < 0 && w >= 0.0f))
            continue;

        ull act_idx = e * (ull)TOTAL_CLAUSES * N_PATCHES + clause_id * (ull)N_PATCHES + p;
        if (patch_output[act_idx] <= 0)
            continue;

        float wm = fabsf(w);
        const FBOUND_T* cfb = &clause_feat_bounds[clause_id * (ull)N_RAW_PATCH_FEATS * 2];
        const NFEAT_T* cfids = &clause_feat_ids[clause_id * (ull)N_RAW_PATCH_FEATS];
        const int n_feats = (int)clause_n_feats[clause_id];
        float* out_e = &output[e * (ull)HEIGHT * WIDTH * DEPTH];

#if (N_PATCHES > 1)
        int py = p / N_PATCHES_X, px = p % N_PATCHES_X;
        int y0 = py * STRIDE_Y, x0 = px * STRIDE_X;
#endif

        for (int i = 0; i < n_feats; ++i) {
            int k = (int)cfids[i];
            float cp = (float)((int)cfb[i * 2] + (int)cfb[i * 2 + 1] - (int)therm_bits[k]);

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
