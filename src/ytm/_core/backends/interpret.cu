#ifdef IS_NEOVIM_CLANGD_ENV
#include "common.h"
#include "cuda.h"
#endif

extern "C" __global__ void wic(int class_id, int polarity, const float* clause_weights, const FBOUND_T* clause_feat_bounds,
                               const NFEAT_T* clause_feat_ids, const NFEAT_T* clause_n_feats,
                               const PBOUND_T* clause_position_bounds, const int8_t* has_contra,
                               const float* patch_weights_norm, const FBOUND_T* therm_bits, float pw_th,
                               float* output) {
    GRID_STRIDE_LOOP(idx, (ull)TOTAL_CLAUSES * N_PATCHES) {
        ull clause_id = idx / N_PATCHES;
        int p = idx % N_PATCHES;

#if COALESCED == 0
        if ((ull)class_id != clause_id / (ull)CLAUSES_PER_CLASS)
            continue;
#endif

        if (has_contra[clause_id])
            continue;

        float w = clause_weights[weight_offset(class_id, clause_id)];
        if ((polarity > 0 && w <= 0.0f) || (polarity < 0 && w >= 0.0f))
            continue;

        int py = p / N_PATCHES_X, px = p % N_PATCHES_X;

#if (N_PATCHES > 1)
        const PBOUND_T* pos = &clause_position_bounds[pos_bounds_offset(clause_id, 0)];
        if (py < pos[0] || py > pos[1] || px < pos[2] || px > pos[3])
            continue;

        float pwv = patch_weights_norm[patch_weights_offset(clause_id, p)];
        if (pwv <= pw_th)
            continue;
#else
        float pwv = 1.0f;
#endif

        float wm = fabsf(w) * pwv;
        const FBOUND_T* cfb = &clause_feat_bounds[feat_bounds_offset(clause_id, 0, 0)];
        const NFEAT_T* cfids = &clause_feat_ids[feat_ids_offset(clause_id, 0)];
        const int n_feats = (int)clause_n_feats[clause_id];

        for (int i = 0; i < n_feats; ++i) {
            int k = (int)cfids[i];
            float cp = (float)((int)cfb[i * 2] + (int)cfb[i * 2 + 1] - (int)therm_bits[k]);
            atomicAdd(&output[feature_offset(k, py, px)], cp * wm);
        }
    }
}

extern "C" __global__ void wac(const int* target_classes, int polarity, int N, const float* clause_weights,
                               const FBOUND_T* clause_feat_bounds, const NFEAT_T* clause_feat_ids, const NFEAT_T* clause_n_feats,
                               const int8_t* patch_output, const int8_t* has_contra,
                               const FBOUND_T* therm_bits, float* output) {
    GRID_STRIDE_LOOP(idx, (ull)N * TOTAL_CLAUSES * N_PATCHES) {
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

        float w = clause_weights[weight_offset(class_id, clause_id)];
        if ((polarity > 0 && w <= 0.0f) || (polarity < 0 && w >= 0.0f))
            continue;

        ull act_idx = patch_output_offset(e, clause_id, p);
        if (patch_output[act_idx] <= 0)
            continue;

        float wm = fabsf(w);
        const FBOUND_T* cfb = &clause_feat_bounds[feat_bounds_offset(clause_id, 0, 0)];
        const NFEAT_T* cfids = &clause_feat_ids[feat_ids_offset(clause_id, 0)];
        const int n_feats = (int)clause_n_feats[clause_id];
        float* out_e = &output[sample_offset(e)];

        int py = p / N_PATCHES_X, px = p % N_PATCHES_X;

        for (int i = 0; i < n_feats; ++i) {
            int k = (int)cfids[i];
            float cp = (float)((int)cfb[i * 2] + (int)cfb[i * 2 + 1] - (int)therm_bits[k]);
            atomicAdd(&out_e[feature_offset(k, py, px)], cp * wm);
        }
    }
}
