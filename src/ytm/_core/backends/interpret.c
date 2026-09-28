#ifdef IS_NEOVIM_CLANGD_ENV
#include "common.h"
#endif

#include <math.h>

void wic(int class_id, int polarity, const float* restrict clause_weights, const FBOUND_T* restrict clause_feat_bounds,
         const NFEAT_T* restrict clause_feat_ids, const NFEAT_T* restrict clause_n_feats,
         const PBOUND_T* restrict clause_position_bounds, const int8_t* restrict has_contra, const float* restrict patch_weights_norm,
         const FBOUND_T* restrict therm_bits, float pw_th, float* restrict output) {
#pragma omp parallel for schedule(dynamic) reduction(+ : output[ : HEIGHT * WIDTH * DEPTH]) num_threads(ytm_n_threads)
    for (ull clause_id = 0; clause_id < (ull)TOTAL_CLAUSES; clause_id++) {
#if COALESCED == 0
        if ((ull)class_id != clause_id / (ull)CLAUSES_PER_CLASS)
            continue;
#endif
        if (has_contra[clause_id])
            continue;

        float w = clause_weights[weight_offset(class_id, clause_id)];
        if ((polarity > 0 && w <= 0.0f) || (polarity < 0 && w >= 0.0f))
            continue;

        const FBOUND_T* cfb = &clause_feat_bounds[feat_bounds_offset(clause_id, 0, 0)];
        const NFEAT_T* cfids = &clause_feat_ids[feat_ids_offset(clause_id, 0)];
        const int n_feats = (int)clause_n_feats[clause_id];

#if (N_PATCHES > 1)
        const PBOUND_T* pos = &clause_position_bounds[pos_bounds_offset(clause_id, 0)];
        for (int py = pos[0]; py <= pos[1]; py++) {
            for (int px = pos[2]; px <= pos[3]; px++) {
                int p = py * N_PATCHES_X + px;
                float pwv = patch_weights_norm[patch_weights_offset(clause_id, p)];
                if (pwv <= pw_th)
                    continue;

                float wm = fabsf(w) * pwv;

                for (int i = 0; i < n_feats; i++) {
                    int k = (int)cfids[i];
                    float cp = (float)((int)cfb[i * 2] + (int)cfb[i * 2 + 1] - (int)therm_bits[k]);
                    output[feature_offset(k, py, px)] += cp * wm;
                }
            }
        }
#else
        float wm = fabsf(w);
        for (int i = 0; i < n_feats; i++) {
            int k = (int)cfids[i];
            float cp = (float)((int)cfb[i * 2] + (int)cfb[i * 2 + 1] - (int)therm_bits[k]);
            output[feature_offset(k, 0, 0)] += cp * wm;
        }
#endif
    }
}

void wac_sample(int class_id, int polarity, const int8_t* restrict patch_output, const int e, const float* restrict clause_weights,
                const FBOUND_T* restrict clause_feat_bounds, const NFEAT_T* restrict clause_feat_ids, const NFEAT_T* restrict clause_n_feats,
                const int8_t* restrict has_contra, const FBOUND_T* restrict therm_bits,
                float* restrict output) {
    const int8_t* patch_e = &patch_output[patch_output_offset(e, 0, 0)];
    float* out_e = &output[sample_offset(e)];

#pragma omp parallel for schedule(dynamic) reduction(+ : out_e[ : HEIGHT * WIDTH * DEPTH]) num_threads(ytm_n_threads)
    for (ull clause_id = 0; clause_id < (ull)TOTAL_CLAUSES; clause_id++) {
#if COALESCED == 0
        if ((ull)class_id != clause_id / (ull)CLAUSES_PER_CLASS)
            continue;
#endif
        if (has_contra[clause_id])
            continue;

        float w = clause_weights[weight_offset(class_id, clause_id)];
        if ((polarity > 0 && w <= 0.0f) || (polarity < 0 && w >= 0.0f))
            continue;

        const int8_t* clause_act = &patch_e[clause_id * (ull)N_PATCHES];
        const FBOUND_T* cfb = &clause_feat_bounds[feat_bounds_offset(clause_id, 0, 0)];
        const NFEAT_T* cfids = &clause_feat_ids[feat_ids_offset(clause_id, 0)];
        const int n_feats = (int)clause_n_feats[clause_id];
        float wm = fabsf(w);

        for (int p = 0; p < N_PATCHES; p++) {
            if (clause_act[p] <= 0)
                continue;

            int py = p / N_PATCHES_X, px = p % N_PATCHES_X;

            for (int i = 0; i < n_feats; i++) {
                int k = (int)cfids[i];
                float cp = (float)((int)cfb[i * 2] + (int)cfb[i * 2 + 1] - (int)therm_bits[k]);
                out_e[feature_offset(k, py, px)] += cp * wm;
            }
        }
    }
}
