#ifdef IS_NEOVIM_CLANGD_ENV
#include "common.c"
#endif

#include <math.h>

void wic(int class_id, int polarity, const float* clause_weights, const int* clause_feat_bounds,
         const int* clause_position_bounds, const int* clause_density, const float* patch_weights_norm,
         const int* feat_min, const int* feat_max, float pw_th, float* output) {
#pragma omp parallel for schedule(dynamic) reduction(+ : output[ : HEIGHT * WIDTH * DEPTH])
    for (ull clause_id = 0; clause_id < (ull)TOTAL_CLAUSES; clause_id++) {
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

        const int* cfb = &clause_feat_bounds[clause_id * (ull)N_RAW_PATCH_FEATS * 2];

#if (N_PATCHES > 1)
        const int* pos = &clause_position_bounds[clause_id * 4];
        for (int py = pos[0]; py <= pos[1]; py++) {
            for (int px = pos[2]; px <= pos[3]; px++) {
                int p = py * N_PATCHES_X + px;
                float pwv = patch_weights_norm[clause_id * (ull)N_PATCHES + p];
                if (pwv <= pw_th)
                    continue;

                float wm = fabsf(w) * pwv;
                int y0 = py * STRIDE_Y, x0 = px * STRIDE_X;

                for (int k = 0; k < N_RAW_PATCH_FEATS; k++) {
                    float cp = (float)(cfb[k * 2] + cfb[k * 2 + 1] - feat_min[k] - feat_max[k]);
                    int rel_y = k / (PATCH_WIDTH * DEPTH);
                    int rel_x = (k / DEPTH) % PATCH_WIDTH;
                    int d = k % DEPTH;
                    int out_idx = (y0 + rel_y) * (WIDTH * DEPTH) + (x0 + rel_x) * DEPTH + d;
                    output[out_idx] += cp * wm;
                }
            }
        }
#else
        float wm = fabsf(w);
        for (int k = 0; k < N_RAW_PATCH_FEATS; k++) {
            float cp = (float)(cfb[k * 2] + cfb[k * 2 + 1] - feat_min[k] - feat_max[k]);
            output[k] += cp * wm;
        }
#endif
    }
}

void wac_sample(int class_id, int polarity, const int8_t* patch_output, const int e, const float* clause_weights,
                const int* clause_feat_bounds, const int* clause_density, const int* feat_min, const int* feat_max,
                float* output) {
    const int8_t* patch_e = &patch_output[(ull)e * TOTAL_CLAUSES * N_PATCHES];
    float* out_e = &output[(ull)e * HEIGHT * WIDTH * DEPTH];

#pragma omp parallel for schedule(dynamic) reduction(+ : out_e[ : HEIGHT * WIDTH * DEPTH])
    for (ull clause_id = 0; clause_id < (ull)TOTAL_CLAUSES; clause_id++) {
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

        const int8_t* clause_act = &patch_e[clause_id * (ull)N_PATCHES];
        const int* cfb = &clause_feat_bounds[clause_id * (ull)N_RAW_PATCH_FEATS * 2];
        float wm = fabsf(w);

        for (int p = 0; p < N_PATCHES; p++) {
            if (clause_act[p] <= 0)
                continue;

#if (N_PATCHES > 1)
            int py = p / N_PATCHES_X, px = p % N_PATCHES_X;
            int y0 = py * STRIDE_Y, x0 = px * STRIDE_X;
#endif
            for (int k = 0; k < N_RAW_PATCH_FEATS; k++) {
                float cp = (float)(cfb[k * 2] + cfb[k * 2 + 1] - feat_min[k] - feat_max[k]);
#if (N_PATCHES > 1)
                int rel_y = k / (PATCH_WIDTH * DEPTH);
                int rel_x = (k / DEPTH) % PATCH_WIDTH;
                int d = k % DEPTH;
                int out_idx = (y0 + rel_y) * (WIDTH * DEPTH) + (x0 + rel_x) * DEPTH + d;
#else
                int out_idx = k;
#endif
                out_e[out_idx] += cp * wm;
            }
        }
    }
}
