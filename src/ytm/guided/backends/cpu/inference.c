#ifdef IS_NEOVIM_CLANGD_ENV
#include "common.c"
#endif

void infer_sample(const float* clause_weights, const float* bias, const int* clause_position_bounds,
                  const int* clause_feat_bounds, const int* bounded_feat_ids, const int* n_bounded_feats,
                  const int* clause_density, const int* X, const int e, float* class_sums) {
    const int* Xe = &X[(ull)e * HEIGHT * WIDTH * DEPTH];
    float* sums_e = &class_sums[(ull)e * CLASSES];

#if BIAS
    for (int c = 0; c < CLASSES; c++)
        sums_e[c] = bias[c];
#endif

#pragma omp parallel for schedule(dynamic) reduction(+ : sums_e[ : CLASSES])
    for (ull clause = 0; clause < (ull)TOTAL_CLAUSES; clause++) {
        int cd = clause_density[clause];
        int clause_output;

        if (cd < 0) {
            clause_output = 0;
        } else if (cd == 0) {
            clause_output = 1;
        } else {
            const int* cfb = &clause_feat_bounds[clause * (ull)N_RAW_PATCH_FEATS * 2];
            const int* bounded_fids = &bounded_feat_ids[clause * (ull)N_RAW_PATCH_FEATS];
            int n_bounded_fids = n_bounded_feats[clause];

#if (N_PATCHES > 1)
            const int* pos = &clause_position_bounds[clause * 4];
            int pos0 = pos[0], pos1 = pos[1], pos2 = pos[2], pos3 = pos[3];
            int n_y = pos1 - pos0 + 1;
            int n_x = pos3 - pos2 + 1;

            clause_output = 0;
            for (int iy = 0; iy < n_y && !clause_output; iy++) {
                for (int ix = 0; ix < n_x && !clause_output; ix++) {
                    if (match_patch(Xe, pos0 + iy, pos2 + ix, cfb, bounded_fids, n_bounded_fids))
                        clause_output = 1;
                }
            }
#else
            clause_output = match_patch(Xe, 0, 0, cfb, bounded_fids, n_bounded_fids) ? 1 : 0;
#endif
        }

        if (clause_output) {
            ull rel_clause = clause % (ull)CLAUSES_PER_CLASS;
            ull class_id;
            LOOP_CLASS_ID(class_id, clause) {
                sums_e[class_id] += clause_weights[class_id * (ull)CLAUSES_PER_CLASS + rel_clause];
            }
        }
    }
}

void eval_sample_patchwise(const int* clause_position_bounds, const int* clause_feat_bounds,
                           const int* bounded_feat_ids, const int* n_bounded_feats, const int* clause_density,
                           const int* X, const int e, int8_t* patch_output) {
    const int* Xe = &X[(ull)e * HEIGHT * WIDTH * DEPTH];
    int8_t* out_e = &patch_output[(ull)e * TOTAL_CLAUSES * N_PATCHES];

#pragma omp parallel for schedule(dynamic)
    for (ull clause = 0; clause < (ull)TOTAL_CLAUSES; clause++) {
        int cd = clause_density[clause];
        int8_t* clause_out = &out_e[clause * (ull)N_PATCHES];

        if (cd < 0) {
            for (int p = 0; p < N_PATCHES; p++)
                clause_out[p] = 0;
            continue;
        }

        if (cd == 0) {
            for (int p = 0; p < N_PATCHES; p++)
                clause_out[p] = 1;
            continue;
        }

        const int* pos = &clause_position_bounds[clause * 4];
        const int* cfb = &clause_feat_bounds[clause * (ull)N_RAW_PATCH_FEATS * 2];
        const int* bounded_fids = &bounded_feat_ids[clause * (ull)N_RAW_PATCH_FEATS];
        int n_bounded_fids = n_bounded_feats[clause];

        for (int patch = 0; patch < N_PATCHES; patch++) {
            int py = patch / N_PATCHES_X;
            int px = patch % N_PATCHES_X;

            if (py < pos[0] || py > pos[1] || px < pos[2] || px > pos[3]) {
                clause_out[patch] = 0;
                continue;
            }

            clause_out[patch] = match_patch(Xe, py, px, cfb, bounded_fids, n_bounded_fids) ? 1 : 0;
        }
    }
}
