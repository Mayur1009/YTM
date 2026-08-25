#ifdef IS_NEOVIM_CLANGD_ENV
#include "common.c"
#endif

static inline void evaluate_noconv(const int* Xe, const int8_t* clause_drop_mask, const int* clause_feat_bounds,
                                   const int* bounded_feat_ids, const int* n_bounded_feats, const int* clause_density,
                                   int* selected_patch_ids) {
#pragma omp parallel for schedule(dynamic)
    for (ull clause = 0; clause < (ull)TOTAL_CLAUSES; clause++) {
        int cd = clause_density[clause];
        if (clause_drop_mask[clause] == 1 || cd < 0) {
            selected_patch_ids[clause] = -1;
            continue;
        }

        if (cd == 0) {
            selected_patch_ids[clause] = 0;
            continue;
        }

        const int* feat_bounds = &clause_feat_bounds[clause * (ull)N_RAW_PATCH_FEATS * 2];
        const int* bounded_fids = &bounded_feat_ids[clause * (ull)N_RAW_PATCH_FEATS];
        int n_bounded_fids = n_bounded_feats[clause];

        bool matched = match_patch(Xe, 0, 0, feat_bounds, bounded_fids, n_bounded_fids);
        selected_patch_ids[clause] = matched ? 0 : -1;
    }
}

static inline void evaluate_conv(const int* Xe, const int8_t* clause_drop_mask, const int* clause_position_bounds,
                                 const int* clause_feat_bounds, const int* bounded_feat_ids, const int* n_bounded_feats,
                                 const int* clause_density, const ull seed, const ull e, int* selected_patch_ids,
                                 int* patch_weights) {
#pragma omp parallel for schedule(dynamic)
    for (ull clause = 0; clause < (ull)TOTAL_CLAUSES; clause++) {
        int cd = clause_density[clause];
        if (clause_drop_mask[clause] == 1 || cd < 0) {
            selected_patch_ids[clause] = -1;
            continue;
        }

        ull rng_k = rng_hash(seed, clause, e, 0xDEADBEEFULL);
        uint rng_counter = 0;

        if (cd == 0) {
            int selected_id = (int)(rand_uniform(rng_k, &rng_counter) * N_PATCHES);
            selected_patch_ids[clause] = selected_id;
#if TRACK_PATCH_WEIGHTS
            patch_weights[clause * (ull)N_PATCHES + selected_id]++;
#endif
            continue;
        }

        const int* pos = &clause_position_bounds[clause * 4];
        const int pos0 = pos[0], pos1 = pos[1], pos2 = pos[2], pos3 = pos[3];
        const int* feat_bounds = &clause_feat_bounds[clause * (ull)N_RAW_PATCH_FEATS * 2];
        const int* bounded_fids = &bounded_feat_ids[clause * (ull)N_RAW_PATCH_FEATS];
        const int n_bounded_fids = n_bounded_feats[clause];

        int selected_id = -1;
        int count = 0;
        int n_y = pos1 - pos0 + 1;
        int n_x = pos3 - pos2 + 1;

        for (int iy = 0; iy < n_y; iy++) {
            for (int ix = 0; ix < n_x; ix++) {
                int py = pos0 + iy;
                int px = pos2 + ix;
                if (!match_patch(Xe, py, px, feat_bounds, bounded_fids, n_bounded_fids))
                    continue;
                count++;
                if ((rand_uniform(rng_k, &rng_counter) * count) < 1.0f)
                    selected_id = py * N_PATCHES_X + px;
            }
        }

        selected_patch_ids[clause] = selected_id;
#if TRACK_PATCH_WEIGHTS
        if (selected_id >= 0)
            patch_weights[clause * (ull)N_PATCHES + selected_id]++;
#endif
    }
}

void evaluate(const int* X, const int e, const int8_t* clause_drop_mask, const int* clause_position_bounds,
              const int* clause_feat_bounds, const int* bounded_feat_ids, const int* n_bounded_feats,
              const int* clause_density, const ull seed, int* selected_patch_ids, int* patch_weights) {
    const int* Xe = &X[(ull)e * HEIGHT * WIDTH * DEPTH];
#if (N_PATCHES > 1)
    evaluate_conv(Xe, clause_drop_mask, clause_position_bounds, clause_feat_bounds, bounded_feat_ids, n_bounded_feats,
                  clause_density, seed, (ull)e, selected_patch_ids, patch_weights);
#else
    evaluate_noconv(Xe, clause_drop_mask, clause_feat_bounds, bounded_feat_ids, n_bounded_feats, clause_density,
                    selected_patch_ids);
#endif
}

void count_votes(const int* selected_patch_ids, const float* clause_weights, const float* bias, float* votes) {
    for (int c = 0; c < CLASSES; c++)
#if BIAS
        votes[c] = bias[c];
#else
        votes[c] = 0.0f;
#endif

#pragma omp parallel for schedule(dynamic) reduction(+ : votes[ : CLASSES])
    for (ull clause = 0; clause < TOTAL_CLAUSES; clause++) {
        if (selected_patch_ids[clause] != -1) {
            ull class_id, rel_clause = clause % CLAUSES_PER_CLASS;
            LOOP_CLASS_ID(class_id, clause) {
                votes[class_id] += clause_weights[class_id * CLAUSES_PER_CLASS + rel_clause];
            }
        }
    }
}
