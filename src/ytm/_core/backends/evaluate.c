#ifdef IS_NEOVIM_CLANGD_ENV
#include "common.h"
#include "rng.h"
#endif

void calc_clause_outputs(const PBOUND_T* restrict clause_position_bounds, const FBOUND_T* restrict clause_feat_bounds,
                         const NFEAT_T* restrict clause_feat_ids, const NFEAT_T* restrict clause_n_feats, const int8_t* restrict has_contra,
                         const NLITS_T* restrict clause_len, const FBOUND_T* restrict X, const int e,
                         int8_t* restrict clause_outputs) {
    const FBOUND_T* Xe = &X[(ull)e * HEIGHT * WIDTH * DEPTH];
    int8_t* out_e = &clause_outputs[(ull)e * TOTAL_CLAUSES];

#pragma omp parallel for schedule(dynamic)
    for (ull clause = 0; clause < (ull)TOTAL_CLAUSES; clause++) {
        out_e[clause] = (int8_t)clause_output(Xe, clause, clause_position_bounds, clause_feat_ids, clause_feat_bounds,
                                              clause_n_feats, has_contra[clause], clause_len[clause]);
    }
}

void calc_class_sums(const float* restrict clause_weights, const PBOUND_T* restrict clause_position_bounds,
                     const FBOUND_T* restrict clause_feat_bounds, const NFEAT_T* restrict clause_feat_ids, const NFEAT_T* restrict clause_n_feats,
                     const int8_t* restrict has_contra, const NLITS_T* restrict clause_len, const FBOUND_T* restrict X, const int N,
                     float* restrict class_sums) {
    for (int e = 0; e < N; e++) {
        const FBOUND_T* Xe = &X[(ull)e * HEIGHT * WIDTH * DEPTH];
        float* sums_e = &class_sums[(ull)e * CLASSES];

#pragma omp parallel for schedule(dynamic) reduction(+ : sums_e[ : CLASSES])
        for (ull clause = 0; clause < (ull)TOTAL_CLAUSES; clause++) {
            int out = clause_output(Xe, clause, clause_position_bounds, clause_feat_ids, clause_feat_bounds,
                                    clause_n_feats, has_contra[clause], clause_len[clause]);

            if (out) {
                ull rel_clause = clause % (ull)CLAUSES_PER_CLASS;
                ull class_id;
                LOOP_CLASS_ID(class_id, clause) {
                    sums_e[class_id] += clause_weights[class_id * (ull)CLAUSES_PER_CLASS + rel_clause];
                }
            }
        }
    }
}

void calc_clause_outputs_patchwise(const PBOUND_T* restrict clause_position_bounds, const FBOUND_T* restrict clause_feat_bounds,
                                   const NFEAT_T* restrict clause_feat_ids, const NFEAT_T* restrict clause_n_feats, const int8_t* restrict has_contra,
                                   const NLITS_T* restrict clause_len, const FBOUND_T* restrict X, const int e, int8_t* restrict patch_output) {
    const FBOUND_T* Xe = &X[(ull)e * HEIGHT * WIDTH * DEPTH];
    int8_t* out_e = &patch_output[(ull)e * TOTAL_CLAUSES * N_PATCHES];

#pragma omp parallel for schedule(dynamic)
    for (ull clause = 0; clause < (ull)TOTAL_CLAUSES; clause++) {
        int8_t* clause_out = &out_e[clause * (ull)N_PATCHES];

        if (has_contra[clause]) {
            for (int p = 0; p < N_PATCHES; p++)
                clause_out[p] = 0;
            continue;
        }

        if (clause_len[clause] == 0) {
            for (int p = 0; p < N_PATCHES; p++)
                clause_out[p] = 1;
            continue;
        }

        const FBOUND_T* cfb = &clause_feat_bounds[clause * (ull)N_RAW_PATCH_FEATS * 2];
        const NFEAT_T* bounded_fids = &clause_feat_ids[clause * (ull)N_RAW_PATCH_FEATS];
        int n_bounded_fids = (int)clause_n_feats[clause];

#if (N_PATCHES > 1)
        const PBOUND_T* pos = &clause_position_bounds[clause * 4];

        for (int patch = 0; patch < N_PATCHES; patch++) {
            int py = patch / N_PATCHES_X;
            int px = patch % N_PATCHES_X;

            if (py < pos[0] || py > pos[1] || px < pos[2] || px > pos[3]) {
                clause_out[patch] = 0;
                continue;
            }

            clause_out[patch] = match_patch(Xe, py, px, bounded_fids, cfb, n_bounded_fids) ? 1 : 0;
        }
#else
        clause_out[0] = match_patch(Xe, 0, 0, bounded_fids, cfb, n_bounded_fids) ? 1 : 0;
#endif
    }
}

INLINE_FN void evaluate_noconv(const FBOUND_T* restrict Xe, const int8_t* restrict clause_drop_mask, const FBOUND_T* restrict clause_feat_bounds,
                                   const NFEAT_T* restrict clause_feat_ids, const NFEAT_T* restrict clause_n_feats, const int8_t* restrict has_contra,
                                   const NLITS_T* restrict clause_len, int8_t* restrict clause_output) {
#pragma omp parallel for schedule(dynamic)
    for (ull clause = 0; clause < (ull)TOTAL_CLAUSES; clause++) {
        if (clause_drop_mask[clause] == 1 || has_contra[clause]) {
            clause_output[clause] = 0;
            continue;
        }

        if (clause_len[clause] == 0) {
            clause_output[clause] = 1;
            continue;
        }

        const FBOUND_T* feat_bounds = &clause_feat_bounds[clause * (ull)N_RAW_PATCH_FEATS * 2];
        const NFEAT_T* bounded_fids = &clause_feat_ids[clause * (ull)N_RAW_PATCH_FEATS];
        int n_bounded_fids = (int)clause_n_feats[clause];

        clause_output[clause] = match_patch(Xe, 0, 0, bounded_fids, feat_bounds, n_bounded_fids) ? 1 : 0;
    }
}

INLINE_FN void evaluate_conv(const ull seed, const FBOUND_T* restrict Xe, const int8_t* restrict clause_drop_mask,
                                 const PBOUND_T* restrict clause_position_bounds, const FBOUND_T* restrict clause_feat_bounds,
                                 const NFEAT_T* restrict clause_feat_ids, const NFEAT_T* restrict clause_n_feats, const int8_t* restrict has_contra,
                                 const NLITS_T* restrict clause_len, int8_t* restrict clause_output,
                                 NPATCHES_T* restrict selected_patch_ids, int* restrict patch_weights) {
#pragma omp parallel for schedule(dynamic)
    for (ull clause = 0; clause < (ull)TOTAL_CLAUSES; clause++) {
        if (clause_drop_mask[clause] == 1 || has_contra[clause]) {
            clause_output[clause] = 0;
            continue;
        }

        ull rng_k = rng_hash(seed, clause, 0xDEADBEEFULL);
        uint rng_counter = 0;

        if (clause_len[clause] == 0) {
            int selected_id = (int)(rand_uniform(rng_k, &rng_counter) * N_PATCHES);
            clause_output[clause] = 1;
            selected_patch_ids[clause] = (NPATCHES_T)selected_id;
#if TRACK_PATCH_WEIGHTS
            patch_weights[clause * (ull)N_PATCHES + selected_id]++;
#endif
            continue;
        }

        const PBOUND_T* pos = &clause_position_bounds[clause * 4];
        const int pos0 = pos[0], pos1 = pos[1], pos2 = pos[2], pos3 = pos[3];
        const FBOUND_T* feat_bounds = &clause_feat_bounds[clause * (ull)N_RAW_PATCH_FEATS * 2];
        const NFEAT_T* bounded_fids = &clause_feat_ids[clause * (ull)N_RAW_PATCH_FEATS];
        const int n_bounded_fids = (int)clause_n_feats[clause];

        int selected_id = -1;
        int count = 0;
        int n_y = pos1 - pos0 + 1;
        int n_x = pos3 - pos2 + 1;

        for (int iy = 0; iy < n_y; iy++) {
            for (int ix = 0; ix < n_x; ix++) {
                int py = pos0 + iy;
                int px = pos2 + ix;
                if (!match_patch(Xe, py, px, bounded_fids, feat_bounds, n_bounded_fids))
                    continue;
                count++;
                if ((rand_uniform(rng_k, &rng_counter) * count) < 1.0f)
                    selected_id = py * N_PATCHES_X + px;
            }
        }

        clause_output[clause] = (selected_id >= 0) ? 1 : 0;
        if (selected_id >= 0) {
            selected_patch_ids[clause] = (NPATCHES_T)selected_id;
#if TRACK_PATCH_WEIGHTS
            patch_weights[clause * (ull)N_PATCHES + selected_id]++;
#endif
        }
    }
}

void evaluate(const ull seed, const FBOUND_T* restrict X, const int e, const int8_t* restrict clause_drop_mask,
              const PBOUND_T* restrict clause_position_bounds, const FBOUND_T* restrict clause_feat_bounds, const NFEAT_T* restrict clause_feat_ids,
              const NFEAT_T* restrict clause_n_feats, const int8_t* restrict has_contra, const NLITS_T* restrict clause_len,
              int8_t* restrict clause_output, NPATCHES_T* restrict selected_patch_ids, int* restrict patch_weights) {
    const FBOUND_T* Xe = &X[(ull)e * HEIGHT * WIDTH * DEPTH];
#if (N_PATCHES > 1)
    evaluate_conv(seed, Xe, clause_drop_mask, clause_position_bounds, clause_feat_ids, clause_feat_bounds,
                  clause_n_feats, has_contra, clause_len, clause_output, selected_patch_ids, patch_weights);
#else
    evaluate_noconv(Xe, clause_drop_mask, clause_feat_bounds, clause_feat_ids, clause_n_feats, has_contra,
                    clause_len, clause_output);
#endif
}

void count_votes(const int8_t* restrict clause_output, const float* restrict clause_weights, float* restrict votes) {
    for (int c = 0; c < CLASSES; c++)
        votes[c] = 0.0f;

#pragma omp parallel for schedule(dynamic) reduction(+ : votes[ : CLASSES])
    for (ull clause = 0; clause < TOTAL_CLAUSES; clause++) {
        if (clause_output[clause]) {
            ull class_id, rel_clause = clause % CLAUSES_PER_CLASS;
            LOOP_CLASS_ID(class_id, clause) {
                votes[class_id] += clause_weights[class_id * CLAUSES_PER_CLASS + rel_clause];
            }
        }
    }
}
