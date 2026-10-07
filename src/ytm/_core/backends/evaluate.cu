#ifdef IS_NEOVIM_CLANGD_ENV
#include "common.h"
#include "cuda.h"
#endif

__device__ void calc_clause_outputs_conv(const FBOUND_T* X, int8_t* clause_outputs, const int N,
                                         const PBOUND_T* clause_position_bounds, const FBOUND_T* clause_feat_bounds,
                                         const NFEAT_T* clause_feat_ids, const NFEAT_T* clause_n_feats,
                                         const int8_t* has_contra, const NLITS_T* clause_len) {
    auto [warp, lane, warp_id, total_warps] = warp_grid();
    ull total_work = (ull)N * TOTAL_CLAUSES;

    WARP_STRIDE_LOOP(idx, total_work) {
        ull e = idx / (ull)TOTAL_CLAUSES;
        ull clause = idx % (ull)TOTAL_CLAUSES;

        if (has_contra[clause] || clause_len[clause] == 0) {
            if (lane == 0)
                clause_outputs[idx] = has_contra[clause] ? 0 : 1;
            continue;
        }

        const FBOUND_T* Xe = &X[sample_offset(e)];
        const PBOUND_T* pos = &clause_position_bounds[pos_bounds_offset(clause, 0)];
        int pos0 = pos[0], pos1 = pos[1], pos2 = pos[2], pos3 = pos[3];
        const FBOUND_T* cfb = &clause_feat_bounds[feat_bounds_offset(clause, 0, 0)];
        const NFEAT_T* bounded_fids = &clause_feat_ids[feat_ids_offset(clause, 0)];
        int n_bounded_fids = (int)clause_n_feats[clause];

        int n_y = pos1 - pos0 + 1;
        int n_x = pos3 - pos2 + 1;
        int patches_to_consider = n_y * n_x;

        bool found = false;
        for (int base = 0; base < patches_to_consider && !found; base += WARP_SIZE) {
            int i = base + lane;
            bool match = false;
            if (i < patches_to_consider) {
                int py = pos0 + i / n_x;
                int px = pos2 + i % n_x;
                match = match_patch(Xe, py, px, bounded_fids, cfb, n_bounded_fids);
            }
            if (warp.any(match))
                found = true;
        }

        if (lane == 0)
            clause_outputs[idx] = found ? 1 : 0;
    }
}

__device__ void calc_clause_outputs_noconv(const FBOUND_T* X, int8_t* clause_outputs, const int N,
                                           const FBOUND_T* clause_feat_bounds, const NFEAT_T* clause_feat_ids,
                                           const NFEAT_T* clause_n_feats, const int8_t* has_contra,
                                           const NLITS_T* clause_len) {
    auto [warp, lane, warp_id, total_warps] = warp_grid();
    ull total_work = (ull)N * TOTAL_CLAUSES;

    WARP_STRIDE_LOOP(idx, total_work) {
        ull e = idx / (ull)TOTAL_CLAUSES;
        ull clause = idx % (ull)TOTAL_CLAUSES;

        if (has_contra[clause] || clause_len[clause] == 0) {
            if (lane == 0) {
                clause_outputs[idx] = has_contra[clause] ? 0 : 1;
            }
            continue;
        }

        const FBOUND_T* Xe = &X[sample_offset(e)];
        const FBOUND_T* cfb = &clause_feat_bounds[feat_bounds_offset(clause, 0, 0)];
        const NFEAT_T* bounded_fids = &clause_feat_ids[feat_ids_offset(clause, 0)];
        int n_bounded_fids = (int)clause_n_feats[clause];

        bool is_matching = true;
        for (int base = 0; base < n_bounded_fids; base += WARP_SIZE) {
            int i = base + lane;
            if (i < n_bounded_fids) {
                FBOUND_T val = get_feature_value(Xe, 0, 0, (int)bounded_fids[i]);
                if (val < cfb[i * 2] || val > cfb[i * 2 + 1])
                    is_matching = false;
            }
            if (warp.any(!is_matching))
                break;
        }
        bool matched = warp.all(is_matching);

        if (lane == 0)
            clause_outputs[idx] = matched ? 1 : 0;
    }
}

extern "C" __global__ void calc_clause_outputs(const FBOUND_T* X, int8_t* clause_outputs, const int N,
                                               const PBOUND_T* clause_position_bounds, const FBOUND_T* clause_feat_bounds,
                                               const NFEAT_T* clause_feat_ids, const NFEAT_T* clause_n_feats,
                                               const int8_t* has_contra, const NLITS_T* clause_len) {
#if (N_PATCHES > 1)
    calc_clause_outputs_conv(X, clause_outputs, N, clause_position_bounds, clause_feat_bounds, clause_feat_ids,
                             clause_n_feats, has_contra, clause_len);
#else
    calc_clause_outputs_noconv(X, clause_outputs, N, clause_feat_bounds, clause_feat_ids, clause_n_feats,
                               has_contra, clause_len);
#endif
}

// Smallest clause range one warp sums alone, below this the combine costs more than the split saves.
#ifndef VOTE_MIN_CHUNK
#define VOTE_MIN_CHUNK 256
#endif

// Class sums for N samples. The N x CLASSES rows are each split into equal clause chunks so that even a
// single sample during training fills every resident warp. Each warp sums its chunk, and the last warp
// to finish a row adds the row's partials in chunk order, so the result is the same on every run.
// `partials` and `row_done` need 2 x the resident warp count entries, and `row_done` must start at zero;
// each row's counter is reset by the warp that finishes it.
extern "C" __global__ void sum_votes(const int8_t* clause_outputs, const float* clause_weights, float* class_sums,
                                     const int N, float* partials, uint* row_done) {
    auto [warp, lane, warp_id, total_warps] = warp_grid();
    const ull rows = (ull)N * CLASSES;

    ull max_chunks = ((ull)CLAUSES_PER_CLASS + VOTE_MIN_CHUNK - 1) / VOTE_MIN_CHUNK;
    max_chunks = max_chunks < 1 ? 1 : max_chunks;
    ull chunks = rows >= total_warps ? 1 : (total_warps + rows - 1) / rows;
    chunks = chunks > max_chunks ? max_chunks : chunks;
    const int chunk_len = (int)(((ull)CLAUSES_PER_CLASS + chunks - 1) / chunks);

    WARP_STRIDE_LOOP(w, rows * chunks) {
        ull row = w / chunks;
        int chunk = (int)(w % chunks);
        ull e = row / (ull)CLASSES;
        ull class_id = row % (ull)CLASSES;

        const int8_t* co = &clause_outputs[e * (ull)TOTAL_CLAUSES];
        const float* cw = &clause_weights[class_id * (ull)CLAUSES_PER_CLASS];
        ull clause_base = class_id * (ull)CLAUSES_PER_CLASS;

        int c0 = chunk * chunk_len;
        int c1 = (c0 + chunk_len < (int)CLAUSES_PER_CLASS) ? c0 + chunk_len : (int)CLAUSES_PER_CLASS;

        float partial = 0.0f;
        for (int c = c0 + lane; c < c1; c += (int)warp.size()) {
#if COALESCED == 0
            if (co[clause_base + c])
                partial += cw[c];
#else
            if (co[c])
                partial += cw[c];
#endif
        }
        partial = cg::reduce(warp, partial, cg::plus<float>());

        if (chunks == 1) {
            if (lane == 0)
                class_sums[row] = partial;
            continue;
        }

        int last = 0;
        if (lane == 0) {
            partials[row * chunks + chunk] = partial;
            __threadfence();
            last = (atomicAdd(&row_done[row], 1u) == (uint)(chunks - 1));
        }
        last = warp.shfl(last, 0);
        if (last) {
            __threadfence();
            float s = 0.0f;
            for (ull k = lane; k < chunks; k += warp.size())
                s += __ldcg(&partials[row * chunks + k]);
            s = cg::reduce(warp, s, cg::plus<float>());
            if (lane == 0) {
                class_sums[row] = s;
                row_done[row] = 0;
            }
        }
    }
}

extern "C" __global__ void calc_clause_outputs_patchwise(const FBOUND_T* X, int8_t* patch_output, const int N,
                                                         const PBOUND_T* clause_position_bounds,
                                                         const FBOUND_T* clause_feat_bounds, const NFEAT_T* clause_feat_ids,
                                                         const NFEAT_T* clause_n_feats, const int8_t* has_contra,
                                                         const NLITS_T* clause_len) {
    GRID_STRIDE_LOOP(idx, (ull)N * TOTAL_CLAUSES * N_PATCHES) {
        ull e = idx / ((ull)TOTAL_CLAUSES * N_PATCHES);
        ull clause_patch = idx % ((ull)TOTAL_CLAUSES * N_PATCHES);
        ull clause = clause_patch / (ull)N_PATCHES;
        int patch = clause_patch % N_PATCHES;

        int8_t* output = &patch_output[patch_output_offset(e, clause, patch)];

        if (has_contra[clause] || clause_len[clause] == 0) {
            *output = has_contra[clause] ? 0 : 1;
            continue;
        }

        int py = patch / N_PATCHES_X;
        int px = patch % N_PATCHES_X;

#if (N_PATCHES > 1)
        const PBOUND_T* pos = &clause_position_bounds[pos_bounds_offset(clause, 0)];
        if (py < pos[0] || py > pos[1] || px < pos[2] || px > pos[3]) {
            *output = 0;
            continue;
        }
#endif

        const FBOUND_T* Xe = &X[sample_offset(e)];
        const FBOUND_T* cfb = &clause_feat_bounds[feat_bounds_offset(clause, 0, 0)];
        const NFEAT_T* bounded_fids = &clause_feat_ids[feat_ids_offset(clause, 0)];
        int n_bounded_fids = (int)clause_n_feats[clause];

        *output = match_patch(Xe, py, px, bounded_fids, cfb, n_bounded_fids) ? 1 : 0;
    }
}

__device__ void evaluate_noconv(const FBOUND_T* Xe, const int8_t* clause_drop_mask, const FBOUND_T* clause_feat_bounds,
                                const NFEAT_T* clause_feat_ids, const NFEAT_T* clause_n_feats, const int8_t* has_contra,
                                const NLITS_T* clause_len, int8_t* clause_output) {
    auto [warp, lane, warp_id, total_warps] = warp_grid();

    WARP_STRIDE_LOOP(clause, (ull)TOTAL_CLAUSES) {
        if (clause_drop_mask[clause] == 1 || has_contra[clause]) {
            if (lane == 0)
                clause_output[clause] = 0;
            continue;
        }

        if (clause_len[clause] == 0) {
            if (lane == 0)
                clause_output[clause] = 1;
            continue;
        }

        const FBOUND_T* feat_bounds = &clause_feat_bounds[feat_bounds_offset(clause, 0, 0)];
        const NFEAT_T* bounded_fids = &clause_feat_ids[feat_ids_offset(clause, 0)];
        int n_bounded_fids = (int)clause_n_feats[clause];

        bool is_matching = true;
        for (int base = 0; base < n_bounded_fids; base += WARP_SIZE) {
            int i = base + lane;
            if (i < n_bounded_fids) {
                FBOUND_T val = get_feature_value(Xe, 0, 0, (int)bounded_fids[i]);
                if (val < feat_bounds[i * 2] || val > feat_bounds[i * 2 + 1])
                    is_matching = false;
            }
            if (warp.any(!is_matching))
                break;
        }
        bool matched = warp.all(is_matching);

        if (lane == 0)
            clause_output[clause] = matched ? 1 : 0;
    }
}

__device__ void evaluate_conv(const ull seed, const FBOUND_T* Xe, const int8_t* clause_drop_mask,
                              const PBOUND_T* clause_position_bounds, const FBOUND_T* clause_feat_bounds,
                              const NFEAT_T* clause_feat_ids, const NFEAT_T* clause_n_feats, const int8_t* has_contra,
                              const NLITS_T* clause_len, int8_t* clause_output, NPATCHES_T* selected_patch_ids,
                              int* patch_weights) {
    auto [warp, lane, warp_id, total_warps] = warp_grid();

    WARP_STRIDE_LOOP(clause, (ull)TOTAL_CLAUSES) {
        if (clause_drop_mask[clause] == 1 || has_contra[clause]) {
            if (lane == 0)
                clause_output[clause] = 0;
            continue;
        }

        ull rng_k = rng_hash(seed, clause, 0xDEADBEEFULL);
        uint rng_counter = 0;

        if (clause_len[clause] == 0) {
            if (lane == 0) {
                int selected_id = (int)(rand_uniform(rng_k, &rng_counter) * N_PATCHES);
                clause_output[clause] = 1;
                selected_patch_ids[clause] = (NPATCHES_T)selected_id;
#if TRACK_PATCH_WEIGHTS
                patch_weights[patch_weights_offset(clause, selected_id)]++;
#endif
            }
            continue;
        }

        const PBOUND_T* pos = &clause_position_bounds[pos_bounds_offset(clause, 0)];
        const int pos0 = pos[0], pos1 = pos[1], pos2 = pos[2], pos3 = pos[3];
        const FBOUND_T* feat_bounds = &clause_feat_bounds[feat_bounds_offset(clause, 0, 0)];
        const NFEAT_T* bounded_fids = &clause_feat_ids[feat_ids_offset(clause, 0)];
        const int n_bounded_fids = (int)clause_n_feats[clause];

        int selected_id = -1;
        int count = 0;
        int n_y = pos1 - pos0 + 1;
        int n_x = pos3 - pos2 + 1;
        int patches_to_consider = n_y * n_x;

        for (int base = 0; base < patches_to_consider; base += WARP_SIZE) {
            int i = base + lane;
            bool match = false;
            if (i < patches_to_consider) {
                int py = pos0 + i / n_x;
                int px = pos2 + i % n_x;
                match = match_patch(Xe, py, px, bounded_fids, feat_bounds, n_bounded_fids);
            }

            uint ballot = warp.ballot(match);
            int n_matches = __popc(ballot);
            count += n_matches;

            if (lane == 0 && n_matches > 0 && (rand_uniform(rng_k, &rng_counter) * count) < (float)n_matches) {
                int k = (int)(rand_uniform(rng_k, &rng_counter) * n_matches);
                uint tmp = ballot;
                for (int j = 0; j < k; j++)
                    tmp &= tmp - 1;
                int selected_i = base + (__ffs(tmp) - 1);
                int py = pos0 + selected_i / n_x;
                int px = pos2 + selected_i % n_x;
                selected_id = py * N_PATCHES_X + px;
            }
        }

        if (lane == 0) {
            clause_output[clause] = (selected_id >= 0) ? 1 : 0;
            if (selected_id >= 0) {
                selected_patch_ids[clause] = (NPATCHES_T)selected_id;
#if TRACK_PATCH_WEIGHTS
                patch_weights[patch_weights_offset(clause, selected_id)]++;
#endif
            }
        }
    }
}

extern "C" __global__ void evaluate(const ull* keys, const int* e_dev, const FBOUND_T* X, const int8_t* clause_drop_mask,
                                    const PBOUND_T* clause_position_bounds, const FBOUND_T* clause_feat_bounds,
                                    const NFEAT_T* clause_feat_ids, const NFEAT_T* clause_n_feats, const int8_t* has_contra,
                                    const NLITS_T* clause_len, int8_t* clause_output, NPATCHES_T* selected_patch_ids,
                                    int* patch_weights) {
    const int e = *e_dev;
    const ull seed = keys[e];
    const FBOUND_T* Xe = &X[sample_offset(e)];
#if (N_PATCHES > 1)
    evaluate_conv(seed, Xe, clause_drop_mask, clause_position_bounds, clause_feat_bounds, clause_feat_ids,
                  clause_n_feats, has_contra, clause_len, clause_output, selected_patch_ids, patch_weights);
#else
    evaluate_noconv(Xe, clause_drop_mask, clause_feat_bounds, clause_feat_ids, clause_n_feats, has_contra,
                    clause_len, clause_output);
#endif
}
