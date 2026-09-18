#ifdef IS_NEOVIM_CLANGD_ENV
#include "common.h"
#include "cuda.h"
#endif

__device__ void calc_clause_outputs_conv(const FBOUND_T* X, int8_t* clause_outputs, const int N,
                                         const PBOUND_T* clause_position_bounds, const FBOUND_T* clause_feat_bounds,
                                         const NFEAT_T* clause_feat_ids, const NFEAT_T* clause_n_feats,
                                         const int8_t* has_contra, const NLITS_T* clause_len) {
    auto warp = cg::tiled_partition<WARP_SIZE>(cg::this_thread_block());
    auto grid = cg::this_grid();
    int lane = warp.thread_rank();
    ull warp_id = grid.thread_rank() / warp.size();
    ull total_warps = grid.size() / warp.size();
    ull total_work = (ull)N * TOTAL_CLAUSES;

    for (ull idx = warp_id; idx < total_work; idx += total_warps) {
        ull e = idx / (ull)TOTAL_CLAUSES;
        ull clause = idx % (ull)TOTAL_CLAUSES;

        if (has_contra[clause] || clause_len[clause] == 0) {
            if (lane == 0)
                clause_outputs[idx] = has_contra[clause] ? 0 : 1;
            continue;
        }

        const FBOUND_T* Xe = &X[e * (ull)HEIGHT * WIDTH * DEPTH];
        const PBOUND_T* pos = &clause_position_bounds[clause * 4];
        int pos0 = pos[0], pos1 = pos[1], pos2 = pos[2], pos3 = pos[3];
        const FBOUND_T* cfb = &clause_feat_bounds[clause * (ull)N_RAW_PATCH_FEATS * 2];
        const NFEAT_T* bounded_fids = &clause_feat_ids[clause * (ull)N_RAW_PATCH_FEATS];
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
    auto warp = cg::tiled_partition<WARP_SIZE>(cg::this_thread_block());
    auto grid = cg::this_grid();
    int lane = warp.thread_rank();
    ull warp_id = grid.thread_rank() / warp.size();
    ull total_warps = grid.size() / warp.size();
    ull total_work = (ull)N * TOTAL_CLAUSES;

    for (ull idx = warp_id; idx < total_work; idx += total_warps) {
        ull e = idx / (ull)TOTAL_CLAUSES;
        ull clause = idx % (ull)TOTAL_CLAUSES;

        if (has_contra[clause] || clause_len[clause] == 0) {
            if (lane == 0) {
                clause_outputs[idx] = has_contra[clause] ? 0 : 1;
            }
            continue;
        }

        const FBOUND_T* Xe = &X[e * (ull)HEIGHT * WIDTH * DEPTH];
        const FBOUND_T* cfb = &clause_feat_bounds[clause * (ull)N_RAW_PATCH_FEATS * 2];
        const NFEAT_T* bounded_fids = &clause_feat_ids[clause * (ull)N_RAW_PATCH_FEATS];
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

extern "C" __global__ void sum_votes(const int8_t* clause_outputs, const float* clause_weights, float* class_sums,
                                     const int N) {
    auto warp = cg::tiled_partition<WARP_SIZE>(cg::this_thread_block());
    auto grid = cg::this_grid();
    int lane = warp.thread_rank();
    ull warp_id = grid.thread_rank() / warp.size();
    ull total_warps = grid.size() / warp.size();
    ull total_work = (ull)N * CLASSES;

    for (ull idx = warp_id; idx < total_work; idx += total_warps) {
        ull e = idx / (ull)CLASSES;
        ull class_id = idx % (ull)CLASSES;

        const int8_t* co = &clause_outputs[e * (ull)TOTAL_CLAUSES];
        const float* cw = &clause_weights[class_id * (ull)CLAUSES_PER_CLASS];
        ull clause_base = class_id * (ull)CLAUSES_PER_CLASS;

        float partial = 0.0f;
        for (int c = lane; c < CLAUSES_PER_CLASS; c += (int)warp.size()) {
#if COALESCED == 0
            if (co[clause_base + c])
                partial += cw[c];
#else
            if (co[c])
                partial += cw[c];
#endif
        }

        partial = cg::reduce(warp, partial, cg::plus<float>());

        if (lane == 0)
            class_sums[e * (ull)CLASSES + class_id] = partial;
    }
}

extern "C" __global__ void calc_clause_outputs_patchwise(const FBOUND_T* X, int8_t* patch_output, const int N,
                                                         const PBOUND_T* clause_position_bounds,
                                                         const FBOUND_T* clause_feat_bounds, const NFEAT_T* clause_feat_ids,
                                                         const NFEAT_T* clause_n_feats, const int8_t* has_contra,
                                                         const NLITS_T* clause_len) {
    ull tid = threadIdx.x + blockIdx.x * blockDim.x;
    ull stride = blockDim.x * gridDim.x;

    for (ull idx = tid; idx < (ull)N * TOTAL_CLAUSES * N_PATCHES; idx += stride) {
        ull e = idx / ((ull)TOTAL_CLAUSES * N_PATCHES);
        ull clause_patch = idx % ((ull)TOTAL_CLAUSES * N_PATCHES);
        ull clause = clause_patch / (ull)N_PATCHES;
        int patch = clause_patch % N_PATCHES;

        int8_t* output = &patch_output[e * (ull)TOTAL_CLAUSES * N_PATCHES + clause * (ull)N_PATCHES + patch];

        if (has_contra[clause] || clause_len[clause] == 0) {
            *output = has_contra[clause] ? 0 : 1;
            continue;
        }

        int py = patch / N_PATCHES_X;
        int px = patch % N_PATCHES_X;

#if (N_PATCHES > 1)
        const PBOUND_T* pos = &clause_position_bounds[clause * 4];
        if (py < pos[0] || py > pos[1] || px < pos[2] || px > pos[3]) {
            *output = 0;
            continue;
        }
#endif

        const FBOUND_T* Xe = &X[e * (ull)HEIGHT * WIDTH * DEPTH];
        const FBOUND_T* cfb = &clause_feat_bounds[clause * (ull)N_RAW_PATCH_FEATS * 2];
        const NFEAT_T* bounded_fids = &clause_feat_ids[clause * (ull)N_RAW_PATCH_FEATS];
        int n_bounded_fids = (int)clause_n_feats[clause];

        *output = match_patch(Xe, py, px, bounded_fids, cfb, n_bounded_fids) ? 1 : 0;
    }
}

__device__ void evaluate_noconv(const FBOUND_T* Xe, const int8_t* clause_drop_mask, const FBOUND_T* clause_feat_bounds,
                                const NFEAT_T* clause_feat_ids, const NFEAT_T* clause_n_feats, const int8_t* has_contra,
                                const NLITS_T* clause_len, int8_t* clause_output) {
    auto warp = cg::tiled_partition<WARP_SIZE>(cg::this_thread_block());
    auto grid = cg::this_grid();
    int lane = warp.thread_rank();
    ull warp_id = grid.thread_rank() / warp.size();
    ull total_warps = grid.size() / warp.size();

    for (ull clause = warp_id; clause < (ull)TOTAL_CLAUSES; clause += total_warps) {
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

        const FBOUND_T* feat_bounds = &clause_feat_bounds[clause * (ull)N_RAW_PATCH_FEATS * 2];
        const NFEAT_T* bounded_fids = &clause_feat_ids[clause * (ull)N_RAW_PATCH_FEATS];
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
    auto warp = cg::tiled_partition<WARP_SIZE>(cg::this_thread_block());
    auto grid = cg::this_grid();
    int lane = warp.thread_rank();
    ull warp_id = grid.thread_rank() / warp.size();
    ull total_warps = grid.size() / warp.size();

    for (ull clause = warp_id; clause < (ull)TOTAL_CLAUSES; clause += total_warps) {
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
                patch_weights[clause * (ull)N_PATCHES + selected_id]++;
#endif
            }
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
                patch_weights[clause * (ull)N_PATCHES + selected_id]++;
#endif
            }
        }
    }
}

extern "C" __global__ void evaluate(const ull seed, const FBOUND_T* X, const int e, const int8_t* clause_drop_mask,
                                    const PBOUND_T* clause_position_bounds, const FBOUND_T* clause_feat_bounds,
                                    const NFEAT_T* clause_feat_ids, const NFEAT_T* clause_n_feats, const int8_t* has_contra,
                                    const NLITS_T* clause_len, int8_t* clause_output, NPATCHES_T* selected_patch_ids,
                                    int* patch_weights) {
    const FBOUND_T* Xe = &X[(ull)e * HEIGHT * WIDTH * DEPTH];
#if (N_PATCHES > 1)
    evaluate_conv(seed, Xe, clause_drop_mask, clause_position_bounds, clause_feat_bounds, clause_feat_ids,
                  clause_n_feats, has_contra, clause_len, clause_output, selected_patch_ids, patch_weights);
#else
    evaluate_noconv(Xe, clause_drop_mask, clause_feat_bounds, clause_feat_ids, clause_n_feats, has_contra,
                    clause_len, clause_output);
#endif
}

extern "C" __global__ void count_votes(const int8_t* clause_output, const float* clause_weights, float* votes) {
    auto warp = cg::tiled_partition<WARP_SIZE>(cg::this_thread_block());
    auto grid = cg::this_grid();
    int lane = warp.thread_rank();
    ull warp_id = grid.thread_rank() / warp.size();
    ull total_warps = grid.size() / warp.size();

    for (ull class_id = warp_id; class_id < (ull)CLASSES; class_id += total_warps) {
        const float* cw = &clause_weights[class_id * (ull)CLAUSES_PER_CLASS];
        float partial = 0.0f;

        for (int c = lane; c < CLAUSES_PER_CLASS; c += (int)warp.size()) {
#if COALESCED == 0
            ull clause = class_id * (ull)CLAUSES_PER_CLASS + c;
#else
            ull clause = c;
#endif
            if (clause_output[clause])
                partial += cw[c];
        }

        partial = cg::reduce(warp, partial, cg::plus<float>());

        if (lane == 0)
            votes[class_id] = partial;
    }
}
