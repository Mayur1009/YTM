#ifdef IS_NEOVIM_CLANGD_ENV
#include "common.h"
#include "cuda.h"
#endif

__device__ void calc_clause_outputs_conv(const int* X, int8_t* clause_outputs, const int N,
                                         const int* clause_position_bounds, const int* clause_feat_bounds,
                                         const int* bounded_feat_ids, const int* n_bounded_feats,
                                         const int* clause_density) {
    auto warp = cg::tiled_partition<WARP_SIZE>(cg::this_thread_block());
    auto grid = cg::this_grid();
    int lane = warp.thread_rank();
    ull warp_id = grid.thread_rank() / warp.size();
    ull total_warps = grid.size() / warp.size();
    ull total_work = (ull)N * TOTAL_CLAUSES;

    for (ull idx = warp_id; idx < total_work; idx += total_warps) {
        ull e = idx / (ull)TOTAL_CLAUSES;
        ull clause = idx % (ull)TOTAL_CLAUSES;

        int cd = clause_density[clause];
        if (cd <= 0) {
            if (lane == 0)
                clause_outputs[idx] = (cd == 0) ? 1 : 0;
            continue;
        }

        const int* Xe = &X[e * (ull)HEIGHT * WIDTH * DEPTH];
        const int* pos = &clause_position_bounds[clause * 4];
        int pos0 = pos[0], pos1 = pos[1], pos2 = pos[2], pos3 = pos[3];
        const int* cfb = &clause_feat_bounds[clause * (ull)N_RAW_PATCH_FEATS * 2];
        const int* bounded_fids = &bounded_feat_ids[clause * (ull)N_RAW_PATCH_FEATS];
        int n_bounded_fids = n_bounded_feats[clause];

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
                match = match_patch(Xe, py, px, cfb, bounded_fids, n_bounded_fids);
            }
            if (warp.any(match))
                found = true;
        }

        if (lane == 0)
            clause_outputs[idx] = found ? 1 : 0;
    }
}

__device__ void calc_clause_outputs_noconv(const int* X, int8_t* clause_outputs, const int N,
                                           const int* clause_feat_bounds, const int* bounded_feat_ids,
                                           const int* n_bounded_feats, const int* clause_density) {
    auto warp = cg::tiled_partition<WARP_SIZE>(cg::this_thread_block());
    auto grid = cg::this_grid();
    int lane = warp.thread_rank();
    ull warp_id = grid.thread_rank() / warp.size();
    ull total_warps = grid.size() / warp.size();
    ull total_work = (ull)N * TOTAL_CLAUSES;

    for (ull idx = warp_id; idx < total_work; idx += total_warps) {
        ull e = idx / (ull)TOTAL_CLAUSES;
        ull clause = idx % (ull)TOTAL_CLAUSES;

        int cd = clause_density[clause];
        if (cd <= 0) {
            if (lane == 0) {
                clause_outputs[idx] = (cd == 0) ? 1 : 0;
            }
            continue;
        }

        const int* Xe = &X[e * (ull)HEIGHT * WIDTH * DEPTH];
        const int* cfb = &clause_feat_bounds[clause * (ull)N_RAW_PATCH_FEATS * 2];
        const int* bounded_fids = &bounded_feat_ids[clause * (ull)N_RAW_PATCH_FEATS];
        int n_bounded_fids = n_bounded_feats[clause];

        bool is_matching = true;
        for (int base = 0; base < n_bounded_fids; base += WARP_SIZE) {
            int i = base + lane;
            if (i < n_bounded_fids) {
                int fid = bounded_fids[i];
                int val = get_feature_value(Xe, 0, 0, fid);
                if (val < cfb[fid * 2] || val > cfb[fid * 2 + 1])
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

extern "C" __global__ void calc_clause_outputs(const int* X, int8_t* clause_outputs, const int N,
                                               const int* clause_position_bounds, const int* clause_feat_bounds,
                                               const int* bounded_feat_ids, const int* n_bounded_feats,
                                               const int* clause_density) {
#if (N_PATCHES > 1)
    calc_clause_outputs_conv(X, clause_outputs, N, clause_position_bounds, clause_feat_bounds, bounded_feat_ids,
                             n_bounded_feats, clause_density);
#else
    calc_clause_outputs_noconv(X, clause_outputs, N, clause_feat_bounds, bounded_feat_ids, n_bounded_feats,
                               clause_density);
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

extern "C" __global__ void calc_clause_outputs_patchwise(const int* X, int8_t* patch_output, const int N,
                                                         const int* clause_position_bounds,
                                                         const int* clause_feat_bounds, const int* bounded_feat_ids,
                                                         const int* n_bounded_feats, const int* clause_density) {
    ull tid = threadIdx.x + blockIdx.x * blockDim.x;
    ull stride = blockDim.x * gridDim.x;

    for (ull idx = tid; idx < (ull)N * TOTAL_CLAUSES * N_PATCHES; idx += stride) {
        ull e = idx / ((ull)TOTAL_CLAUSES * N_PATCHES);
        ull clause_patch = idx % ((ull)TOTAL_CLAUSES * N_PATCHES);
        ull clause = clause_patch / (ull)N_PATCHES;
        int patch = clause_patch % N_PATCHES;

        int8_t* output = &patch_output[e * (ull)TOTAL_CLAUSES * N_PATCHES + clause * (ull)N_PATCHES + patch];

        int cd = clause_density[clause];
        if (cd <= 0) {
            *output = (cd == 0) ? 1 : 0;
            continue;
        }

        const int* pos = &clause_position_bounds[clause * 4];
        int py = patch / N_PATCHES_X;
        int px = patch % N_PATCHES_X;

        if (py < pos[0] || py > pos[1] || px < pos[2] || px > pos[3]) {
            *output = 0;
            continue;
        }

        const int* Xe = &X[e * (ull)HEIGHT * WIDTH * DEPTH];
        const int* cfb = &clause_feat_bounds[clause * (ull)N_RAW_PATCH_FEATS * 2];
        const int* bounded_fids = &bounded_feat_ids[clause * (ull)N_RAW_PATCH_FEATS];
        int n_bounded_fids = n_bounded_feats[clause];

        *output = match_patch(Xe, py, px, cfb, bounded_fids, n_bounded_fids) ? 1 : 0;
    }
}

__device__ void evaluate_noconv(const int* Xe, const int8_t* clause_drop_mask, const int* clause_feat_bounds,
                                const int* bounded_feat_ids, const int* n_bounded_feats, const int* clause_density,
                                int* selected_patch_ids) {
    auto warp = cg::tiled_partition<WARP_SIZE>(cg::this_thread_block());
    auto grid = cg::this_grid();
    int lane = warp.thread_rank();
    ull warp_id = grid.thread_rank() / warp.size();
    ull total_warps = grid.size() / warp.size();

    for (ull clause = warp_id; clause < (ull)TOTAL_CLAUSES; clause += total_warps) {
        int cd = clause_density[clause];
        if (clause_drop_mask[clause] == 1 || cd < 0) {
            if (lane == 0)
                selected_patch_ids[clause] = -1;
            continue;
        }

        if (cd == 0) {
            if (lane == 0)
                selected_patch_ids[clause] = 0;
            continue;
        }

        const int* feat_bounds = &clause_feat_bounds[clause * (ull)N_RAW_PATCH_FEATS * 2];
        const int* bounded_fids = &bounded_feat_ids[clause * (ull)N_RAW_PATCH_FEATS];
        int n_bounded_fids = n_bounded_feats[clause];

        bool is_matching = true;
        for (int base = 0; base < n_bounded_fids; base += WARP_SIZE) {
            int i = base + lane;
            if (i < n_bounded_fids) {
                int fid = bounded_fids[i];
                int val = get_feature_value(Xe, 0, 0, fid);
                if (val < feat_bounds[fid * 2] || val > feat_bounds[fid * 2 + 1])
                    is_matching = false;
            }
            if (warp.any(!is_matching))
                break;
        }
        bool matched = warp.all(is_matching);

        if (lane == 0)
            selected_patch_ids[clause] = matched ? 0 : -1;
    }
}

__device__ void evaluate_conv(const ull seed, const int* Xe, const int8_t* clause_drop_mask,
                              const int* clause_position_bounds, const int* clause_feat_bounds,
                              const int* bounded_feat_ids, const int* n_bounded_feats, const int* clause_density,
                              int* selected_patch_ids, int* patch_weights) {
    auto warp = cg::tiled_partition<WARP_SIZE>(cg::this_thread_block());
    auto grid = cg::this_grid();
    int lane = warp.thread_rank();
    ull warp_id = grid.thread_rank() / warp.size();
    ull total_warps = grid.size() / warp.size();

    for (ull clause = warp_id; clause < (ull)TOTAL_CLAUSES; clause += total_warps) {
        int cd = clause_density[clause];
        if (clause_drop_mask[clause] == 1 || cd < 0) {
            if (lane == 0)
                selected_patch_ids[clause] = -1;
            continue;
        }

        ull rng_k = rng_hash(seed, clause, 0xDEADBEEFULL);
        uint rng_counter = 0;

        if (cd == 0) {
            if (lane == 0) {
                int selected_id = (int)(rand_uniform(rng_k, &rng_counter) * N_PATCHES);
                selected_patch_ids[clause] = selected_id;
#if TRACK_PATCH_WEIGHTS
                patch_weights[clause * (ull)N_PATCHES + selected_id]++;
#endif
            }
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
        int patches_to_consider = n_y * n_x;

        for (int base = 0; base < patches_to_consider; base += WARP_SIZE) {
            int i = base + lane;
            bool match = false;
            if (i < patches_to_consider) {
                int py = pos0 + i / n_x;
                int px = pos2 + i % n_x;
                match = match_patch(Xe, py, px, feat_bounds, bounded_fids, n_bounded_fids);
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
            selected_patch_ids[clause] = selected_id;
#if TRACK_PATCH_WEIGHTS
            if (selected_id >= 0)
                patch_weights[clause * (ull)N_PATCHES + selected_id]++;
#endif
        }
    }
}

extern "C" __global__ void evaluate(const ull seed, const int* X, const int e, const int8_t* clause_drop_mask,
                                    const int* clause_position_bounds, const int* clause_feat_bounds,
                                    const int* bounded_feat_ids, const int* n_bounded_feats, const int* clause_density,
                                    int* selected_patch_ids, int* patch_weights) {
    const int* Xe = &X[(ull)e * HEIGHT * WIDTH * DEPTH];
#if (N_PATCHES > 1)
    evaluate_conv(seed, Xe, clause_drop_mask, clause_position_bounds, clause_feat_bounds, bounded_feat_ids,
                  n_bounded_feats, clause_density, selected_patch_ids, patch_weights);
#else
    evaluate_noconv(Xe, clause_drop_mask, clause_feat_bounds, bounded_feat_ids, n_bounded_feats, clause_density,
                    selected_patch_ids);
#endif
}

extern "C" __global__ void count_votes(const int* selected_patch_ids, const float* clause_weights, float* votes) {
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
            if (selected_patch_ids[clause] >= 0)
                partial += cw[c];
        }

        partial = cg::reduce(warp, partial, cg::plus<float>());

        if (lane == 0)
            votes[class_id] = partial;
    }
}
