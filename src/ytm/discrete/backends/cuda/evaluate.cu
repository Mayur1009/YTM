#ifdef IS_NEOVIM_CLANGD_ENV
#include "common.cu"
#endif

__device__ void evaluate_noconv(const int* X, const int e, const int8_t* clause_drop_mask,
                                const int* clause_feat_bounds, const int* bounded_feat_ids, const int* n_bounded_feats,
                                const int* clause_density, int* selected_patch_ids) {
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    auto grid = cg::this_grid();
    int lane = warp.thread_rank();
    ull warp_id = grid.thread_rank() / warp.size();
    ull total_warps = grid.size() / warp.size();

    const int* Xe = &X[(ull)e * HEIGHT * WIDTH * DEPTH];
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

        bool my_match = true;
        for (int base = 0; base < n_bounded_fids; base += 32) {
            int i = base + lane;
            if (i < n_bounded_fids) {
                int fid = bounded_fids[i];
                int val = get_feature_value(Xe, 0, 0, fid);
                if (val < feat_bounds[fid * 2] || val > feat_bounds[fid * 2 + 1])
                    my_match = false;
            }
            if (warp.any(!my_match))
                break;
        }
        bool matched = warp.all(my_match);

        if (lane == 0)
            selected_patch_ids[clause] = matched ? 0 : -1;
    }
}

__device__ void evaluate_conv(const int* X, const int e, const int8_t* clause_drop_mask,
                              const int* clause_position_bounds, const int* clause_feat_bounds,
                              const int* bounded_feat_ids, const int* n_bounded_feats, const int* clause_density,
                              const ull seed, int* selected_patch_ids, int* patch_weights) {
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    auto grid = cg::this_grid();
    int lane = warp.thread_rank();
    ull warp_id = grid.thread_rank() / warp.size();
    ull total_warps = grid.size() / warp.size();

    const int* Xe = &X[(ull)e * HEIGHT * WIDTH * DEPTH];
    ull rng_k = rng_hash(seed, warp_id, (ull)e, 0xDEADBEEFULL);
    uint rng_counter = 0;

    for (ull clause = warp_id; clause < (ull)TOTAL_CLAUSES; clause += total_warps) {
        int cd = clause_density[clause];
        if (clause_drop_mask[clause] == 1 || cd < 0) {
            if (lane == 0)
                selected_patch_ids[clause] = -1;
            continue;
        }

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

        for (int base = 0; base < patches_to_consider; base += 32) {
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
                int bit = __ffs(tmp) - 1;
                int selected_i = base + bit;
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

extern "C" __global__ void evaluate(const int* X, const int e, const int8_t* clause_drop_mask,
                                    const int* clause_position_bounds, const int* clause_feat_bounds,
                                    const int* bounded_feat_ids, const int* n_bounded_feats, const int* clause_density,
                                    const ull seed, int* selected_patch_ids, int* patch_weights) {
#if (N_PATCHES > 1)
    evaluate_conv(X, e, clause_drop_mask, clause_position_bounds, clause_feat_bounds, bounded_feat_ids, n_bounded_feats,
                  clause_density, seed, selected_patch_ids, patch_weights);
#else
    evaluate_noconv(X, e, clause_drop_mask, clause_feat_bounds, bounded_feat_ids, n_bounded_feats, clause_density,
                    selected_patch_ids);
#endif
}

extern "C" __global__ void count_votes(const int* selected_patch_ids, const float* clause_weights, float* votes) {
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
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
