#ifdef IS_NEOVIM_CLANGD_ENV
#include "common.cu"
#endif

__device__ void infer_clauses_conv(const int* X, int8_t* clause_outputs, const int N, const int* clause_position_bounds,
                                   const int* clause_feat_bounds, const int* bounded_feat_ids,
                                   const int* n_bounded_feats, const int* clause_density) {
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
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
        for (int base = 0; base < patches_to_consider && !found; base += 32) {
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

__device__ void infer_clauses_noconv(const int* X, int8_t* clause_outputs, const int N, const int* clause_feat_bounds,
                                     const int* bounded_feat_ids, const int* n_bounded_feats,
                                     const int* clause_density) {
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
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
        for (int base = 0; base < n_bounded_fids; base += 32) {
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

extern "C" __global__ void infer_clauses(const int* X, int8_t* clause_outputs, const int N,
                                         const int* clause_position_bounds, const int* clause_feat_bounds,
                                         const int* bounded_feat_ids, const int* n_bounded_feats,
                                         const int* clause_density) {
#if (N_PATCHES > 1)
    infer_clauses_conv(X, clause_outputs, N, clause_position_bounds, clause_feat_bounds, bounded_feat_ids,
                       n_bounded_feats, clause_density);
#else
    infer_clauses_noconv(X, clause_outputs, N, clause_feat_bounds, bounded_feat_ids, n_bounded_feats, clause_density);
#endif
}

extern "C" __global__ void sum_votes(const int8_t* clause_outputs, const float* clause_weights, const float* bias,
                                     float* class_sums, const int N) {
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
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
#if BIAS
            class_sums[e * (ull)CLASSES + class_id] = partial + bias[class_id];
#else
            class_sums[e * (ull)CLASSES + class_id] = partial;
#endif
    }
}

extern "C" __global__ void infer_clauses_patchwise(const int* X, int8_t* patch_output, const int N,
                                                   const int* clause_position_bounds, const int* clause_feat_bounds,
                                                   const int* bounded_feat_ids, const int* n_bounded_feats,
                                                   const int* clause_density) {
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
