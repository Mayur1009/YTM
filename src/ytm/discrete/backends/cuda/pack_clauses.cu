#ifdef IS_NEOVIM_CLANGD_ENV
#include "common.cu"
#endif

__device__ inline bool is_included(uint ta_state) { return ta_state >= INCLUDE_STATE; }

struct PositionResult {
    int pos0, pos1, pos2, pos3;
    uint includes;
    bool valid;
};

__device__ inline PositionResult scan_position_literals(const warp_t& warp, const uint* ta_state) {
#if POSITION_LITERALS
    int lane = warp.thread_rank();
    int pos0 = 0, pos1 = N_PATCHES_Y - 1;
    int pos2 = 0, pos3 = N_PATCHES_X - 1;
    uint includes = 0;

    for (int base = 0; base < N_POSITION_FEATS_Y; base += WARP_SIZE) {
        int lit = base + lane;
        if (lit < N_POSITION_FEATS_Y) {
            if (is_included(ta_state[lit])) {
                pos0 = max(pos0, lit + 1);
                includes++;
            }
#if NEGATED_LITERALS
            if (is_included(ta_state[lit + N_LITERALS / 2])) {
                pos1 = min(pos1, lit);
                includes++;
            }
#endif
        }
        if (warp.any(pos0 > pos1))
            return {pos0, pos1, pos2, pos3, includes, false};
    }

    for (int base = 0; base < N_POSITION_FEATS_X; base += WARP_SIZE) {
        int lit = base + lane;
        if (lit < N_POSITION_FEATS_X) {
            if (is_included(ta_state[N_POSITION_FEATS_Y + lit])) {
                pos2 = max(pos2, lit + 1);
                includes++;
            }
#if NEGATED_LITERALS
            if (is_included(ta_state[N_POSITION_FEATS_Y + lit + N_LITERALS / 2])) {
                pos3 = min(pos3, lit);
                includes++;
            }
#endif
        }
        if (warp.any(pos2 > pos3))
            return {pos0, pos1, pos2, pos3, includes, false};
    }

    pos0 = cg::reduce(warp, pos0, cg::greater<int>());
    pos1 = cg::reduce(warp, pos1, cg::less<int>());
    pos2 = cg::reduce(warp, pos2, cg::greater<int>());
    pos3 = cg::reduce(warp, pos3, cg::less<int>());
    return {pos0, pos1, pos2, pos3, includes, (pos0 <= pos1 && pos2 <= pos3)};
#else
    return {0, N_PATCHES_Y - 1, 0, N_PATCHES_X - 1, 0, true};
#endif
}

struct FeatureResult {
    int n_bounded_feats;
    uint includes;
    bool all_valid;
};

__device__ inline FeatureResult scan_feature_literals(const warp_t& warp, const uint* ta_state, const int* feat_mins,
                                                      const int* feat_maxs, const int* literal_offsets,
                                                      int* feat_bounds, int* bounded_feat_id) {
    int lane = warp.thread_rank();
    uint n_includes = 0;
    int write_offset = 0;
    bool all_valid = true;

    for (int base = 0; base < N_RAW_PATCH_FEATS; base += WARP_SIZE) {
        int fid = base + lane;
        bool in_range = (fid < N_RAW_PATCH_FEATS);
        bool is_bounded = false;
        int lb = 0, ub = 0;

        if (in_range) {
            int n_bits = literal_offsets[fid + 1] - literal_offsets[fid];
            int lstart = N_POSITION_FEATS + literal_offsets[fid];
            lb = feat_mins[fid];
            ub = feat_maxs[fid];

            for (int bit = 0; bit < n_bits; ++bit) {
                if (is_included(ta_state[lstart + bit])) {
                    lb = max(lb, feat_mins[fid] + bit + 1);
                    n_includes++;
                    is_bounded = true;
                }
#if NEGATED_LITERALS
                if (is_included(ta_state[lstart + bit + N_LITERALS / 2])) {
                    ub = min(ub, feat_mins[fid] + bit);
                    n_includes++;
                    is_bounded = true;
                }
#endif
            }
        }

        uint mask = warp.ballot(is_bounded);
        int slot = write_offset + __popc(mask & ((1u << lane) - 1));
        if (in_range) {
            feat_bounds[fid * 2 + 0] = lb;
            feat_bounds[fid * 2 + 1] = ub;
        }
        if (is_bounded) {
            bounded_feat_id[slot] = fid;
        }
        write_offset += __popc(mask);

        all_valid = warp.all(!(in_range && lb > ub));
        if (!all_valid)
            break;
    }

    return {write_offset, n_includes, all_valid};
}

extern "C" __global__ void pack_clauses(const uint* global_ta_states, const int* feat_mins, const int* feat_maxs,
                                        const int* literal_offsets, int* clause_position_bounds,
                                        int* clause_feat_bounds, int* bounded_feat_ids, int* n_bounded_feats,
                                        int* clause_density, int8_t* is_clause_synced) {
    auto warp = cg::tiled_partition<WARP_SIZE>(cg::this_thread_block());
    auto grid = cg::this_grid();
    int lane = warp.thread_rank();
    ull warp_id = grid.thread_rank() / warp.size();
    ull total_warps = grid.size() / warp.size();

    for (ull clause = warp_id; clause < (ull)TOTAL_CLAUSES; clause += total_warps) {
        if (is_clause_synced[clause])
            continue;

        const uint* ta_state = &global_ta_states[clause * (ull)N_LITERALS];
        int* pos = &clause_position_bounds[clause * 4];

        PositionResult pr = scan_position_literals(warp, ta_state);

        if (lane == 0) {
            pos[0] = pr.pos0;
            pos[1] = pr.pos1;
            pos[2] = pr.pos2;
            pos[3] = pr.pos3;
        }

        if (!pr.valid) {
            if (lane == 0) {
                clause_density[clause] = -1;
                is_clause_synced[clause] = 1;
            }
            continue;
        }

        int* cfb = &clause_feat_bounds[clause * (ull)N_RAW_PATCH_FEATS * 2];
        int* cfids = &bounded_feat_ids[clause * (ull)N_RAW_PATCH_FEATS];

        FeatureResult fr = scan_feature_literals(warp, ta_state, feat_mins, feat_maxs, literal_offsets, cfb, cfids);
        uint total_inc = cg::reduce(warp, pr.includes + fr.includes, cg::plus<uint>());

        if (lane == 0) {
            n_bounded_feats[clause] = fr.n_bounded_feats;
            clause_density[clause] = fr.all_valid ? (int)total_inc : -1;
            is_clause_synced[clause] = 1;
        }
    }
}
