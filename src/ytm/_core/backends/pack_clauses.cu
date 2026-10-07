#ifdef IS_NEOVIM_CLANGD_ENV
#include "common.h"
#include "cuda.h"
#include "pack_clauses.h"
#endif

__device__ inline PositionResult scan_position_literals(const warp_t& warp, const TA_STATE_T* ta_state, int dont_skip) {
    int lane = warp.thread_rank();
    int pos0 = 0, pos1 = N_PATCHES_Y - 1;
    int pos2 = 0, pos3 = N_PATCHES_X - 1;
    uint includes = 0;
    bool valid = true;

    for (int base = 0; base < N_POSITION_FEATS_Y; base += WARP_SIZE) {
        int lit = base + lane;
        if (lit < N_POSITION_FEATS_Y)
            update_pos_bounds(ta_state, lit, 0, &pos0, &pos1, &includes);
        if (warp.any(pos0 > pos1)) {
            valid = false;
            if (!dont_skip)
                return {pos0, pos1, pos2, pos3, includes, false};
        }
    }

    for (int base = 0; base < N_POSITION_FEATS_X; base += WARP_SIZE) {
        int lit = base + lane;
        if (lit < N_POSITION_FEATS_X)
            update_pos_bounds(ta_state, lit, N_POSITION_FEATS_Y, &pos2, &pos3, &includes);
        if (warp.any(pos2 > pos3)) {
            valid = false;
            if (!dont_skip)
                return {pos0, pos1, pos2, pos3, includes, false};
        }
    }

    pos0 = cg::reduce(warp, pos0, cg::greater<int>());
    pos1 = cg::reduce(warp, pos1, cg::less<int>());
    pos2 = cg::reduce(warp, pos2, cg::greater<int>());
    pos3 = cg::reduce(warp, pos3, cg::less<int>());
    return {pos0, pos1, pos2, pos3, includes, (valid && pos0 <= pos1 && pos2 <= pos3)};
}

// Features each lane scans per round. More independent loads in flight per lane and fewer warp syncs per feature,
// a warp otherwise mostly waits on the latency of two one byte loads per feature. Only as many as keeps every lane
// busy: with few features per patch one feature per lane is faster.
#ifndef PACK_FEATS_PER_LANE_MAX
#define PACK_FEATS_PER_LANE_MAX 8
#endif
#define PACK_FEATS_PER_LANE                                                                                            \
    (N_RAW_PATCH_FEATS / WARP_SIZE < 1                          ? 1                                                    \
     : N_RAW_PATCH_FEATS / WARP_SIZE > PACK_FEATS_PER_LANE_MAX ? PACK_FEATS_PER_LANE_MAX                              \
                                                                : N_RAW_PATCH_FEATS / WARP_SIZE)

__device__ inline FeatureResult scan_feature_literals(const warp_t& warp, const TA_STATE_T* ta_state,
                                                      const FBOUND_T* therm_bits, const NLITS_T* literal_offsets,
                                                      NFEAT_T* feat_ids, FBOUND_T* feat_bounds, int dont_skip) {
    int lane = warp.thread_rank();
    uint n_includes = 0;
    int write_offset = 0;
    bool all_valid = true;

    // Each lane takes PACK_FEATS_PER_LANE consecutive features, so lane order then feature order is fid order and
    // the compacted list keeps the same order as one feature per lane.
    for (int base = 0; base < N_RAW_PATCH_FEATS; base += WARP_SIZE * PACK_FEATS_PER_LANE) {
        const int first = base + lane * PACK_FEATS_PER_LANE;
        FeatScan fs[PACK_FEATS_PER_LANE];
        int n_bounded = 0;
        bool lane_valid = true;

#pragma unroll
        for (int k = 0; k < PACK_FEATS_PER_LANE; ++k) {
            fs[k] = FeatScan{};
            if (first + k < N_RAW_PATCH_FEATS) {
                fs[k] = calc_single_feat_bounds(ta_state, therm_bits, literal_offsets, first + k);
                n_includes += fs[k].includes;
                n_bounded += fs[k].is_bounded ? 1 : 0;
                lane_valid = lane_valid && !(fs[k].lb > fs[k].ub);
            }
        }

        int slot = write_offset + cg::exclusive_scan(warp, n_bounded, cg::plus<int>());
        write_offset += cg::reduce(warp, n_bounded, cg::plus<int>());
#pragma unroll
        for (int k = 0; k < PACK_FEATS_PER_LANE; ++k) {
            if (first + k < N_RAW_PATCH_FEATS && fs[k].is_bounded) {
                feat_ids[slot] = (NFEAT_T)(first + k);
                feat_bounds[slot * 2 + 0] = fs[k].lb;
                feat_bounds[slot * 2 + 1] = fs[k].ub;
                ++slot;
            }
        }

        all_valid = all_valid && warp.all(lane_valid);
        if (!all_valid && !dont_skip)
            break;
    }

    return {write_offset, n_includes, all_valid};
}

extern "C" __global__ void pack_clauses(const TA_STATE_T* global_ta_states, const FBOUND_T* therm_bits,
                                        const NLITS_T* literal_offsets, PBOUND_T* clause_position_bounds,
                                        FBOUND_T* clause_feat_bounds, NFEAT_T* clause_feat_ids, NFEAT_T* clause_n_feats,
                                        int8_t* has_contra, NLITS_T* clause_len, int8_t* is_clause_synced,
                                        int dont_skip) {
    auto [warp, lane, warp_id, total_warps] = warp_grid();

    WARP_STRIDE_LOOP(clause, (ull)TOTAL_CLAUSES) {
        if (is_clause_synced[clause])
            continue;

        const TA_STATE_T* ta_state = &global_ta_states[ta_offset(clause, 0)];

#if POSITION_LITERALS
        PositionResult pr = scan_position_literals(warp, ta_state, dont_skip);
#else
        PositionResult pr = {0, N_PATCHES_Y - 1, 0, N_PATCHES_X - 1, 0, true};
#endif

#if (N_PATCHES > 1)
        if (lane == 0) {
            PBOUND_T* pos = &clause_position_bounds[pos_bounds_offset(clause, 0)];
            pos[0] = pr.pos0;
            pos[1] = pr.pos1;
            pos[2] = pr.pos2;
            pos[3] = pr.pos3;
        }
#endif

        if (!pr.valid && !dont_skip) {
            if (lane == 0) {
                has_contra[clause] = 1;
                clause_len[clause] = 0;
                is_clause_synced[clause] = 1;
            }
            continue;
        }

        NFEAT_T* cfids = &clause_feat_ids[feat_ids_offset(clause, 0)];
        FBOUND_T* cfb = &clause_feat_bounds[feat_bounds_offset(clause, 0, 0)];

        FeatureResult fr = scan_feature_literals(warp, ta_state, therm_bits, literal_offsets, cfids, cfb, dont_skip);
        uint total_inc = cg::reduce(warp, pr.includes + fr.includes, cg::plus<uint>());

        if (lane == 0) {
            clause_n_feats[clause] = (NFEAT_T)fr.n_bounded_feats;
            has_contra[clause] = !(pr.valid && fr.all_valid);
            clause_len[clause] = (NLITS_T)total_inc;
            is_clause_synced[clause] = 1;
        }
    }
}
