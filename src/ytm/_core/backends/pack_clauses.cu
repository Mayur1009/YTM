#ifdef IS_NEOVIM_CLANGD_ENV
#include "common.h"
#include "cuda.h"
#include "pack_clauses.h"
#endif

// X or Y position literal scan. Returns false on contradiction.
__device__ inline bool scan_position_axis(const warp_t& warp, const TA_STATE_T* __restrict__ ta_state, int off, int n,
                                          int* __restrict__ lo, int* __restrict__ hi, uint* __restrict__ includes,
                                          bool stop) {
    const int lane = (int)warp.thread_rank();

    for (int base = 0; base < n; base += WARP_SIZE) {
        const int lit = base + lane;                                    // One lane per literal
        const bool in = lit < n;                                        // check if lane is in the bounds.
        uint pos = warp.ballot(in && is_included(ta_state[off + lit])); // Sync lanes
        uint neg = NEGATED_LITERALS ? warp.ballot(in && is_included(ta_state[off + lit + N_LITERALS / 2])) : 0u;
        if (pos)
            *lo = base + WARP_SIZE - __clz(pos); // __clz(Count leading zero) gives the lower bound.
        if (neg && *hi == n)
            *hi = base + __ffs(neg) - 1;        // __ffs(Find first set) on the negations gives the upper bound.
        *includes += __popc(pos) + __popc(neg); // Count the total includes in the pos and neg using pop-count
        if (stop && *lo > *hi)                  // If the positions are contradictory, then stop
            return false;
    }
    return *lo <= *hi;
}

// Find the position bounds of a clause.
__device__ inline PositionResult scan_position_literals(const warp_t& warp, const TA_STATE_T* __restrict__ ta_state,
                                                        int dont_skip) {
    int pos0 = 0, pos1 = N_POSITION_FEATS_Y;
    int pos2 = 0, pos3 = N_POSITION_FEATS_X;
    uint includes = 0;
    // Scan Y portion of the position literals
    bool y_ok = scan_position_axis(warp, ta_state, 0, N_POSITION_FEATS_Y, &pos0, &pos1, &includes, !dont_skip);
    bool x_ok = false;
    if (dont_skip || y_ok)
        // Scan X portion of the position literals.
        x_ok = scan_position_axis(warp, ta_state, N_POSITION_FEATS_Y, N_POSITION_FEATS_X, &pos2, &pos3, &includes,
                                  !dont_skip);
    return {pos0, pos1, pos2, pos3, includes, y_ok && x_ok}; // Includes is the warp total, not per lane.
}

__device__ inline FeatureResult scan_feature_literals(const warp_t& warp, const TA_STATE_T* __restrict__ ta_state,
                                                      const FBOUND_T* __restrict__ therm_bits,
                                                      const NLITS_T* __restrict__ literal_offsets,
                                                      NFEAT_T* __restrict__ feat_ids,
                                                      FBOUND_T* __restrict__ feat_bounds, int dont_skip) {
    int lane = warp.thread_rank();
    uint n_includes = 0;
    int write_offset = 0;
    bool all_valid = true;

    // One feature per lane. Bounded features are compacted in fid order with a ballot.
    const uint lanes_before = (1u << lane) - 1u; // Bitmask setting bit 1 for all the lanes before this one.
    for (int base = 0; base < N_RAW_PATCH_FEATS; base += WARP_SIZE) {
        int fid = base + lane;
        bool in_range = (fid < N_RAW_PATCH_FEATS);

        FeatScan fs = {};
        if (in_range) {
            // Calculate bound for a single feature
            fs = calc_single_feat_bounds(ta_state, therm_bits, literal_offsets, fid);
            n_includes += fs.includes;
        }

        uint bounded = warp.ballot(fs.is_bounded); // Sync lanes, each bit set meaning that lane found a bounded feature.
        if (fs.is_bounded) {
            // Use lanes before and bounded to calculate which slot should be filled in the compacted lists.
            int slot = write_offset + __popc(bounded & lanes_before);
            feat_ids[slot] = (NFEAT_T)fid;
            feat_bounds[slot * 2 + 0] = fs.lb;
            feat_bounds[slot * 2 + 1] = fs.ub;
        }
        write_offset += __popc(bounded);

        all_valid = all_valid && warp.all(!(in_range && fs.lb > fs.ub));
        if (!all_valid && !dont_skip)
            break;
    }

    return {write_offset, n_includes, all_valid};
}

extern "C" __global__ void
pack_clauses(const TA_STATE_T* __restrict__ global_ta_states, const FBOUND_T* __restrict__ therm_bits,
             const NLITS_T* __restrict__ literal_offsets, PBOUND_T* __restrict__ clause_position_bounds,
             FBOUND_T* __restrict__ clause_feat_bounds, NFEAT_T* __restrict__ clause_feat_ids,
             NFEAT_T* __restrict__ clause_n_feats, int8_t* __restrict__ has_contra, NLITS_T* __restrict__ clause_len,
             int8_t* __restrict__ is_clause_synced, int dont_skip) {
    auto [warp, lane, warp_id, total_warps] = warp_grid();

    WARP_STRIDE_LOOP(clause, (ull)TOTAL_CLAUSES) {
        if (is_clause_synced[clause])
            continue;

        const TA_STATE_T* ta_state = &global_ta_states[ta_offset(clause, 0)];

        PositionResult pr = {0, N_PATCHES_Y - 1, 0, N_PATCHES_X - 1, 0, true};
        if (POSITION_LITERALS)
            pr = scan_position_literals(warp, ta_state, dont_skip);

        if (N_PATCHES > 1 && lane == 0) {
            PBOUND_T* pos = &clause_position_bounds[pos_bounds_offset(clause, 0)];
            pos[0] = pr.pos0;
            pos[1] = pr.pos1;
            pos[2] = pr.pos2;
            pos[3] = pr.pos3;
        }

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
        uint total_inc = pr.includes + cg::reduce(warp, fr.includes, cg::plus<uint>());

        if (lane == 0) {
            clause_n_feats[clause] = (NFEAT_T)fr.n_bounded_feats;
            has_contra[clause] = !(pr.valid && fr.all_valid);
            clause_len[clause] = (NLITS_T)total_inc;
            is_clause_synced[clause] = 1;
        }
    }
}
