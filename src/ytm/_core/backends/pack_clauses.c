#ifdef IS_NEOVIM_CLANGD_ENV
#include "common.h"
#endif

typedef struct {
    int pos0, pos1, pos2, pos3;
    uint includes;
    bool valid;
} PositionResult;

static inline PositionResult scan_position_literals(const uint* ta_state, int full) {
#if POSITION_LITERALS
    int pos0 = 0, pos1 = N_PATCHES_Y - 1;
    int pos2 = 0, pos3 = N_PATCHES_X - 1;
    uint includes = 0;
    bool valid = true;

    for (int lit = 0; lit < N_POSITION_FEATS_Y; ++lit) {
        if (is_included(ta_state[lit])) {
            if (lit + 1 > pos0)
                pos0 = lit + 1;
            includes++;
        }
#if NEGATED_LITERALS
        if (is_included(ta_state[lit + N_LITERALS / 2])) {
            if (lit < pos1)
                pos1 = lit;
            includes++;
        }
#endif
        if (pos0 > pos1) {
            valid = false;
            if (!full)
                return (PositionResult){pos0, pos1, pos2, pos3, includes, false};
        }
    }

    for (int lit = 0; lit < N_POSITION_FEATS_X; ++lit) {
        if (is_included(ta_state[N_POSITION_FEATS_Y + lit])) {
            if (lit + 1 > pos2)
                pos2 = lit + 1;
            includes++;
        }
#if NEGATED_LITERALS
        if (is_included(ta_state[N_POSITION_FEATS_Y + lit + N_LITERALS / 2])) {
            if (lit < pos3)
                pos3 = lit;
            includes++;
        }
#endif
        if (pos2 > pos3) {
            valid = false;
            if (!full)
                return (PositionResult){pos0, pos1, pos2, pos3, includes, false};
        }
    }

    return (PositionResult){pos0, pos1, pos2, pos3, includes, valid};
#else
    (void)full;
    return (PositionResult){0, N_PATCHES_Y - 1, 0, N_PATCHES_X - 1, 0, true};
#endif
}

typedef struct {
    int n_bounded_feats;
    uint includes;
    bool all_valid;
} FeatureResult;

static inline FeatureResult scan_feature_literals(const uint* ta_state, const int* feat_mins, const int* feat_maxs,
                                                  const int* literal_offsets, int* feat_bounds, int* bounded_feat_ids,
                                                  int full) {
    uint n_includes = 0;
    int write_offset = 0;
    bool all_valid = true;

    for (int fid = 0; fid < N_RAW_PATCH_FEATS; ++fid) {
        int n_bits = literal_offsets[fid + 1] - literal_offsets[fid];
        int lstart = N_POSITION_FEATS + literal_offsets[fid];
        int lb = feat_mins[fid];
        int ub = feat_maxs[fid];
        bool is_bounded = false;

        for (int bit = 0; bit < n_bits; ++bit) {
            if (is_included(ta_state[lstart + bit])) {
                int v = feat_mins[fid] + bit + 1;
                if (v > lb)
                    lb = v;
                n_includes++;
                is_bounded = true;
            }
#if NEGATED_LITERALS
            if (is_included(ta_state[lstart + bit + N_LITERALS / 2])) {
                int v = feat_mins[fid] + bit;
                if (v < ub)
                    ub = v;
                n_includes++;
                is_bounded = true;
            }
#endif
        }

        feat_bounds[fid * 2 + 0] = lb;
        feat_bounds[fid * 2 + 1] = ub;
        if (is_bounded) {
            bounded_feat_ids[write_offset++] = fid;
        }

        if (lb > ub) {
            all_valid = false;
            if (!full)
                break;
        }
    }

    return (FeatureResult){write_offset, n_includes, all_valid};
}

void pack_clauses(const uint* restrict global_ta_states, const int* restrict feat_mins, const int* restrict feat_maxs,
                  const int* restrict literal_offsets, int* restrict clause_position_bounds,
                  int* restrict clause_feat_bounds, int* restrict bounded_feat_ids, int* restrict n_bounded_feats,
                  int32_t* restrict clause_density, int8_t* restrict is_clause_synced, int full) {
#pragma omp parallel for schedule(dynamic)
    for (ull clause = 0; clause < (ull)TOTAL_CLAUSES; clause++) {
        if (is_clause_synced[clause])
            continue;

        const uint* ta_state = &global_ta_states[clause * (ull)N_LITERALS];
        int* pos = &clause_position_bounds[clause * 4];

        PositionResult pr = scan_position_literals(ta_state, full);
        pos[0] = pr.pos0;
        pos[1] = pr.pos1;
        pos[2] = pr.pos2;
        pos[3] = pr.pos3;

        if (!pr.valid && !full) {
            clause_density[clause] = -1;
            is_clause_synced[clause] = 1;
            continue;
        }

        int* cfb = &clause_feat_bounds[clause * (ull)N_RAW_PATCH_FEATS * 2];
        int* cfids = &bounded_feat_ids[clause * (ull)N_RAW_PATCH_FEATS];

        FeatureResult fr = scan_feature_literals(ta_state, feat_mins, feat_maxs, literal_offsets, cfb, cfids, full);

        n_bounded_feats[clause] = fr.n_bounded_feats;
        clause_density[clause] = (pr.valid && fr.all_valid) ? (int32_t)(pr.includes + fr.includes) : -1;
        is_clause_synced[clause] = 1;
    }
}
