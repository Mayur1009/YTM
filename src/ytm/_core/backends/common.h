#pragma once

#ifdef IS_NEOVIM_CLANGD_ENV
#define TOTAL_CLAUSES 1000
#define INCLUDE_STATE 128
#define MAX_TA_STATE 255
#define BOOST_TP_INC 1
#define BOOST_TP_DEC 0
#define CLASSES 10
#define COALESCED 0
#define NEGATIVE_CLAUSES 1
#define S 10.0f
#define HEIGHT 28
#define WIDTH 28
#define DEPTH 1
#define PATCH_HEIGHT 10
#define PATCH_WIDTH 10
#define STRIDE_Y 1
#define STRIDE_X 1
#define N_PATCHES_Y 19
#define N_PATCHES_X 19
#define N_PATCHES 361
#define N_RAW_PATCH_FEATS 100
#define N_PATCH_FEATS 100
#define N_POSITION_FEATS 36
#define N_LITERALS 272
#define MAX_INCLUDED_LITERALS 272
#define NEGATED_LITERALS 1
#define POSITION_LITERALS 1
#define ALL_BINARY_FEATS 0
#define WEIGHTED 1
#define MAX_WEIGHT 1073741824.0f
#define ALLOW_POLARITY_CHANGE 1
#define TYPE1A_FB 1
#define TYPE1B_FB 1
#define TYPE2_FB 1
#define TRACK_PATCH_WEIGHTS 1

#define TA_STATE_T uint32_t
#define FBOUND_T uint32_t
#define PBOUND_T uint32_t
#define NFEAT_T uint32_t
#define NPATCHES_T uint32_t
#define NLITS_T uint32_t

#include "cpu.h"
#endif


#define S_INV (1.0f / (float)(S))
#define N_POSITION_FEATS_Y (N_PATCHES_Y - 1)
#define N_POSITION_FEATS_X (N_PATCHES_X - 1)

#if COALESCED == 0
#define CLAUSES_PER_CLASS (TOTAL_CLAUSES / CLASSES)
#define LOOP_CLASS_ID(class_id, clause) class_id = (ull)clause / (CLAUSES_PER_CLASS);
#else
#define CLAUSES_PER_CLASS TOTAL_CLAUSES
#define LOOP_CLASS_ID(class_id, clause) for (class_id = 0; class_id < CLASSES; ++class_id)
#endif

INLINE_FN float clip(float val, float lo, float hi) { return (val < lo) ? lo : ((val > hi) ? hi : val); }

// A literal is included in the clause once its TA has crossed into the include half.
INLINE_FN bool is_included(uint ta_state) { return ta_state >= INCLUDE_STATE; }

// Index of raw patch feature `fid` of the patch at (patch_idx_y, patch_idx_x) in the flat image.
INLINE_FN int flat_index(int fid, int patch_idx_y, int patch_idx_x) {
#if (PATCH_HEIGHT == HEIGHT && PATCH_WIDTH == WIDTH)
    return fid;
#else
    int rel_y = fid / (PATCH_WIDTH * DEPTH);
    int rel_x = (fid / DEPTH) % PATCH_WIDTH;
    int z = fid % DEPTH;
    int abs_y = patch_idx_y * STRIDE_Y + rel_y;
    int abs_x = patch_idx_x * STRIDE_X + rel_x;
    return abs_y * (WIDTH * DEPTH) + abs_x * DEPTH + z;
#endif
}

// Value of raw patch feature `fid` in the patch at (patch_idx_y, patch_idx_x).
INLINE_FN FBOUND_T get_feature_value(const FBOUND_T* X, int patch_idx_y, int patch_idx_x, int fid) {
    return X[flat_index(fid, patch_idx_y, patch_idx_x)];
}

// True when the patch matches a clause (strict AND matching).
INLINE_FN bool match_patch(const FBOUND_T* X, int patch_idx_y, int patch_idx_x, const NFEAT_T* feat_ids,
                           const FBOUND_T* feat_bounds, int n_feats) {
    for (int i = 0; i < n_feats; ++i) {
        FBOUND_T val = get_feature_value(X, patch_idx_y, patch_idx_x, (int)feat_ids[i]);
        if (val < feat_bounds[i * 2] || val > feat_bounds[i * 2 + 1])
            return false;
    }
    return true;
}

// Clause output on this one sample.
INLINE_FN int clause_output(const FBOUND_T* Xe, ull clause, const PBOUND_T* clause_position_bounds,
                            const NFEAT_T* clause_feat_ids, const FBOUND_T* clause_feat_bounds,
                            const NFEAT_T* clause_n_feats, bool has_contra, NLITS_T clause_len) {
    if (has_contra)
        return 0;
    if (clause_len == 0)
        return 1;

    const NFEAT_T* fids = &clause_feat_ids[clause * (ull)N_RAW_PATCH_FEATS];
    const FBOUND_T* feat_bounds = &clause_feat_bounds[clause * (ull)N_RAW_PATCH_FEATS * 2];
    int n_fids = (int)clause_n_feats[clause];

#if (N_PATCHES > 1)
    const PBOUND_T* pos = &clause_position_bounds[clause * 4];
    for (int py = pos[0]; py <= pos[1]; py++)
        for (int px = pos[2]; px <= pos[3]; px++)
            if (match_patch(Xe, py, px, fids, feat_bounds, n_fids))
                return 1;
    return 0;
#else
    return match_patch(Xe, 0, 0, fids, feat_bounds, n_fids) ? 1 : 0;
#endif
}
