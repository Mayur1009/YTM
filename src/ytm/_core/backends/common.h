#ifdef IS_NEOVIM_CLANGD_ENV
#include "cpu.h"
#define TOTAL_CLAUSES 1000
#define INCLUDE_STATE 128
#define CLASSES 10
#define COALESCED 0
#define S 10.0f
#define HEIGHT 28
#define WIDTH 28
#define DEPTH 1
#define PATCH_WIDTH 10
#define STRIDE_Y 1
#define STRIDE_X 1
#define N_PATCHES_Y 19
#define N_PATCHES_X 19
#define N_PATCHES 361
#define N_RAW_PATCH_FEATS 100
#define N_POSITION_FEATS 36
#define N_LITERALS 272
#define NEGATED_LITERALS 1
#define POSITION_LITERALS 1
#endif

// Derived macros and patch helpers, shared by the C and CUDA builds.
#pragma once

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

// Value of raw patch feature `fid` in the patch at (patch_idx_y, patch_idx_x).
INLINE_FN int get_feature_value(const int* X, int patch_idx_y, int patch_idx_x, int fid) {
    int rel_y = fid / (PATCH_WIDTH * DEPTH);
    int rel_x = (fid / DEPTH) % PATCH_WIDTH;
    int z = fid % DEPTH;
    int abs_y = patch_idx_y * STRIDE_Y + rel_y;
    int abs_x = patch_idx_x * STRIDE_X + rel_x;
    return X[abs_y * (WIDTH * DEPTH) + abs_x * DEPTH + z];
}

// True when the patch falls inside every bound the clause constrains.
INLINE_FN bool match_patch(const int* X, int patch_idx_y, int patch_idx_x, const int* feat_bounds,
                           const int* bounded_feat_ids, int n_bounded_feat_ids) {
    for (int i = 0; i < n_bounded_feat_ids; ++i) {
        int fid = bounded_feat_ids[i];
        int val = get_feature_value(X, patch_idx_y, patch_idx_x, fid);
        if (val < feat_bounds[fid * 2] || val > feat_bounds[fid * 2 + 1])
            return false;
    }
    return true;
}
