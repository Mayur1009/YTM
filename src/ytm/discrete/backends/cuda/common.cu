#ifdef IS_NEOVIM_CLANGD_ENV
#define TOTAL_CLAUSES 1000
#define T_MIN -100.0f
#define T_MAX 100.0f
#define S 10.0f
#define CLASSES 10
#define Q 1.0f
#define HEIGHT 28
#define WIDTH 28
#define DEPTH 1
#define PATCH_HEIGHT 10
#define PATCH_WIDTH 10
#define STRIDE_Y 1
#define STRIDE_X 1
#define MAX_WEIGHT 10.0f
#define MAX_INCLUDED_LITERALS 100
#define INCLUDE_STATE 128
#define MAX_TA_STATE 255
#define N_RAW_PATCH_FEATS 100
#define N_PATCH_FEATS 100
#define N_POSITION_FEATS 36
#define N_PATCHES_Y 19
#define N_PATCHES_X 19
#define N_PATCHES 361
#define N_LITERALS 272
#define NEGATED_LITERALS 1
#define POSITION_LITERALS 1
#define COALESCED 0
#define WEIGHTED 1
#define NEGATIVE_CLAUSES 1
#define ALLOW_POLARITY_CHANGE 1
#define TYPE1A_FB 1
#define TYPE1B_FB 1
#define TYPE2_FB 1
#define TRACK_PATCH_WEIGHTS 1
#define BOOST_TP_FB 1
#endif

#define S_INV (1.0f / (float)(S))
#define N_POSITION_FEATS_Y (N_PATCHES_Y - 1)
#define N_POSITION_FEATS_X (N_PATCHES_X - 1)

#if COALESCED == 0
#define CLAUSES_PER_CLASS (TOTAL_CLAUSES / CLASSES)
#else
#define CLAUSES_PER_CLASS TOTAL_CLAUSES
#endif

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;
using warp_t = cg::thread_block_tile<32>;
typedef unsigned int uint;
typedef signed char int8_t;
typedef unsigned long long ull;

__device__ inline ull hash_combine(ull a, ull b) { return a ^ (b + 0x9e3779b97f4a7c15ULL + (a << 6) + (a >> 2)); }

__device__ inline ull rng_hash(ull seed, ull a, ull b, ull c) {
    ull k = hash_combine(seed, a);
    k = hash_combine(k, b);
    k = hash_combine(k, c);
    return k;
}

__device__ inline float rand_uniform(ull key, uint* counter) {
    ull x = key ^ (ull)((*counter)++);
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    x = x ^ (x >> 31);
    return (float)(x >> 32) * 0x1p-32f;
}

__device__ inline int get_feature_value(const int* X, int patch_idx_y, int patch_idx_x, int fid) {
    int rel_y = fid / (PATCH_WIDTH * DEPTH);
    int rel_x = (fid / DEPTH) % PATCH_WIDTH;
    int z = fid % DEPTH;
    int abs_y = patch_idx_y * STRIDE_Y + rel_y;
    int abs_x = patch_idx_x * STRIDE_X + rel_x;
    return X[abs_y * (WIDTH * DEPTH) + abs_x * DEPTH + z];
}

__device__ inline bool match_patch(const int* X, int patch_idx_y, int patch_idx_x, const int* feat_bounds,
                                   const int* bounded_feat_ids, int n_bounded_feat_ids) {
    for (int i = 0; i < n_bounded_feat_ids; ++i) {
        int fid = bounded_feat_ids[i];
        int val = get_feature_value(X, patch_idx_y, patch_idx_x, fid);
        if (val < feat_bounds[fid * 2] || val > feat_bounds[fid * 2 + 1])
            return false;
    }
    return true;
}
template <typename T> __device__ inline T clip(T val, T lo, T hi) { return val < lo ? lo : (val > hi ? hi : val); }
