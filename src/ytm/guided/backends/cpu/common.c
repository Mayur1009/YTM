#ifdef IS_NEOVIM_CLANGD_ENV
#define TOTAL_CLAUSES 1000
#define T_MIN -100.0f
#define T_MAX 100.0f
#define S 10.0f
#define CLASSES 10
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
#define BIAS 0
#define ACT_SOFTMAX 0
#define ACT_SIGMOID 1
#define ACT_IDENTITY 2
#define LOSS_CE 0
#define LOSS_MSE 1
#define LOSS_MAE 2
#define LOSS_SCE 3
#define LOSS_ASL 4
#define LOSS_TVERSKY 5
#define LOSS_HUBER 6
#define FB_SIGNAL_GRAD 0
#define FB_SIGNAL_DELTA_L 1
#define ACT_FN 0
#define LOSS_FN 0
#define LOSS_GAMMA 0.0f
#define LOSS_EPS 1e-6f
#define LOSS_ALPHA 0.5f
#define LOSS_BETA 0.5f
#define LOSS_DELTA 1.0f
#define LOSS_CLIP 0.05f
#define LOSS_GAMMA_POS 0.0f
#define LOSS_GAMMA_NEG 4.0f
#define FB_SIGNAL 0
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

#if _OPENMP
#include <omp.h>
#define GET_THREAD_ID omp_get_thread_num()
void set_num_threads(int n) { omp_set_num_threads(n); }
#else
#define GET_THREAD_ID 0
void set_num_threads(int n) {}
#endif

#include <stdbool.h>
#include <stdint.h>
typedef unsigned int uint;
typedef unsigned long long ull;

static inline ull mix64(ull x) {
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static inline ull hash_combine(ull a, ull b) { return a ^ mix64(b + 0x9e3779b97f4a7c15ULL); }

static inline ull rng_hash(ull seed, ull a, ull b, ull c) {
    ull k = hash_combine(seed, a);
    k = hash_combine(k, b);
    k = hash_combine(k, c);
    return k;
}

static inline float rand_uniform(ull key, uint* counter) {
    ull x = key ^ (ull)((*counter)++);
    x = mix64(x);
    return (float)(x >> 32) * 0x1p-32f;
}

static inline int get_feature_value(const int* X, int patch_idx_y, int patch_idx_x, int fid) {
    int rel_y = fid / (PATCH_WIDTH * DEPTH);
    int rel_x = (fid / DEPTH) % PATCH_WIDTH;
    int z = fid % DEPTH;
    int abs_y = patch_idx_y * STRIDE_Y + rel_y;
    int abs_x = patch_idx_x * STRIDE_X + rel_x;
    return X[abs_y * (WIDTH * DEPTH) + abs_x * DEPTH + z];
}

static inline bool match_patch(const int* X, int patch_idx_y, int patch_idx_x, const int* feat_bounds,
                               const int* bounded_feat_ids, int n_bounded_feat_ids) {
    for (int i = 0; i < n_bounded_feat_ids; ++i) {
        int fid = bounded_feat_ids[i];
        int val = get_feature_value(X, patch_idx_y, patch_idx_x, fid);
        if (val < feat_bounds[fid * 2] || val > feat_bounds[fid * 2 + 1])
            return false;
    }
    return true;
}

static inline float clip(float val, float lo, float hi) { return (val < lo) ? lo : ((val > hi) ? hi : val); }
