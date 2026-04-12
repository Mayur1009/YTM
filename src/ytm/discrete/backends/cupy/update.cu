#ifdef IS_NEOVIM_CLANGD_ENV
#define TOTAL_CLAUSES 1000
#define THRESH 100
#define S 10.0
#define CLASSES 10
#define HEIGHT 28
#define WIDTH 28
#define DEPTH 1
#define PATCH_HEIGHT 10
#define PATCH_WIDTH 10
#define STRIDE_Y 1
#define STRIDE_X 1
#define NEGATED_LITERALS 1
#define POSITION_LITERALS 1
#define COALESCED 0
#define WEIGHTED 1
#define MAX_WEIGHT 10.0f
#define NEGATIVE_CLAUSES 1
#define ALLOW_POLARITY_CHANGE 1
#define MAX_INCLUDED_LITERALS 100
#define INCLUDE_STATE 128
#define MAX_TA_STATE 255
#define TYPE1A_FB 1
#define TYPE1B_FB 1
#define TYPE2_FB 1
#define TRACK_PATCH_WEIGHTS 1
#define BOOST_TP_FB 1
#define N_RAW_PATCH_FEATS 100
#define N_PATCH_FEATS 100
#define N_POSITION_FEATS 36
#define N_PATCHES_Y 19
#define N_PATCHES_X 19
#define N_PATCHES 361
#define N_LITERALS 272
#define WARP_SIZE 32
#endif

#define N_POSITION_FEATS_Y (N_PATCHES_Y - 1)
#define N_POSITION_FEATS_X (N_PATCHES_X - 1)
#define S_INV (1.0f / S)
#define UINT_MAX_INV (1.0f / 4294967295.0f)

#if COALESCED == 0
#define CLAUSES_PER_CLASS (TOTAL_CLAUSES / CLASSES)
#define LOOP_CLASS_ID(class_id, clause) class_id = (ull)clause / (CLAUSES_PER_CLASS);
#else
#define CLAUSES_PER_CLASS TOTAL_CLAUSES
#define LOOP_CLASS_ID(class_id, clause) for (class_id = 0; class_id < CLASSES; ++class_id)
#endif

#define CLIP(val, lo, hi) ((val < lo) ? lo : ((val > hi) ? hi : val))

typedef unsigned long long ull;
typedef unsigned int uint;
typedef signed char int8_t;

extern "C" {

__device__ inline float xorshift32(uint* state) {
    uint x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return (float)x * UINT_MAX_INV;
}

__device__ inline int geometric_sample(uint* rng, float p) {
    float u = xorshift32(rng);
    if (u >= 1.0f)
        u = 0.9999999f;
    return (int)(logf(1.0f - u) / logf(1.0f - p)) + 1;
}

__device__ inline void literal_dec_with_p(uint* rng, uint* ta_state, int start, int end, int offset, float p) {
    int li = start + geometric_sample(rng, p) - 1;
    while (li < end) {
        if (ta_state[li + offset] > 0)
            ta_state[li + offset] -= 1;
        li += geometric_sample(rng, p);
    }
}

__device__ inline void literal_inc(uint* ta_state, int start, int end, int offset, uint max_val) {
    for (int li = start; li < end; ++li) {
        ta_state[li + offset] += (ta_state[li + offset] < max_val);
    }
}

__device__ inline void literal_inc_maybe_p(uint* rng, uint* ta_state, int start, int end, int offset, float p) {
#if BOOST_TP_FB
    literal_inc(ta_state, start, end, offset, MAX_TA_STATE);
#else
    int li = start + geometric_sample(rng, p) - 1;
    while (li < end) {
        if (ta_state[li + offset] < MAX_TA_STATE)
            ta_state[li + offset] += 1;
        li += geometric_sample(rng, p);
    }
#endif
}

__device__ inline float uprob_fun(float v, float y) { return (y - v) / (2 * y); }

__device__ inline int get_feature_value(const int* X, int patch_idx_y, int patch_idx_x, int fid) {
    int rel_y = fid / (PATCH_WIDTH * DEPTH);
    int rel_x = (fid / DEPTH) % PATCH_WIDTH;
    int z = fid % DEPTH;
    int abs_y = patch_idx_y * STRIDE_Y + rel_y;
    int abs_x = patch_idx_x * STRIDE_X + rel_x;
    return X[abs_y * (WIDTH * DEPTH) + abs_x * DEPTH + z];
}

// --- Warp-parallel feedback functions ---

__device__ void warp_type1a_fb(uint* rng, uint* ta_state, float* weight, const int* X,
                               int patch_idx_y, int patch_idx_x, int sign, const int* feat_mins,
                               const int* literal_offsets, int lane) {
#if TYPE1A_FB
#if WEIGHTED
    if (lane == 0 && fabsf(*weight) < MAX_WEIGHT)
        (*weight) += sign * 1.0f;
#endif

#if POSITION_LITERALS
    if (lane == 0) {
        literal_inc_maybe_p(rng, ta_state, 0, patch_idx_y, 0, 1.0f - S_INV);
        literal_dec_with_p(rng, ta_state, patch_idx_y, N_POSITION_FEATS_Y, 0, S_INV);

        literal_inc_maybe_p(rng, ta_state, N_POSITION_FEATS_Y, N_POSITION_FEATS_Y + patch_idx_x, 0, 1.0f - S_INV);
        literal_dec_with_p(rng, ta_state, N_POSITION_FEATS_Y + patch_idx_x, N_POSITION_FEATS, 0, S_INV);

#if NEGATED_LITERALS
        literal_dec_with_p(rng, ta_state, 0, patch_idx_y, N_LITERALS / 2, S_INV);
        literal_inc_maybe_p(rng, ta_state, patch_idx_y, N_POSITION_FEATS_Y, N_LITERALS / 2, 1.0f - S_INV);

        literal_dec_with_p(rng, ta_state, N_POSITION_FEATS_Y, N_POSITION_FEATS_Y + patch_idx_x, N_LITERALS / 2, S_INV);
        literal_inc_maybe_p(rng, ta_state, N_POSITION_FEATS_Y + patch_idx_x, N_POSITION_FEATS, N_LITERALS / 2,
                            1.0f - S_INV);
#endif
    }
#endif

    for (int fid = lane; fid < N_RAW_PATCH_FEATS; fid += WARP_SIZE) {
        int lit_start = N_POSITION_FEATS + literal_offsets[fid];
        int lit_end = N_POSITION_FEATS + literal_offsets[fid + 1];
        int shifted_val = get_feature_value(X, patch_idx_y, patch_idx_x, fid) - feat_mins[fid];

        literal_inc_maybe_p(rng, ta_state, lit_start, lit_start + shifted_val, 0, 1.0f - S_INV);
        literal_dec_with_p(rng, ta_state, lit_start + shifted_val, lit_end, 0, S_INV);

#if NEGATED_LITERALS
        literal_dec_with_p(rng, ta_state, lit_start, lit_start + shifted_val, N_LITERALS / 2, S_INV);
        literal_inc_maybe_p(rng, ta_state, lit_start + shifted_val, lit_end, N_LITERALS / 2, 1.0f - S_INV);
#endif
    }
#endif
}

__device__ void warp_type1b_fb(uint* rng, uint* ta_state, int lane) {
#if TYPE1B_FB
    for (int li = lane; li < N_LITERALS; li += WARP_SIZE) {
        if (xorshift32(rng) <= S_INV && ta_state[li] > 0)
            ta_state[li] -= 1;
    }
#endif
}

__device__ void warp_type2_fb(uint* ta_state, float* weight, const int* X, int patch_idx_y,
                              int patch_idx_x, int sign, const int* feat_mins,
                              const int* literal_offsets, int lane) {
#if TYPE2_FB
#if WEIGHTED
    if (lane == 0) {
        if (fabsf(*weight) < MAX_WEIGHT)
            (*weight) -= sign * 1.0f;
#if ALLOW_POLARITY_CHANGE == 0
        if (sign == 1 && *weight < 0)
            *weight = 1;
        if (sign == -1 && *weight >= 0)
            *weight = -1;
#endif
    }
#endif
#if NEGATIVE_CLAUSES == 0
    if (lane == 0 && *weight < 1)
        *weight = 1;
#endif

#if POSITION_LITERALS
    if (lane == 0) {
        literal_inc(ta_state, patch_idx_y, N_POSITION_FEATS_Y, 0, INCLUDE_STATE);
        literal_inc(ta_state, N_POSITION_FEATS_Y + patch_idx_x, N_POSITION_FEATS, 0, INCLUDE_STATE);

#if NEGATED_LITERALS
        literal_inc(ta_state, 0, patch_idx_y, N_LITERALS / 2, INCLUDE_STATE);
        literal_inc(ta_state, N_POSITION_FEATS_Y, N_POSITION_FEATS_Y + patch_idx_x, N_LITERALS / 2, INCLUDE_STATE);
#endif
    }
#endif

    for (int fid = lane; fid < N_RAW_PATCH_FEATS; fid += WARP_SIZE) {
        int lit_start = N_POSITION_FEATS + literal_offsets[fid];
        int lit_end = N_POSITION_FEATS + literal_offsets[fid + 1];
        int shifted_val = get_feature_value(X, patch_idx_y, patch_idx_x, fid) - feat_mins[fid];

        literal_inc(ta_state, lit_start + shifted_val, lit_end, 0, INCLUDE_STATE);

#if NEGATED_LITERALS
        literal_inc(ta_state, lit_start, lit_start + shifted_val, N_LITERALS / 2, INCLUDE_STATE);
#endif
    }
#endif
}

// --- Kernels ---

__global__ void calc_update_prob(const float* votes, const float* targets, const int e, float* prob) {
    ull tid = threadIdx.x + blockIdx.x * blockDim.x;
    ull stride = blockDim.x * gridDim.x;
    for (ull class_id = tid; class_id < (ull)CLASSES; class_id += stride) {
        float target = targets[(ull)e * CLASSES + class_id];
        if (target == 0.0f) {
            prob[class_id] = 0.0f;
            continue;
        }
        float y = (float)THRESH * (target > 0.0f ? 1.0f : -1.0f);
        float v = (float)CLIP(votes[class_id], -THRESH, THRESH);
        prob[class_id] = uprob_fun(v, y);
    }
}

__global__ void update_clauses(uint* rng, const int* selected_patch_ids, const uint* num_includes,
                               const int8_t* clause_drop_mask, const int* X, const float* targets, const int e,
                               const float* prob, uint* ta_states, float* clause_weights, const int* feat_mins,
                               const int* literal_offsets, int8_t* is_clause_synced) {
    ull tid = threadIdx.x + blockIdx.x * blockDim.x;
    int lane = threadIdx.x % WARP_SIZE;
    ull warp_id = tid / WARP_SIZE;
    ull total_warps = (ull)(blockDim.x * gridDim.x) / WARP_SIZE;

    const int* Xe = &X[(ull)e * HEIGHT * WIDTH * DEPTH];
    const float* targets_e = &targets[(ull)e * CLASSES];

    // Lane 0: load clause RNG state
    uint local_rng;
    if (lane == 0)
        local_rng = rng[warp_id];

    // Per-lane xorshift for parallel feature work
    uint lane_rng;

    for (ull clause = warp_id; clause < (ull)TOTAL_CLAUSES; clause += total_warps) {
        if (clause_drop_mask[clause] == 1)
            continue;

        // Reseed per-lane RNG from lane 0 each clause
        uint xseed = 0;
        if (lane == 0)
            xseed = (uint)(xorshift32(&local_rng) * 4294967295.0f);
        xseed = __shfl_sync(0xFFFFFFFF, xseed, 0);
        lane_rng = (xseed ^ (lane * 2654435761u)) | 1u;

        uint* ta_state = &ta_states[clause * (ull)N_LITERALS];
        int patch_id = selected_patch_ids[clause];
        int clause_output = (patch_id >= 0) ? 1 : 0;

        int patch_idx_y = -1, patch_idx_x = -1;
        if (clause_output) {
            patch_idx_y = patch_id / N_PATCHES_X;
            patch_idx_x = patch_id % N_PATCHES_X;
        }

        uint clause_includes = num_includes[clause];

        ull class_id, rel_clause = clause % (ull)CLAUSES_PER_CLASS;
        LOOP_CLASS_ID(class_id, clause) {
            int target = 0;
            bool should_update = false;
            if (lane == 0) {
                float q_prob = targets_e[class_id];
                if (q_prob != 0.0f && xorshift32(&local_rng) <= fabsf(q_prob)) {
                    target = (q_prob > 0.0f) ? 1 : -1;
                    should_update = (xorshift32(&local_rng) <= prob[class_id]);
                }
            }
            target = __shfl_sync(0xFFFFFFFF, target, 0);
            should_update = __shfl_sync(0xFFFFFFFF, (int)should_update, 0);

            if (target == 0 || !should_update)
                continue;

            float* weight = &clause_weights[class_id * (ull)CLAUSES_PER_CLASS + rel_clause];
            int sign = (*weight >= 0) - (*weight < 0);
            bool has_space = (clause_includes <= (uint)MAX_INCLUDED_LITERALS);
            bool t1 = (target * sign) > 0;

            if (t1 && clause_output && has_space) {
                warp_type1a_fb(&lane_rng, ta_state, weight, Xe, patch_idx_y, patch_idx_x, sign, feat_mins,
                               literal_offsets, lane);
                is_clause_synced[clause] = 0;
            }

            if (t1 && !(clause_output && has_space)) {
                warp_type1b_fb(&lane_rng, ta_state, lane);
                is_clause_synced[clause] = 0;
            }

            if ((target * sign) < 0 && clause_output) {
                warp_type2_fb(ta_state, weight, Xe, patch_idx_y, patch_idx_x, sign, feat_mins, literal_offsets, lane);
                is_clause_synced[clause] = 0;
            }
        }
    }

    if (lane == 0)
        rng[warp_id] = local_rng;
}

} // extern "C"
