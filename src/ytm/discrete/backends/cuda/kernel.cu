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
#endif

#define N_POSITION_FEATS_Y (N_PATCHES_Y - 1)
#define N_POSITION_FEATS_X (N_PATCHES_X - 1)
#define S_INV (1.0f / S)

#if COALESCED == 0
#define CLAUSES_PER_CLASS (TOTAL_CLAUSES / CLASSES)
#define LOOP_CLASS_ID(class_id, clause) class_id = (ull)clause / (CLAUSES_PER_CLASS);
#else
#define CLAUSES_PER_CLASS TOTAL_CLAUSES
#define LOOP_CLASS_ID(class_id, clause) for (class_id = 0; class_id < CLASSES; ++class_id)
#endif

#define CLIP(val, lo, hi) ((val < lo) ? lo : ((val > hi) ? hi : val))

#include <curand_kernel.h>

typedef unsigned long long ull;
typedef unsigned int uint;

extern "C" {

__device__ inline int geometric_sample(curandState* rng, float p) {
    float u = curand_uniform(rng);
    if (u >= 1.0f)
        u = 0.9999999f;
    return (int)(logf(1.0f - u) / logf(1.0f - p)) + 1;
}

__device__ inline void literal_dec_with_p(curandState* rng, uint* ta_state, int start, int end, int offset, float p) {
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

__device__ inline void literal_inc_maybe_p(curandState* rng, uint* ta_state, int start, int end, int offset, float p) {
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
__device__ inline bool is_included(uint ta_state) { return ta_state >= INCLUDE_STATE; }

__device__ inline int get_feature_value(const int* X, int patch_idx_y, int patch_idx_x, int fid) {
    int rel_y = fid / (PATCH_WIDTH * DEPTH);
    int rel_x = (fid / DEPTH) % PATCH_WIDTH;
    int z = fid % DEPTH;
    int abs_y = patch_idx_y * STRIDE_Y + rel_y;
    int abs_x = patch_idx_x * STRIDE_X + rel_x;
    return X[abs_y * (WIDTH * DEPTH) + abs_x * DEPTH + z];
}

// Check if a patch matches a clause using sparse range representation
__device__ inline bool match_patch(const int* X, int patch_idx_y, int patch_idx_x, const int* cfb, const int* cfids,
                                   int n_cfids) {
    for (int i = 0; i < n_cfids; ++i) {
        int fid = cfids[i];
        int val = get_feature_value(X, patch_idx_y, patch_idx_x, fid);
        if (val < cfb[fid * 2] || val > cfb[fid * 2 + 1])
            return false;
    }
    return true;
}

__device__ void type1a_fb(curandState* rng, uint* ta_state, float* weight, const int* X, int patch_idx_y,
                          int patch_idx_x, int sign, const int* feat_mins, const int* literal_offsets) {
#if TYPE1A_FB
#if WEIGHTED
    if (fabsf(*weight) < MAX_WEIGHT)
        (*weight) += sign * 1.0f;
#endif

#if POSITION_LITERALS
    // Position Y
    literal_inc_maybe_p(rng, ta_state, 0, patch_idx_y, 0, 1.0f - S_INV);
    literal_dec_with_p(rng, ta_state, patch_idx_y, N_POSITION_FEATS_Y, 0, S_INV);

    // Position X
    literal_inc_maybe_p(rng, ta_state, N_POSITION_FEATS_Y, N_POSITION_FEATS_Y + patch_idx_x, 0, 1.0f - S_INV);
    literal_dec_with_p(rng, ta_state, N_POSITION_FEATS_Y + patch_idx_x, N_POSITION_FEATS, 0, S_INV);

#if NEGATED_LITERALS
    // Negated position Y
    literal_dec_with_p(rng, ta_state, 0, patch_idx_y, N_LITERALS / 2, S_INV);
    literal_inc_maybe_p(rng, ta_state, patch_idx_y, N_POSITION_FEATS_Y, N_LITERALS / 2, 1.0f - S_INV);

    // Negated position X
    literal_dec_with_p(rng, ta_state, N_POSITION_FEATS_Y, N_POSITION_FEATS_Y + patch_idx_x, N_LITERALS / 2, S_INV);
    literal_inc_maybe_p(rng, ta_state, N_POSITION_FEATS_Y + patch_idx_x, N_POSITION_FEATS, N_LITERALS / 2,
                        1.0f - S_INV);
#endif
#endif

    // Feature literals
    for (int fid = 0; fid < N_RAW_PATCH_FEATS; ++fid) {
        int lit_start = N_POSITION_FEATS + literal_offsets[fid];
        int lit_end = N_POSITION_FEATS + literal_offsets[fid + 1];
        int shifted_val = get_feature_value(X, patch_idx_y, patch_idx_x, fid) - feat_mins[fid];

        // Positive: [0, shifted_val) are 1, [shifted_val, n_bits) are 0
        literal_inc_maybe_p(rng, ta_state, lit_start, lit_start + shifted_val, 0, 1.0f - S_INV);
        literal_dec_with_p(rng, ta_state, lit_start + shifted_val, lit_end, 0, S_INV);

#if NEGATED_LITERALS
        // Negated: [0, shifted_val) are 0, [shifted_val, n_bits) are 1
        literal_dec_with_p(rng, ta_state, lit_start, lit_start + shifted_val, N_LITERALS / 2, S_INV);
        literal_inc_maybe_p(rng, ta_state, lit_start + shifted_val, lit_end, N_LITERALS / 2, 1.0f - S_INV);
#endif
    }
#endif
}

__device__ void type1b_fb(curandState* rng, uint* ta_state) {
#if TYPE1B_FB
    literal_dec_with_p(rng, ta_state, 0, N_LITERALS, 0, S_INV);
#endif
}

__device__ void type2_fb(uint* ta_state, float* weight, const int* X, int patch_idx_y, int patch_idx_x, int sign,
                         const int* feat_mins, const int* literal_offsets) {
#if TYPE2_FB
#if WEIGHTED
    if (fabsf(*weight) < MAX_WEIGHT)
        (*weight) -= sign * 1.0f;
#if ALLOW_POLARITY_CHANGE == 0
    if (sign == 1 && *weight < 0)
        *weight = 1;
    if (sign == -1 && *weight >= 0)
        *weight = -1;
#endif
#endif
#if NEGATIVE_CLAUSES == 0
    if (*weight < 1)
        *weight = 1;
#endif

#if POSITION_LITERALS
    // Position Y
    literal_inc(ta_state, patch_idx_y, N_POSITION_FEATS_Y, 0, INCLUDE_STATE);
    // Position X
    literal_inc(ta_state, N_POSITION_FEATS_Y + patch_idx_x, N_POSITION_FEATS, 0, INCLUDE_STATE);

#if NEGATED_LITERALS
    // Negated position Y
    literal_inc(ta_state, 0, patch_idx_y, N_LITERALS / 2, INCLUDE_STATE);
    // Negated position X
    literal_inc(ta_state, N_POSITION_FEATS_Y, N_POSITION_FEATS_Y + patch_idx_x, N_LITERALS / 2, INCLUDE_STATE);
#endif
#endif

    // Feature literals
    for (int fid = 0; fid < N_RAW_PATCH_FEATS; ++fid) {
        int lit_start = N_POSITION_FEATS + literal_offsets[fid];
        int lit_end = N_POSITION_FEATS + literal_offsets[fid + 1];
        int shifted_val = get_feature_value(X, patch_idx_y, patch_idx_x, fid) - feat_mins[fid];

        // Positive: [shifted_val, n_bits) are 0 → increment
        literal_inc(ta_state, lit_start + shifted_val, lit_end, 0, INCLUDE_STATE);

#if NEGATED_LITERALS
        // Negated: [0, shifted_val) are 0 → increment
        literal_inc(ta_state, lit_start, lit_start + shifted_val, N_LITERALS / 2, INCLUDE_STATE);
#endif
    }
#endif
}

__global__ void pack_clauses(const uint* global_ta_states, const int* feat_mins, const int* feat_maxs,
                             const int* literal_offsets, int* clause_position_bounds, int* clause_feat_bounds,
                             int* constrained_fids, int* n_constrained, uint* num_includes, int8_t* is_clause_valid,
                             int8_t* is_clause_synced) {
    ull tid = threadIdx.x + blockIdx.x * blockDim.x;
    ull stride = blockDim.x * gridDim.x;

    for (ull clause = tid; clause < (ull)TOTAL_CLAUSES; clause += stride) {
        // Skip if clause is already synced
        if (is_clause_synced[clause])
            continue;

        const uint* ta_state = &global_ta_states[clause * (ull)N_LITERALS];
        int* pos = &clause_position_bounds[clause * 4];
        is_clause_valid[clause] = 1;

        // Initialize position bounds — closed interval [min, max]
        pos[0] = 0;               // min_row (inclusive)
        pos[1] = N_PATCHES_Y - 1; // max_row (inclusive)
        pos[2] = 0;               // min_col (inclusive)
        pos[3] = N_PATCHES_X - 1; // max_col (inclusive)
        uint total_includes = 0;

#if POSITION_LITERALS
        // Scan Y position literals
        for (int lit = 0; lit < N_POSITION_FEATS_Y; ++lit) {
            if (is_included(ta_state[lit])) {
                pos[0] = max(pos[0], lit + 1);
                total_includes++;
            }
#if NEGATED_LITERALS
            if (is_included(ta_state[lit + N_LITERALS / 2])) {
                pos[1] = min(pos[1], lit);
                total_includes++;
            }
#endif
        }

        // Scan X position literals
        for (int lit = 0; lit < N_POSITION_FEATS_X; ++lit) {
            if (is_included(ta_state[N_POSITION_FEATS_Y + lit])) {
                pos[2] = max(pos[2], lit + 1);
                total_includes++;
            }
#if NEGATED_LITERALS
            if (is_included(ta_state[N_POSITION_FEATS_Y + lit + N_LITERALS / 2])) {
                pos[3] = min(pos[3], lit);
                total_includes++;
            }
#endif
        }
#endif

        // Early exit if position contradiction
        if (pos[0] > pos[1] || pos[2] > pos[3]) {
            is_clause_valid[clause] = 0;
            is_clause_synced[clause] = 1;
            num_includes[clause] = total_includes;
            continue;
        }

        int* cfb = &clause_feat_bounds[clause * (ull)N_RAW_PATCH_FEATS * 2];
        int* cfids = &constrained_fids[clause * (ull)N_RAW_PATCH_FEATS];
        int local_n_constrained = 0;

        // Find inclusive bounds for each feature
        for (ull fid = 0; fid < N_RAW_PATCH_FEATS; ++fid) {
            int n_bits = literal_offsets[fid + 1] - literal_offsets[fid];
            int lstart = N_POSITION_FEATS + literal_offsets[fid];
            bool has_constraint = false;

            // Init bounds to full range [feat_min, feat_max]
            cfb[fid * 2 + 0] = feat_mins[fid];
            cfb[fid * 2 + 1] = feat_maxs[fid];

            for (int bit = 0; bit < n_bits; ++bit) {
                if (is_included(ta_state[lstart + bit])) {
                    cfb[fid * 2 + 0] = max(cfb[fid * 2 + 0], feat_mins[fid] + bit + 1);
                    total_includes++;
                    has_constraint = true;
                }
#if NEGATED_LITERALS
                if (is_included(ta_state[lstart + bit + N_LITERALS / 2])) {
                    cfb[fid * 2 + 1] = min(cfb[fid * 2 + 1], feat_mins[fid] + bit);
                    total_includes++;
                    has_constraint = true;
                }
#endif
            }

            if (cfb[fid * 2 + 0] > cfb[fid * 2 + 1]) {
                is_clause_valid[clause] = 0;
            }
            if (has_constraint) {
                cfids[local_n_constrained++] = fid;
            }
        }

        n_constrained[clause] = local_n_constrained;
        num_includes[clause] = total_includes;
        is_clause_synced[clause] = 1;
    }
}

__global__ void eval_clauses(const int* X, const int e, const int8_t* clause_drop_mask,
                             const int* clause_position_bounds, const int* clause_feat_bounds,
                             const int* constrained_fids, const int* n_constrained, const uint* num_includes,
                             const int8_t* is_clause_valid, int8_t* clause_outputs) {
    ull tid = threadIdx.x + blockIdx.x * blockDim.x;
    ull stride = blockDim.x * gridDim.x;

    const int* Xe = &X[(ull)e * HEIGHT * WIDTH * DEPTH];

    for (ull idx = tid; idx < (ull)TOTAL_CLAUSES * N_PATCHES; idx += stride) {
        ull clause = idx / N_PATCHES;
        int patch = idx % N_PATCHES;

        int8_t* output = &clause_outputs[clause * (ull)N_PATCHES + patch];

        // Skip dropped clauses
        if (clause_drop_mask[clause] == 1) {
            *output = 0;
            continue;
        }

        // Empty clause matches all patches
        if (num_includes[clause] == 0) {
            *output = 1;
            continue;
        }

        // Skip invalid clauses (contradictions)
        if (is_clause_valid[clause] == 0) {
            *output = 0;
            continue;
        }

        // Check position bounds
        const int* pos = &clause_position_bounds[clause * 4];
        int patch_row = patch / N_PATCHES_X;
        int patch_col = patch % N_PATCHES_X;

        // Closed interval [pos[0], pos[1]]
        if (patch_row < pos[0] || patch_row > pos[1] || patch_col < pos[2] || patch_col > pos[3]) {
            *output = 0;
            continue;
        }

        // Use match_patch with sparse range representation
        const int* cfb = &clause_feat_bounds[clause * (ull)N_RAW_PATCH_FEATS * 2];
        const int* cfids = &constrained_fids[clause * (ull)N_RAW_PATCH_FEATS];
        int n_cfids = n_constrained[clause];

        bool matches = match_patch(Xe, patch_row, patch_col, cfb, cfids, n_cfids);
        *output = matches ? 1 : 0;
    }
}

__global__ void select_patch_and_count_votes(curandState* rng, const int8_t* clause_outputs,
                                             const float* clause_weights, int* selected_patch_ids, int* patch_weights,
                                             float* votes) {
    ull tid = threadIdx.x + blockIdx.x * blockDim.x;
    ull stride = blockDim.x * gridDim.x;

    curandState local_rng = rng[tid];

    for (ull clause = tid; clause < (ull)TOTAL_CLAUSES; clause += stride) {
        const int8_t* outputs = &clause_outputs[clause * (ull)N_PATCHES];

#if N_PATCHES > 1
        // Reservoir sampling over matching patches
        int count = 0;
        int selected_id = -1;

        for (int patch = 0; patch < N_PATCHES; ++patch) {
            if (outputs[patch]) {
                count++;
                if (curand_uniform(&local_rng) < 1.0f / count) {
                    selected_id = patch;
                }
            }
        }
#else
        // Single patch case (no sampling needed)
        int selected_id = outputs[0] ? 0 : -1;
#endif

        selected_patch_ids[clause] = selected_id;

        if (selected_id >= 0) {
            // Update patch weights (no race - each clause has unique row)
#if TRACK_PATCH_WEIGHTS
            patch_weights[clause * (ull)N_PATCHES + selected_id]++;
#endif

            // Accumulate votes (atomic needed)
            ull class_id, rel_clause = clause % (ull)CLAUSES_PER_CLASS;
            LOOP_CLASS_ID(class_id, clause) {
                atomicAdd(&votes[class_id], clause_weights[class_id * (ull)CLAUSES_PER_CLASS + rel_clause]);
            }
        }
    }

    rng[tid] = local_rng;
}

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

__global__ void update_clauses(curandState* rng, const int* selected_patch_ids, const uint* num_includes,
                               const int8_t* clause_drop_mask, const int* X, const float* targets, const int e,
                               const float* prob, uint* ta_states, float* clause_weights, const int* feat_mins,
                               const int* literal_offsets, int8_t* is_clause_synced) {
    ull tid = threadIdx.x + blockIdx.x * blockDim.x;
    ull stride = blockDim.x * gridDim.x;

    const int* Xe = &X[(ull)e * HEIGHT * WIDTH * DEPTH];
    const float* targets_e = &targets[(ull)e * CLASSES];
    curandState local_rng = rng[tid];

    for (ull clause = tid; clause < (ull)TOTAL_CLAUSES; clause += stride) {
        if (clause_drop_mask[clause] == 1)
            continue;

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
            float q_prob = targets_e[class_id];
            if (q_prob == 0.0f || curand_uniform(&local_rng) > fabsf(q_prob))
                continue;
            int target = (q_prob > 0.0f) ? 1 : -1;

            float* weight = &clause_weights[class_id * (ull)CLAUSES_PER_CLASS + rel_clause];
            int sign = (*weight >= 0) - (*weight < 0);

            float update_prob = prob[class_id];
            bool should_update = (curand_uniform(&local_rng) <= update_prob);
            bool has_space = (clause_includes <= (uint)MAX_INCLUDED_LITERALS);
            bool t1 = (target * sign) > 0;

            // Type 1a: clause active, correct polarity, has space
            if (should_update && t1 && clause_output && has_space) {
                type1a_fb(&local_rng, ta_state, weight, Xe, patch_idx_y, patch_idx_x, sign, feat_mins, literal_offsets);
                is_clause_synced[clause] = 0; // Mark as needing re-pack
            }

            // Type 1b: should have been active but wasn't (or overflowed)
            if (should_update && t1 && !(clause_output && has_space)) {
                type1b_fb(&local_rng, ta_state);
                is_clause_synced[clause] = 0; // Mark as needing re-pack
            }

            // Type 2: clause active but wrong polarity
            if (should_update && (target * sign) < 0 && clause_output) {
                type2_fb(ta_state, weight, Xe, patch_idx_y, patch_idx_x, sign, feat_mins, literal_offsets);
                is_clause_synced[clause] = 0; // Mark as needing re-pack
            }
        }
    }

    rng[tid] = local_rng;
}

__global__ void infer_batch(const int* X, const float* clause_weights, float* class_sums, const int N,
                            const int* clause_position_bounds, const int* clause_feat_bounds,
                            const int* constrained_fids, const int* n_constrained, const uint* num_includes,
                            const int8_t* is_clause_valid) {
    ull tid = threadIdx.x + blockIdx.x * blockDim.x;
    ull stride = blockDim.x * gridDim.x;

    for (ull e_clause = tid; e_clause < (ull)N * TOTAL_CLAUSES; e_clause += stride) {
        ull e = e_clause / (ull)TOTAL_CLAUSES;
        ull clause = e_clause % (ull)TOTAL_CLAUSES;

        // Skip empty clauses
        if (num_includes[clause] == 0)
            continue;

        // Skip invalid clauses (contradictions)
        if (is_clause_valid[clause] == 0)
            continue;

        const int* Xe = &X[e * (ull)HEIGHT * WIDTH * DEPTH];
        const int* pos = &clause_position_bounds[clause * 4];

        // Get sparse range representation for this clause
        const int* cfb = &clause_feat_bounds[clause * (ull)N_RAW_PATCH_FEATS * 2];
        const int* cfids = &constrained_fids[clause * (ull)N_RAW_PATCH_FEATS];
        int n_cfids = n_constrained[clause];

        bool matched = false;

        // Early exit on first matching patch — closed interval [pos[0], pos[1]]
        for (int py = pos[0]; py <= pos[1] && !matched; ++py) {
            for (int px = pos[2]; px <= pos[3] && !matched; ++px) {
                matched = match_patch(Xe, py, px, cfb, cfids, n_cfids);
            }
        }

        if (matched) {
            ull class_id, rel_clause = clause % (ull)CLAUSES_PER_CLASS;
            LOOP_CLASS_ID(class_id, clause) {
                atomicAdd(&class_sums[e * (ull)CLASSES + class_id],
                          clause_weights[class_id * (ull)CLAUSES_PER_CLASS + rel_clause]);
            }
        }
    }
}

__global__ void transform_patchwise(const int* X, int8_t* patch_output, const int N, const int* clause_position_bounds,
                                    const int* clause_feat_bounds, const int* constrained_fids,
                                    const int* n_constrained, const uint* num_includes, const int8_t* is_clause_valid) {
    ull tid = threadIdx.x + blockIdx.x * blockDim.x;
    ull stride = blockDim.x * gridDim.x;

    for (ull idx = tid; idx < (ull)N * TOTAL_CLAUSES * N_PATCHES; idx += stride) {
        ull e = idx / ((ull)TOTAL_CLAUSES * N_PATCHES);
        ull clause_patch = idx % ((ull)TOTAL_CLAUSES * N_PATCHES);
        ull clause = clause_patch / (ull)N_PATCHES;
        int patch = clause_patch % N_PATCHES;

        int8_t* output = &patch_output[e * (ull)TOTAL_CLAUSES * N_PATCHES + clause * (ull)N_PATCHES + patch];

        // Empty clause matches all patches
        if (num_includes[clause] == 0) {
            *output = 1;
            continue;
        }

        // Skip invalid clauses (contradictions)
        if (is_clause_valid[clause] == 0) {
            *output = 0;
            continue;
        }

        // Check position bounds
        const int* pos = &clause_position_bounds[clause * 4];
        int py = patch / N_PATCHES_X;
        int px = patch % N_PATCHES_X;

        // Closed interval [pos[0], pos[1]]
        if (py < pos[0] || py > pos[1] || px < pos[2] || px > pos[3]) {
            *output = 0;
            continue;
        }

        // Use match_patch with sparse range representation
        const int* Xe = &X[e * (ull)HEIGHT * WIDTH * DEPTH];
        const int* cfb = &clause_feat_bounds[clause * (ull)N_RAW_PATCH_FEATS * 2];
        const int* cfids = &constrained_fids[clause * (ull)N_RAW_PATCH_FEATS];
        int n_cfids = n_constrained[clause];

        *output = match_patch(Xe, py, px, cfb, cfids, n_cfids) ? 1 : 0;
    }
}

} // extern "C"
