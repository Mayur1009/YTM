#ifdef IS_NEOVIM_CLANGD_ENV
#include "common.cu"
#endif

__device__ inline int geometric_sample(ull rng_key, uint* rng_counter, float p) {
    float u = rand_uniform(rng_key, rng_counter);
    float p_clamp = fmaxf(p, 1e-9f);
    return (int)(logf(1.0f - u + 1e-9f) / logf(1.0f - p_clamp)) + 1;
}

__device__ inline void literal_dec_with_p(ull rng_key, uint* rng_counter, uint* ta_state, int start, int end,
                                          int offset, float p) {
    int li = start + geometric_sample(rng_key, rng_counter, p) - 1;
    while (li < end) {
        if (ta_state[li + offset] > 0)
            ta_state[li + offset] -= 1;
        li += geometric_sample(rng_key, rng_counter, p);
    }
}

__device__ inline void literal_inc(uint* ta_state, int start, int end, int offset, uint max_val) {
    for (int li = start; li < end; ++li) {
        ta_state[li + offset] += (ta_state[li + offset] < max_val);
    }
}

__device__ inline void literal_inc_maybe_p(ull rng_key, uint* rng_counter, uint* ta_state, int start, int end,
                                           int offset, float p) {
#if BOOST_TP_FB
    literal_inc(ta_state, start, end, offset, MAX_TA_STATE);
#else
    int li = start + geometric_sample(rng_key, rng_counter, p) - 1;
    while (li < end) {
        if (ta_state[li + offset] < MAX_TA_STATE)
            ta_state[li + offset] += 1;
        li += geometric_sample(rng_key, rng_counter, p);
    }
#endif
}

__device__ inline void type1a_fb(ull rng_key, uint* rng_counter, uint* ta_states, const int* X, int patch_idx_y,
                                 int patch_idx_x, int sign, const int* feat_mins, const int* literal_offsets,
                                 int lane) {

#if POSITION_LITERALS
    if (lane == 0) {
        literal_inc_maybe_p(rng_key, rng_counter, ta_states, 0, patch_idx_y, 0, 1.0f - S_INV);
        literal_dec_with_p(rng_key, rng_counter, ta_states, patch_idx_y, N_POSITION_FEATS_Y, 0, S_INV);

        literal_inc_maybe_p(rng_key, rng_counter, ta_states, N_POSITION_FEATS_Y, N_POSITION_FEATS_Y + patch_idx_x, 0,
                            1.0f - S_INV);
        literal_dec_with_p(rng_key, rng_counter, ta_states, N_POSITION_FEATS_Y + patch_idx_x, N_POSITION_FEATS, 0,
                           S_INV);

#if NEGATED_LITERALS
        literal_dec_with_p(rng_key, rng_counter, ta_states, 0, patch_idx_y, N_LITERALS / 2, S_INV);
        literal_inc_maybe_p(rng_key, rng_counter, ta_states, patch_idx_y, N_POSITION_FEATS_Y, N_LITERALS / 2,
                            1.0f - S_INV);

        literal_dec_with_p(rng_key, rng_counter, ta_states, N_POSITION_FEATS_Y, N_POSITION_FEATS_Y + patch_idx_x,
                           N_LITERALS / 2, S_INV);
        literal_inc_maybe_p(rng_key, rng_counter, ta_states, N_POSITION_FEATS_Y + patch_idx_x, N_POSITION_FEATS,
                            N_LITERALS / 2, 1.0f - S_INV);
#endif
    }
#endif

    for (int fid = lane; fid < N_RAW_PATCH_FEATS; fid += 32) {
        int lit_start = N_POSITION_FEATS + literal_offsets[fid];
        int lit_end = N_POSITION_FEATS + literal_offsets[fid + 1];
        int shifted_val = get_feature_value(X, patch_idx_y, patch_idx_x, fid) - feat_mins[fid];

        literal_inc_maybe_p(rng_key, rng_counter, ta_states, lit_start, lit_start + shifted_val, 0, 1.0f - S_INV);
        literal_dec_with_p(rng_key, rng_counter, ta_states, lit_start + shifted_val, lit_end, 0, S_INV);

#if NEGATED_LITERALS
        literal_dec_with_p(rng_key, rng_counter, ta_states, lit_start, lit_start + shifted_val, N_LITERALS / 2, S_INV);
        literal_inc_maybe_p(rng_key, rng_counter, ta_states, lit_start + shifted_val, lit_end, N_LITERALS / 2,
                            1.0f - S_INV);
#endif
    }
}

__device__ inline void type1b_fb(ull rng_key, uint* rng_counter, uint* ta_state, int lane) {
    int suc = geometric_sample(rng_key, rng_counter, S_INV) - 1;
    while (suc * 32 + lane < N_LITERALS) {
        int li = suc * 32 + lane;
        if (ta_state[li] > 0)
            ta_state[li] -= 1;
        suc += geometric_sample(rng_key, rng_counter, S_INV);
    }
}

__device__ inline void type2_fb(uint* ta_state, const int* X, int patch_idx_y, int patch_idx_x, const int* feat_mins,
                                const int* literal_offsets, int lane) {
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

    for (int fid = lane; fid < N_RAW_PATCH_FEATS; fid += 32) {
        int lit_start = N_POSITION_FEATS + literal_offsets[fid];
        int lit_end = N_POSITION_FEATS + literal_offsets[fid + 1];
        int shifted_val = get_feature_value(X, patch_idx_y, patch_idx_x, fid) - feat_mins[fid];

        literal_inc(ta_state, lit_start + shifted_val, lit_end, 0, INCLUDE_STATE);

#if NEGATED_LITERALS
        literal_inc(ta_state, lit_start, lit_start + shifted_val, N_LITERALS / 2, INCLUDE_STATE);
#endif
    }
}

__device__ inline void update_clause_class(const warp_t& warp, int lane, ull class_id, ull clause, ull rel_clause,
                                           int clause_output, int patch_idx_y, int patch_idx_x, int clause_density,
                                           uint* ta_states, float* clause_weights, const int* Xe,
                                           const float* targets_e, const float* prob, const int* feat_mins,
                                           const int* literal_offsets, int8_t* is_clause_synced, ull rng_k,
                                           uint* rng_counter) {
    bool skip = false;
    if (lane == 0) {
        skip = ((targets_e[class_id] < 0.0f && rand_uniform(rng_k, rng_counter) > (Q / fmaxf(1.0f, (CLASSES - 1)))) ||
                prob[class_id] == 0.0f || rand_uniform(rng_k, rng_counter) > fabsf(prob[class_id]));
    }

    if (warp.any(skip))
        return;

    int target = (prob[class_id] > 0.0f) ? 1 : -1;
    is_clause_synced[clause] = 0;

    float* weight = &clause_weights[class_id * (ull)CLAUSES_PER_CLASS + rel_clause];
    int sign = (*weight >= 0) - (*weight < 0);
    bool has_space = (clause_density <= (int)MAX_INCLUDED_LITERALS);
    bool t1 = (target * sign) > 0;

    if (t1 && clause_output && has_space) {
#if TYPE1A_FB
#if WEIGHTED
        if (lane == 0 && fabsf(*weight) < MAX_WEIGHT)
            (*weight) += sign * 1.0f;
#endif
        type1a_fb(rng_k, rng_counter, ta_states, Xe, patch_idx_y, patch_idx_x, sign, feat_mins, literal_offsets, lane);
#endif
    }

    else if (t1 && !(clause_output && has_space)) {
#if TYPE1B_FB
        type1b_fb(rng_k, rng_counter, ta_states, lane);
#endif
    }

    else if ((target * sign) < 0 && clause_output) {
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

        type2_fb(ta_states, Xe, patch_idx_y, patch_idx_x, feat_mins, literal_offsets, lane);
#endif
    }
}

__device__ inline float uprob_fun(float y, float y_hat) { return (y - y_hat) / (T_MAX - T_MIN); }

extern "C" __global__ void calc_update_prob(const float* votes, const float* targets, const int e, float* prob) {
    ull tid = threadIdx.x + blockIdx.x * blockDim.x;
    ull stride = blockDim.x * gridDim.x;
    for (ull class_id = tid; class_id < (ull)CLASSES; class_id += stride) {
        float target = targets[(ull)e * CLASSES + class_id];
        float v = clip(votes[class_id], T_MIN, T_MAX);
        prob[class_id] = uprob_fun(target, v);
    }
}

extern "C" __global__ void update_clauses(const ull seed, const int* selected_patch_ids, const int* clause_density,
                                          const int8_t* clause_drop_mask, const int* X, const float* targets,
                                          const int e, const float* prob, uint* global_ta_states, float* clause_weights,
                                          const int* feat_mins, const int* literal_offsets, int8_t* is_clause_synced) {
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    auto grid = cg::this_grid();
    ull tid = grid.thread_rank();
    int lane = warp.thread_rank();
    ull warp_id = tid / warp.size();
    ull total_warps = grid.size() / warp.size();

    const int* Xe = &X[(ull)e * HEIGHT * WIDTH * DEPTH];
    const float* targets_e = &targets[(ull)e * CLASSES];

    ull rng_k = rng_hash(seed, tid, (ull)e, 0xCAFEBABEULL);
    uint rng_counter = 0;

    for (ull clause = warp_id; clause < (ull)TOTAL_CLAUSES; clause += total_warps) {
        if (clause_drop_mask[clause] == 1)
            continue;

        uint* ta_states = &global_ta_states[clause * (ull)N_LITERALS];
        int patch_id = selected_patch_ids[clause];
        int clause_output = (patch_id >= 0) ? 1 : 0;

        int patch_idx_y = -1, patch_idx_x = -1;
        if (clause_output) {
            patch_idx_y = patch_id / N_PATCHES_X;
            patch_idx_x = patch_id % N_PATCHES_X;
        }

        int cd = clause_density[clause];

        ull rel_clause = clause % (ull)CLAUSES_PER_CLASS;
#if COALESCED == 0
        ull class_id = clause / (ull)CLAUSES_PER_CLASS;
        update_clause_class(warp, lane, class_id, clause, rel_clause, clause_output, patch_idx_y, patch_idx_x, cd,
                            ta_states, clause_weights, Xe, targets_e, prob, feat_mins, literal_offsets,
                            is_clause_synced, rng_k, &rng_counter);
#else
        for (ull class_id = 0; class_id < (ull)CLASSES; ++class_id) {
            update_clause_class(warp, lane, class_id, clause, rel_clause, clause_output, patch_idx_y, patch_idx_x, cd,
                                ta_states, clause_weights, Xe, targets_e, prob, feat_mins, literal_offsets,
                                is_clause_synced, rng_k, &rng_counter);
        }
#endif
    }
}
