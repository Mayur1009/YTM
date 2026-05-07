#ifdef IS_NEOVIM_CLANGD_ENV
#include "common.cu"
#endif

__device__ inline int geom_sample(ull rng_key, uint* rng_counter, float p) {
    float u = rand_uniform(rng_key, rng_counter);
    double u_clamp = clip(u, 1e-7f, 1.0f - 1e-7f);
    double log_u = log1p(-u_clamp);
    double log_p = log1p(-p);
    int sample = (int)(log_u / log_p) + 1;
    return sample;
}

__device__ inline void dec_literals(ull rng_key, uint* rng_counter, uint* ta_state, int start, int end, int offset,
                                    int lane) {
    if (S > 1.0f) {
        int li = start + geom_sample(rng_key, rng_counter, S_INV) - 1;
        while (li < end) {
            if (ta_state[li + offset] > 0)
                ta_state[li + offset] -= 1;
            li += geom_sample(rng_key, rng_counter, S_INV);
        }
    } else {
        for (int li = start; li < end; ++li)
            if (ta_state[li + offset] > 0)
                ta_state[li + offset] -= 1;
    }
}

__device__ inline void t2_inc_literals(uint* ta_state, int start, int end, int offset) {
    for (int li = start; li < end; ++li) {
        if (ta_state[li + offset] < MAX_TA_STATE)
            ta_state[li + offset] += 1;
    }
}

__device__ inline void t1a_inc_literals(ull rng_key, uint* rng_counter, uint* ta_state, int start, int end,
                                        int offset) {
#if BOOST_TP_FB
    for (int li = start; li < end; ++li)
        if (ta_state[li + offset] < MAX_TA_STATE)
            ta_state[li + offset] += 1;
#else
    if (S > 1.0f) {
        int li = start + geom_sample(rng_key, rng_counter, 1 - S_INV) - 1;
        while (li < end) {
            if (ta_state[li + offset] < MAX_TA_STATE)
                ta_state[li + offset] += 1;
            li += geom_sample(rng_key, rng_counter, 1 - S_INV);
        }
    }
#endif
}

__device__ inline void type1a_fb(ull rng_key, uint* rng_counter, uint* ta_states, const int* X, int patch_idx_y,
                                 int patch_idx_x, const int* feat_mins, const int* literal_offsets, int lane) {

#if POSITION_LITERALS
    if (lane == 0) {
        t1a_inc_literals(rng_key, rng_counter, ta_states, 0, patch_idx_y, 0);
        t1a_inc_literals(rng_key, rng_counter, ta_states, N_POSITION_FEATS_Y, N_POSITION_FEATS_Y + patch_idx_x, 0);

        dec_literals(rng_key, rng_counter, ta_states, patch_idx_y, N_POSITION_FEATS_Y, 0, lane);
        dec_literals(rng_key, rng_counter, ta_states, N_POSITION_FEATS_Y + patch_idx_x, N_POSITION_FEATS, 0, lane);

#if NEGATED_LITERALS
        t1a_inc_literals(rng_key, rng_counter, ta_states, patch_idx_y, N_POSITION_FEATS_Y, N_LITERALS / 2);
        t1a_inc_literals(rng_key, rng_counter, ta_states, N_POSITION_FEATS_Y + patch_idx_x, N_POSITION_FEATS,
                         N_LITERALS / 2);

        dec_literals(rng_key, rng_counter, ta_states, 0, patch_idx_y, N_LITERALS / 2, lane);
        dec_literals(rng_key, rng_counter, ta_states, N_POSITION_FEATS_Y, N_POSITION_FEATS_Y + patch_idx_x,
                     N_LITERALS / 2, lane);
#endif
    }
#endif

    for (int fid = lane; fid < N_RAW_PATCH_FEATS; fid += 32) {
        int lit_start = N_POSITION_FEATS + literal_offsets[fid];
        int lit_end = N_POSITION_FEATS + literal_offsets[fid + 1];
        int shifted_val = get_feature_value(X, patch_idx_y, patch_idx_x, fid) - feat_mins[fid];

        t1a_inc_literals(rng_key, rng_counter, ta_states, lit_start, lit_start + shifted_val, 0);
        dec_literals(rng_key, rng_counter, ta_states, lit_start + shifted_val, lit_end, 0, lane);

#if NEGATED_LITERALS
        t1a_inc_literals(rng_key, rng_counter, ta_states, lit_start + shifted_val, lit_end, N_LITERALS / 2);
        dec_literals(rng_key, rng_counter, ta_states, lit_start, lit_start + shifted_val, N_LITERALS / 2, lane);
#endif
    }
}

__device__ inline void type1b_fb(ull rng_key, uint* rng_counter, uint* ta_state, int lane) {
    if (S > 1.0f) {
        int suc = geom_sample(rng_key, rng_counter, S_INV) - 1;
        while (suc * 32 + lane < N_LITERALS) {
            int li = suc * 32 + lane;
            if (ta_state[li] > 0)
                ta_state[li] -= 1;
            suc += geom_sample(rng_key, rng_counter, S_INV);
        }
    } else {
        for (int li = lane; li < N_LITERALS; li += 32)
            if (ta_state[li] > 0)
                ta_state[li] -= 1;
    }
}

__device__ inline void type2_fb(uint* ta_state, const int* X, int patch_idx_y, int patch_idx_x, const int* feat_mins,
                                const int* literal_offsets, int lane) {
#if POSITION_LITERALS
    if (lane == 0) {
        t2_inc_literals(ta_state, patch_idx_y, N_POSITION_FEATS_Y, 0);
        t2_inc_literals(ta_state, N_POSITION_FEATS_Y + patch_idx_x, N_POSITION_FEATS, 0);

#if NEGATED_LITERALS
        t2_inc_literals(ta_state, 0, patch_idx_y, N_LITERALS / 2);
        t2_inc_literals(ta_state, N_POSITION_FEATS_Y, N_POSITION_FEATS_Y + patch_idx_x, N_LITERALS / 2);
#endif
    }
#endif

    for (int fid = lane; fid < N_RAW_PATCH_FEATS; fid += 32) {
        int lit_start = N_POSITION_FEATS + literal_offsets[fid];
        int lit_end = N_POSITION_FEATS + literal_offsets[fid + 1];
        int shifted_val = get_feature_value(X, patch_idx_y, patch_idx_x, fid) - feat_mins[fid];

        t2_inc_literals(ta_state, lit_start + shifted_val, lit_end, 0);

#if NEGATED_LITERALS
        t2_inc_literals(ta_state, lit_start, lit_start + shifted_val, N_LITERALS / 2);
#endif
    }
}

__device__ inline void update_clause_class(const warp_t& warp, int lane, ull class_id, ull clause, ull rel_clause,
                                           int clause_output, int patch_idx_y, int patch_idx_x, int clause_density,
                                           uint* ta_states, float* clause_weights, const int* Xe,
                                           const float* encoded_Y_e, const float* prob, float label_prob_c,
                                           const int* feat_mins, const int* literal_offsets, int8_t* is_clause_synced,
                                           ull rng_k, uint* rng_counter) {
    float update_prob = fabsf(prob[class_id]);
    int target = (prob[class_id] > 0.0f) - (prob[class_id] < 0.0f);
    bool skip = false;
    if (lane == 0) {
        skip = (rand_uniform(rng_k, rng_counter) > label_prob_c || target == 0 ||
                prob[class_id] == 0.0f || rand_uniform(rng_k, rng_counter) > update_prob);
    }

    if (warp.any(skip))
        return;

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
        type1a_fb(rng_k, rng_counter, ta_states, Xe, patch_idx_y, patch_idx_x, feat_mins, literal_offsets, lane);
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

extern "C" __global__ void calc_update_prob(const float* votes, const float* encoded_Y, const int e, float* prob) {
    ull tid = threadIdx.x + blockIdx.x * blockDim.x;
    ull stride = blockDim.x * gridDim.x;
    for (ull class_id = tid; class_id < (ull)CLASSES; class_id += stride) {
        float target = encoded_Y[(ull)e * CLASSES + class_id];
        float v = clip(votes[class_id], T_MIN, T_MAX);
        prob[class_id] = uprob_fun(target, v);
    }
}

extern "C" __global__ void update_clauses(const ull seed, const int* selected_patch_ids, const int* clause_density,
                                          const int8_t* clause_drop_mask, const int* X, const float* encoded_Y,
                                          const int e, const float* prob, const float* label_probs,
                                          uint* global_ta_states, float* clause_weights, const int* feat_mins,
                                          const int* literal_offsets, int8_t* is_clause_synced) {
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    auto grid = cg::this_grid();
    ull tid = grid.thread_rank();
    int lane = warp.thread_rank();
    ull warp_id = tid / warp.size();
    ull total_warps = grid.size() / warp.size();

    const int* Xe = &X[(ull)e * HEIGHT * WIDTH * DEPTH];
    const float* encoded_Y_e = &encoded_Y[(ull)e * CLASSES];
    const float* label_probs_e = &label_probs[(ull)e * CLASSES];

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
                            ta_states, clause_weights, Xe, encoded_Y_e, prob, label_probs_e[class_id], feat_mins,
                            literal_offsets, is_clause_synced, rng_k, &rng_counter);
#else
        for (ull class_id = 0; class_id < (ull)CLASSES; ++class_id) {
            update_clause_class(warp, lane, class_id, clause, rel_clause, clause_output, patch_idx_y, patch_idx_x, cd,
                                ta_states, clause_weights, Xe, encoded_Y_e, prob, label_probs_e[class_id], feat_mins,
                                literal_offsets, is_clause_synced, rng_k, &rng_counter);
        }
#endif
    }
}
