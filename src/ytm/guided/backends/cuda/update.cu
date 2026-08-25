#ifdef IS_NEOVIM_CLANGD_ENV
#include "common.cu"
#endif

#define FB_NONE 0
#define FB_T1A 1
#define FB_T1B 2
#define FB_T2 3

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

    for (int fid = lane; fid < N_RAW_PATCH_FEATS; fid += WARP_SIZE) {
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
        while (suc * WARP_SIZE + lane < N_LITERALS) {
            int li = suc * WARP_SIZE + lane;
            if (ta_state[li] > 0)
                ta_state[li] -= 1;
            suc += geom_sample(rng_key, rng_counter, S_INV);
        }
    } else {
        for (int li = lane; li < N_LITERALS; li += WARP_SIZE)
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

    for (int fid = lane; fid < N_RAW_PATCH_FEATS; fid += WARP_SIZE) {
        int lit_start = N_POSITION_FEATS + literal_offsets[fid];
        int lit_end = N_POSITION_FEATS + literal_offsets[fid + 1];
        int shifted_val = get_feature_value(X, patch_idx_y, patch_idx_x, fid) - feat_mins[fid];

        t2_inc_literals(ta_state, lit_start + shifted_val, lit_end, 0);

#if NEGATED_LITERALS
        t2_inc_literals(ta_state, lit_start, lit_start + shifted_val, N_LITERALS / 2);
#endif
    }
}

__device__ inline void apply_feedback(uint8_t fb_type, ull rng_key, uint* rng_counter, uint* ta_states,
                                      const int* Xe, int patch_idx_y, int patch_idx_x, const int* feat_mins,
                                      const int* literal_offsets, int lane) {
    if (fb_type == FB_T1A) {
#if TYPE1A_FB
        type1a_fb(rng_key, rng_counter, ta_states, Xe, patch_idx_y, patch_idx_x, feat_mins, literal_offsets, lane);
#endif
    } else if (fb_type == FB_T1B) {
#if TYPE1B_FB
        type1b_fb(rng_key, rng_counter, ta_states, lane);
#endif
    } else if (fb_type == FB_T2) {
#if TYPE2_FB
        type2_fb(ta_states, Xe, patch_idx_y, patch_idx_x, feat_mins, literal_offsets, lane);
#endif
    }
}

__device__ inline void decide_one(ull seed, int e, ull idx, ull clause, ull class_id, ull rel_clause,
                                  int clause_output, int clause_density, const float* grad,
                                  const float* clause_weights, float lambda_plus, float lambda_minus,
                                  uint8_t* feedback_type, int8_t* is_clause_synced) {
    ull rng_k = rng_hash(seed, idx, (ull)e, 0xD00D1E00ULL);
    uint rng_counter = 0;

    int target = (grad[class_id] > 0.0f) - (grad[class_id] < 0.0f);
    float lam = (target > 0) ? lambda_plus : lambda_minus;
    float update_prob = 1.0f - expf(-lam * fabsf(grad[class_id]));

    bool skip = (target == 0 || grad[class_id] == 0.0f || rand_uniform(rng_k, &rng_counter) > update_prob);
    if (skip) {
        feedback_type[idx] = FB_NONE;
        return;
    }

    is_clause_synced[clause] = 0;

    float weight_val = clause_weights[class_id * (ull)CLAUSES_PER_CLASS + rel_clause];
    int sign = (weight_val >= 0) - (weight_val < 0);
    bool has_space = (clause_density <= (int)MAX_INCLUDED_LITERALS);
    bool t1 = (target * sign) > 0;

    if (t1 && clause_output && has_space) {
        feedback_type[idx] = FB_T1A;
    } else if (t1 && !(clause_output && has_space)) {
        feedback_type[idx] = FB_T1B;
    } else if ((target * sign) < 0 && clause_output) {
        feedback_type[idx] = FB_T2;
    } else {
        feedback_type[idx] = FB_NONE;
    }
}

extern "C" __global__ void decide_feedback_and_update_weights(const ull seed, const int e, const float* grad, const float lr,
                                           float* clause_weights, const int* clause_density,
                                           const int* selected_patch_ids, const int8_t* clause_drop_mask,
                                           const float lambda_plus, const float lambda_minus, uint8_t* feedback_type,
                                           int8_t* is_clause_synced) {
    ull tid = (ull)blockIdx.x * blockDim.x + threadIdx.x;
    ull stride = (ull)blockDim.x * gridDim.x;

    for (ull clause = tid; clause < (ull)TOTAL_CLAUSES; clause += stride) {
        if (clause_drop_mask[clause] == 1) {
            for (ull c = 0; c < (ull)CLASSES; ++c)
                feedback_type[clause * (ull)CLASSES + c] = FB_NONE;
            continue;
        }

        int clause_output = (selected_patch_ids[clause] >= 0) ? 1 : 0;
        int cd = clause_density[clause];
        ull rel_clause = clause % (ull)CLAUSES_PER_CLASS;

#if COALESCED == 0
        ull class_id = clause / (ull)CLAUSES_PER_CLASS;
        ull idx = clause * (ull)CLASSES + class_id;
        decide_one(seed, e, idx, clause, class_id, rel_clause, clause_output, cd, grad, clause_weights, lambda_plus,
                  lambda_minus, feedback_type, is_clause_synced);
        if (clause_output)
            clause_weights[class_id * (ull)CLAUSES_PER_CLASS + rel_clause] += lr * grad[class_id];
#else
        for (ull class_id = 0; class_id < (ull)CLASSES; ++class_id) {
            ull idx = clause * (ull)CLASSES + class_id;
            decide_one(seed, e, idx, clause, class_id, rel_clause, clause_output, cd, grad, clause_weights,
                      lambda_plus, lambda_minus, feedback_type, is_clause_synced);
        }
        if (clause_output)
            for (ull class_id = 0; class_id < (ull)CLASSES; ++class_id)
                clause_weights[class_id * (ull)CLAUSES_PER_CLASS + rel_clause] += lr * grad[class_id];
#endif
    }
}

extern "C" __global__ void update_bias(const float* grad, const float lr, float* bias) {
#if BIAS
    ull tid = (ull)blockIdx.x * blockDim.x + threadIdx.x;
    ull stride = (ull)blockDim.x * gridDim.x;
    for (ull c = tid; c < (ull)CLASSES; c += stride)
        bias[c] += lr * grad[c];
#endif
}

extern "C" __global__ void update_clauses(const ull seed, const int* selected_patch_ids, const int* X, const int e,
                                          const int e_global, uint* global_ta_states, const int* feat_mins,
                                          const int* literal_offsets, const uint8_t* feedback_type) {
    ull tid = (ull)blockIdx.x * blockDim.x + threadIdx.x;
    ull warp_id = tid / WARP_SIZE;
    ull lane = tid % WARP_SIZE;
    ull total_warps = ((ull)gridDim.x * (ull)blockDim.x) / WARP_SIZE;

    const int* Xe = &X[(ull)e * HEIGHT * WIDTH * DEPTH];

    for (ull clause = warp_id; clause < (ull)TOTAL_CLAUSES; clause += total_warps) {
        ull rng_id = clause * WARP_SIZE + lane;
        ull rng_k = rng_hash(seed, rng_id, (ull)e_global, 0xCAFEBABEULL);
        uint rng_counter = 0;

        int patch_id = selected_patch_ids[clause];
        int clause_output = (patch_id >= 0) ? 1 : 0;
        int patch_idx_y = -1, patch_idx_x = -1;
        if (clause_output) {
            patch_idx_y = patch_id / N_PATCHES_X;
            patch_idx_x = patch_id % N_PATCHES_X;
        }

        uint* ta_states = &global_ta_states[clause * (ull)N_LITERALS];

#if COALESCED == 0
        ull class_id = clause / (ull)CLAUSES_PER_CLASS;
        uint8_t fb = feedback_type[clause * (ull)CLASSES + class_id];
        apply_feedback(fb, rng_k, &rng_counter, ta_states, Xe, patch_idx_y, patch_idx_x, feat_mins, literal_offsets,
                       lane);
#else
        for (ull class_id = 0; class_id < (ull)CLASSES; ++class_id) {
            uint8_t fb = feedback_type[clause * (ull)CLASSES + class_id];
            apply_feedback(fb, rng_k, &rng_counter, ta_states, Xe, patch_idx_y, patch_idx_x, feat_mins,
                           literal_offsets, lane);
        }
#endif
    }
}
