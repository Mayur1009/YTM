#ifdef IS_NEOVIM_CLANGD_ENV
#include "common.cu"
#include "losses.cu"
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

__device__ inline void dec_literals(ull rng_key, uint* rng_counter, int start, int end, int offset, int lane,
                                    uint* ta_state) {
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

__device__ inline void t2_inc_literals(int start, int end, int offset, uint* ta_state) {
    for (int li = start; li < end; ++li) {
        if (ta_state[li + offset] < MAX_TA_STATE)
            ta_state[li + offset] += 1;
    }
}

__device__ inline void t1a_inc_literals(ull rng_key, uint* rng_counter, int start, int end, int offset,
                                        uint* ta_state) {
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

__device__ inline void type1a_fb(ull rng_key, uint* rng_counter, const int* X, int patch_idx_y, int patch_idx_x,
                                 const int* feat_mins, const int* literal_offsets, int lane, ull tile_id,
                                 uint* ta_states) {

#if POSITION_LITERALS
    if (tile_id == 0 && lane == 0) {
        t1a_inc_literals(rng_key, rng_counter, 0, patch_idx_y, 0, ta_states);
        t1a_inc_literals(rng_key, rng_counter, N_POSITION_FEATS_Y, N_POSITION_FEATS_Y + patch_idx_x, 0, ta_states);

        dec_literals(rng_key, rng_counter, patch_idx_y, N_POSITION_FEATS_Y, 0, lane, ta_states);
        dec_literals(rng_key, rng_counter, N_POSITION_FEATS_Y + patch_idx_x, N_POSITION_FEATS, 0, lane, ta_states);

#if NEGATED_LITERALS
        t1a_inc_literals(rng_key, rng_counter, patch_idx_y, N_POSITION_FEATS_Y, N_LITERALS / 2, ta_states);
        t1a_inc_literals(rng_key, rng_counter, N_POSITION_FEATS_Y + patch_idx_x, N_POSITION_FEATS, N_LITERALS / 2,
                         ta_states);

        dec_literals(rng_key, rng_counter, 0, patch_idx_y, N_LITERALS / 2, lane, ta_states);
        dec_literals(rng_key, rng_counter, N_POSITION_FEATS_Y, N_POSITION_FEATS_Y + patch_idx_x, N_LITERALS / 2, lane,
                     ta_states);
#endif
    }
#endif

    for (ull fid = tile_id * WARP_SIZE + lane; fid < N_RAW_PATCH_FEATS; fid += WARPS_PER_CLAUSE * WARP_SIZE) {
        int lit_start = N_POSITION_FEATS + literal_offsets[fid];
        int lit_end = N_POSITION_FEATS + literal_offsets[fid + 1];
        int shifted_val = get_feature_value(X, patch_idx_y, patch_idx_x, fid) - feat_mins[fid];

        t1a_inc_literals(rng_key, rng_counter, lit_start, lit_start + shifted_val, 0, ta_states);
        dec_literals(rng_key, rng_counter, lit_start + shifted_val, lit_end, 0, lane, ta_states);

#if NEGATED_LITERALS
        t1a_inc_literals(rng_key, rng_counter, lit_start + shifted_val, lit_end, N_LITERALS / 2, ta_states);
        dec_literals(rng_key, rng_counter, lit_start, lit_start + shifted_val, N_LITERALS / 2, lane, ta_states);
#endif
    }
}

__device__ inline void type1b_fb(ull rng_key, uint* rng_counter, int lane, ull tile_id, uint* ta_state) {
    ull global_lane = tile_id * WARP_SIZE + lane;
    ull total_lanes = WARPS_PER_CLAUSE * WARP_SIZE;

    if (S > 1.0f) {
        int suc = geom_sample(rng_key, rng_counter, S_INV) - 1;
        while (suc * total_lanes + global_lane < N_LITERALS) {
            ull li = suc * total_lanes + global_lane;
            if (ta_state[li] > 0)
                ta_state[li] -= 1;
            suc += geom_sample(rng_key, rng_counter, S_INV);
        }
    } else {
        for (ull li = global_lane; li < N_LITERALS; li += total_lanes)
            if (ta_state[li] > 0)
                ta_state[li] -= 1;
    }
}

__device__ inline void type2_fb(const int* X, int patch_idx_y, int patch_idx_x, const int* feat_mins,
                                const int* literal_offsets, int lane, ull tile_id, uint* ta_state) {
#if POSITION_LITERALS
    if (tile_id == 0 && lane == 0) {
        t2_inc_literals(patch_idx_y, N_POSITION_FEATS_Y, 0, ta_state);
        t2_inc_literals(N_POSITION_FEATS_Y + patch_idx_x, N_POSITION_FEATS, 0, ta_state);

#if NEGATED_LITERALS
        t2_inc_literals(0, patch_idx_y, N_LITERALS / 2, ta_state);
        t2_inc_literals(N_POSITION_FEATS_Y, N_POSITION_FEATS_Y + patch_idx_x, N_LITERALS / 2, ta_state);
#endif
    }
#endif

    for (ull fid = tile_id * WARP_SIZE + lane; fid < N_RAW_PATCH_FEATS; fid += WARPS_PER_CLAUSE * WARP_SIZE) {
        int lit_start = N_POSITION_FEATS + literal_offsets[fid];
        int lit_end = N_POSITION_FEATS + literal_offsets[fid + 1];
        int shifted_val = get_feature_value(X, patch_idx_y, patch_idx_x, fid) - feat_mins[fid];

        t2_inc_literals(lit_start + shifted_val, lit_end, 0, ta_state);

#if NEGATED_LITERALS
        t2_inc_literals(lit_start, lit_start + shifted_val, N_LITERALS / 2, ta_state);
#endif
    }
}

__device__ inline void apply_feedback(ull rng_key, uint* rng_counter, uint8_t fb_type, const int* Xe, int patch_idx_y,
                                      int patch_idx_x, const int* feat_mins, const int* literal_offsets, int lane,
                                      ull tile_id, uint* ta_states) {
    if (fb_type == FB_T1A) {
#if TYPE1A_FB
        type1a_fb(rng_key, rng_counter, Xe, patch_idx_y, patch_idx_x, feat_mins, literal_offsets, lane, tile_id,
                  ta_states);
#endif
    } else if (fb_type == FB_T1B) {
#if TYPE1B_FB
        type1b_fb(rng_key, rng_counter, lane, tile_id, ta_states);
#endif
    } else if (fb_type == FB_T2) {
#if TYPE2_FB
        type2_fb(Xe, patch_idx_y, patch_idx_x, feat_mins, literal_offsets, lane, tile_id, ta_states);
#endif
    }
}

extern "C" __global__ void compute_votes_neg_ck(const float* votes, const float* clause_weights,
                                                const int* selected_patch_ids, float* votes_neg_ck) {

    ull tid = (ull)blockIdx.x * blockDim.x + threadIdx.x;
    ull stride = (ull)blockDim.x * gridDim.x;

    for (ull clause_class = tid; clause_class < (ull)TOTAL_CLAUSES * (ull)CLASSES; clause_class += stride) {
        ull clause_id = clause_class / (ull)CLASSES;
        ull class_id = clause_class % (ull)CLASSES;

#if COALESCED == 0
        if (class_id != clause_id / (ull)CLAUSES_PER_CLASS) {
            votes_neg_ck[clause_class] = votes[class_id];
            continue;
        }
#endif
        int ck = (selected_patch_ids[clause_id] >= 0) ? 1 : 0;
        ull rel_clause = clause_id % (ull)CLAUSES_PER_CLASS;
        float w = clause_weights[class_id * (ull)CLAUSES_PER_CLASS + rel_clause];
        votes_neg_ck[clause_class] = votes[class_id] - (ck ? w : -w);
    }
}

extern "C" __global__ void compute_loss_neg_ck(const float* y_hat_neg_ck, const float* y,
                                               const float* loss_class_weights, float* loss_neg_ck) {
    ull tid = (ull)blockIdx.x * blockDim.x + threadIdx.x;
    ull stride = (ull)blockDim.x * gridDim.x;

    for (ull clause_id = tid; clause_id < (ull)TOTAL_CLAUSES; clause_id += stride) {
        float loss = 0.0f;
        loss_gradient_impl(&y_hat_neg_ck[clause_id * (ull)CLASSES], y, loss_class_weights, nullptr, &loss);
        loss_neg_ck[clause_id] = loss;
    }
}

__device__ inline uint8_t select_fb_delta_l(const ull seed, ull rng_id, float uprob, float delta_L, int ck,
                                            bool has_space) {
    ull rng_k = rng_hash(seed, rng_id, 0xFEEDFACEULL, 0xD00D1E00ULL);
    uint rng_counter = 0;

    uint8_t fb;
    if (rand_uniform(rng_k, &rng_counter) > uprob) {
        fb = FB_NONE;
    } else {
        // T1a -> ck = 1, and deltaL < 0, meaning turning ck=0 increased the loss
        // T1b -> ck = 0, and deltaL > 0, meaning turning ck=1 decreased the loss
        // T2 -> ck = 1, and deltaL > 0, meaning turning ck=0 decreased the loss
        // None -> ck = 0, and deltaL < 0, meaning turning ck = 1 increased the loss.
        bool t1a = (ck == 1 && delta_L < 0 && has_space);
        bool t1b = ((ck == 0 && delta_L > 0) || (ck == 1 && delta_L < 0 && !has_space));
        bool t2 = (ck == 1 && delta_L > 0);
        fb = t1a ? FB_T1A : (t1b ? FB_T1B : (t2 ? FB_T2 : FB_NONE));
    }
    return fb;
}

extern "C" __global__ void decide_feedback_delta_l(const ull seed, const float* loss, const float* loss_neg_ck,
                                                   const int* clause_density, const int* selected_patch_ids,
                                                   const int8_t* clause_drop_mask, const float lambda_,
                                                   uint8_t* feedback_type) {
    ull tid = (ull)blockIdx.x * blockDim.x + threadIdx.x;
    ull stride = (ull)blockDim.x * gridDim.x;

    for (ull clause_id = tid; clause_id < (ull)TOTAL_CLAUSES; clause_id += stride) {
        if (clause_drop_mask[clause_id] == 1) {
            feedback_type[clause_id] = FB_NONE;
            continue;
        }

        int ck = (selected_patch_ids[clause_id] >= 0) ? 1 : 0;
        bool has_space = (clause_density[clause_id] <= (int)MAX_INCLUDED_LITERALS);
        float delta_L = *loss - loss_neg_ck[clause_id];
        float update_prob = 1.0f - expf(-lambda_ * fabsf(delta_L));
        feedback_type[clause_id] = select_fb_delta_l(seed, clause_id, update_prob, delta_L, ck, has_space);
    }
}

extern "C" __global__ void decide_feedback_grad(const ull seed, const float* grad, const float* clause_weights,
                                                const int* clause_density, const int* selected_patch_ids,
                                                const int8_t* clause_drop_mask, const float lambda_,
                                                uint8_t* feedback_type) {
    ull tid = (ull)blockIdx.x * blockDim.x + threadIdx.x;
    ull stride = (ull)blockDim.x * gridDim.x;

    for (ull clause_class = tid; clause_class < (ull)TOTAL_CLAUSES * (ull)CLASSES; clause_class += stride) {
        ull clause_id = clause_class / (ull)CLASSES;
        ull class_id = clause_class % (ull)CLASSES;

#if COALESCED == 0
        if (class_id != clause_id / (ull)CLAUSES_PER_CLASS)
            continue;
#endif

        if (clause_drop_mask[clause_id] == 1) {
            feedback_type[clause_class] = FB_NONE;
            continue;
        }

        int ck = (selected_patch_ids[clause_id] >= 0) ? 1 : 0;
        bool has_space = (clause_density[clause_id] <= (int)MAX_INCLUDED_LITERALS);
        ull rel_clause = clause_id % (ull)CLAUSES_PER_CLASS;
        float w = clause_weights[class_id * (ull)CLAUSES_PER_CLASS + rel_clause];
        float g = grad[class_id];

        int polarity = (w >= 0) - (w < 0);
        int sign_grad = (g >= 0) - (g < 0);
        int dir = polarity * sign_grad;
        float update_prob = 1.0f - expf(-lambda_ * fabsf(g));

        ull rng_k = rng_hash(seed, clause_class, 0xFEEDFACEULL, 0xD00D1E00ULL);
        uint rng_counter = 0;

        uint8_t fb;
        if (rand_uniform(rng_k, &rng_counter) > update_prob) {
            fb = FB_NONE;
        } else {
            bool t1a = (ck == 1 && dir > 0 && has_space);
            bool t1b = ((ck == 0 && dir > 0) || (ck == 1 && dir > 0 && !has_space));
            bool t2 = (ck == 1 && dir < 0);
            fb = t1a ? FB_T1A : (t1b ? FB_T1B : (t2 ? FB_T2 : FB_NONE));
        }
        feedback_type[clause_class] = fb;
    }
}

extern "C" __global__ void update_weights(const float* grad, const float lr, const int* selected_patch_ids,
                                          const int8_t* clause_drop_mask, float* clause_weights) {
    ull tid = (ull)blockIdx.x * blockDim.x + threadIdx.x;
    ull stride = (ull)blockDim.x * gridDim.x;

    for (ull clause = tid; clause < (ull)TOTAL_CLAUSES; clause += stride) {
        if (clause_drop_mask[clause] == 1)
            continue;

        int ck = selected_patch_ids[clause] >= 0;
        if (ck == 0)
            continue;

        ull rel_clause = clause % (ull)CLAUSES_PER_CLASS;
        ull class_id = 0;
        LOOP_CLASS_ID(class_id, clause) {
            ull widx = class_id * (ull)CLAUSES_PER_CLASS + rel_clause;
            clause_weights[widx] = clip(clause_weights[widx] + lr * grad[class_id] * ck, -MAX_WEIGHT, MAX_WEIGHT);
        }
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
                                          const int* feat_mins, const int* literal_offsets,
                                          const uint8_t* feedback_type, uint* global_ta_states,
                                          int8_t* is_clause_synced) {
    ull tid = (ull)blockIdx.x * blockDim.x + threadIdx.x;
    ull warp_id = tid / WARP_SIZE;
    ull lane = tid % WARP_SIZE;
    ull total_warps = ((ull)gridDim.x * (ull)blockDim.x) / WARP_SIZE;

    const int* Xe = &X[(ull)e * HEIGHT * WIDTH * DEPTH];

    ull total_tiles = (ull)TOTAL_CLAUSES * WARPS_PER_CLAUSE;
    for (ull gtile = warp_id; gtile < total_tiles; gtile += total_warps) {
        ull clause = gtile / (ull)WARPS_PER_CLAUSE;
        ull tile_id = gtile % (ull)WARPS_PER_CLAUSE;

        ull rng_id = gtile * WARP_SIZE + lane;
        ull rng_k = rng_hash(seed, rng_id, 0xC0DEBA5EULL, 0xCAFEBABEULL);
        uint rng_counter = 0;

        int patch_id = selected_patch_ids[clause];
        int clause_output = (patch_id >= 0) ? 1 : 0;
        int patch_idx_y = -1, patch_idx_x = -1;
        if (clause_output) {
            patch_idx_y = patch_id / N_PATCHES_X;
            patch_idx_x = patch_id % N_PATCHES_X;
        }

        uint* ta_states = &global_ta_states[clause * (ull)N_LITERALS];
#if FB_SIGNAL == FB_SIGNAL_GRAD
        ull class_id = 0;
        LOOP_CLASS_ID(class_id, clause) {
            uint8_t fb = feedback_type[clause * (ull)CLASSES + class_id];
            apply_feedback(rng_k, &rng_counter, fb, Xe, patch_idx_y, patch_idx_x, feat_mins, literal_offsets, lane,
                           tile_id, ta_states);
            if (fb != FB_NONE)
                is_clause_synced[clause] = 0;
        }
#else
        uint8_t fb = feedback_type[clause];
        apply_feedback(rng_k, &rng_counter, fb, Xe, patch_idx_y, patch_idx_x, feat_mins, literal_offsets, lane, tile_id,
                       ta_states);
        if (fb != FB_NONE)
            is_clause_synced[clause] = 0;
#endif
    }
}
