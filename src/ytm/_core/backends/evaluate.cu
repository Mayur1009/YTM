#ifdef IS_NEOVIM_CLANGD_ENV
#include "common.h"
#include "cuda.h"
#endif

// Output of a clause that needs no matching: 0 with a contradiction, 1 when empty, -1 when it must be matched.
__device__ inline int eval_trivial(ull clause, const int8_t* __restrict__ has_contra,
                                   const NLITS_T* __restrict__ clause_len) {
    if (has_contra[clause])
        return 0;
    if (clause_len[clause] == 0)
        return 1;
    return -1;
}

// Warp parallel feature matching. Called by every warp, returns true if all the features checked by the warp matches,
// else false.
__device__ inline bool match_features_warp(const warp_t& warp, const FBOUND_T* __restrict__ Xe,
                                           const FBOUND_T* __restrict__ feat_bounds,
                                           const NFEAT_T* __restrict__ feat_ids, int n_feats) {
    const int lane = (int)warp.thread_rank();
    bool ok = true;
    for (int base = 0; base < n_feats; base += WARP_SIZE) {
        int i = base + lane;
        if (i < n_feats) {
            if (!feat_in_bounds(get_feature_value(Xe, 0, 0, (int)feat_ids[i]), feat_bounds, i))
                ok = false;
        }
        if (warp.any(!ok))
            return false;
    }
    return true;
}

// Warp parallel patch matching. Returns 32 bits (warp ballot), where each bit is 1 if the patch mached.
__device__ inline uint match_patches_warp(const warp_t& warp, const FBOUND_T* __restrict__ Xe,
                                          const PBOUND_T* __restrict__ pos, const FBOUND_T* __restrict__ feat_bounds,
                                          const NFEAT_T* __restrict__ feat_ids, int n_feats, int base) {
    const int n_x = pos[3] - pos[2] + 1;
    const int n_window = (pos[1] - pos[0] + 1) * n_x;
    const int i = base + (int)warp.thread_rank();
    bool match = false;
    if (i < n_window)
        match = match_patch(Xe, pos[0] + i / n_x, pos[2] + i % n_x, feat_ids, feat_bounds, n_feats);
    return warp.ballot(match);
}

/*
 * Output of every clause on every patch, for N samples.
 *
 * One thread per (sample, clause, patch).
 *
 * Arguments:
 *   X                       features of the batch
 *   N                       number of samples in the batch
 *   clause_position_bounds  per clause [y_lo, y_hi, x_lo, x_hi] of allowed patches
 *   clause_feat_bounds      per clause [lo, hi] of each bounded feature
 *   clause_feat_ids         per clause ids of its bounded features
 *   clause_n_feats          number of bounded features per clause
 *   has_contra              1 for clauses with a contradiction
 *   clause_len              number of included literals per clause
 *   patch_output            output, (N, TOTAL_CLAUSES, N_PATCHES) clause outputs per patch
 */
extern "C" __global__ void
eval_patchwise(const FBOUND_T* __restrict__ X, const int N, const PBOUND_T* __restrict__ clause_position_bounds,
               const FBOUND_T* __restrict__ clause_feat_bounds, const NFEAT_T* __restrict__ clause_feat_ids,
               const NFEAT_T* __restrict__ clause_n_feats, const int8_t* __restrict__ has_contra,
               const NLITS_T* __restrict__ clause_len, int8_t* __restrict__ patch_output) {
    GRID_STRIDE_LOOP(idx, (ull)N * TOTAL_CLAUSES * N_PATCHES) {
        ull e = idx / ((ull)TOTAL_CLAUSES * N_PATCHES);
        ull clause_patch = idx % ((ull)TOTAL_CLAUSES * N_PATCHES);
        ull clause = clause_patch / (ull)N_PATCHES;
        int patch = clause_patch % N_PATCHES;

        int8_t* output = &patch_output[patch_output_offset(e, clause, patch)];

        // Trivial checks
        int trivial = eval_trivial(clause, has_contra, clause_len);
        if (trivial >= 0) {
            *output = (int8_t)trivial;
            continue;
        }

        int py = patch / N_PATCHES_X;
        int px = patch % N_PATCHES_X;

        if (N_PATCHES > 1) {
            const PBOUND_T* pos = &clause_position_bounds[pos_bounds_offset(clause, 0)];
            // If position bounds contradict, then it cant match any position.
            if (py < pos[0] || py > pos[1] || px < pos[2] || px > pos[3]) {
                *output = 0;
                continue;
            }
        }

        const FBOUND_T* Xe = &X[sample_offset(e)];
        const FBOUND_T* cfb = &clause_feat_bounds[feat_bounds_offset(clause, 0, 0)];
        const NFEAT_T* bounded_fids = &clause_feat_ids[feat_ids_offset(clause, 0)];
        int n_bounded_fids = (int)clause_n_feats[clause];

        // match a patch
        *output = match_patch(Xe, py, px, bounded_fids, cfb, n_bounded_fids);
    }
}

/*
 * Clause outputs of N samples at inference, one warp per (sample, clause).
 *
 * Arguments:
 *   X                       features of the batch
 *   N                       number of samples in the batch
 *   clause_position_bounds  per clause [y_lo, y_hi, x_lo, x_hi] of allowed patches
 *   clause_feat_bounds      per clause [lo, hi] of each bounded feature
 *   clause_feat_ids         per clause ids of its bounded features
 *   clause_n_feats          number of bounded features per clause
 *   has_contra              1 for clauses with a contradiction
 *   clause_len              number of included literals per clause
 *   clause_outputs          output, (N, TOTAL_CLAUSES) clause outputs
 */
extern "C" __global__ void eval_infer(const FBOUND_T* __restrict__ X, const int N,
                                      const PBOUND_T* __restrict__ clause_position_bounds,
                                      const FBOUND_T* __restrict__ clause_feat_bounds,
                                      const NFEAT_T* __restrict__ clause_feat_ids,
                                      const NFEAT_T* __restrict__ clause_n_feats, const int8_t* __restrict__ has_contra,
                                      const NLITS_T* __restrict__ clause_len, int8_t* __restrict__ clause_outputs) {
    auto [warp, lane, warp_id, total_warps] = warp_grid();

    WARP_STRIDE_LOOP(idx, (ull)N * TOTAL_CLAUSES) {
        ull e = idx / (ull)TOTAL_CLAUSES;
        ull clause = idx % (ull)TOTAL_CLAUSES;

        // Trivial checks
        int out = eval_trivial(clause, has_contra, clause_len);
        if (out < 0) {
            const FBOUND_T* Xe = &X[sample_offset(e)];
            const FBOUND_T* feat_bounds = &clause_feat_bounds[feat_bounds_offset(clause, 0, 0)];
            const NFEAT_T* feat_ids = &clause_feat_ids[feat_ids_offset(clause, 0)];
            const int n_feats = (int)clause_n_feats[clause];

            if (N_PATCHES > 1) {
                // Conv
                const PBOUND_T* pos = &clause_position_bounds[pos_bounds_offset(clause, 0)];
                const int n_possible_patches = (pos[1] - pos[0] + 1) * (pos[3] - pos[2] + 1);
                out = 0;
                for (int base = 0; base < n_possible_patches && !out; base += WARP_SIZE) {
                    int warp_out = match_patches_warp(warp, Xe, pos, feat_bounds, feat_ids, n_feats, base);
                    out = warp_out != 0; // out = 1 if any bit of warp_out is set.
                }
            } else {
                // Non conv
                out = match_features_warp(warp, Xe, feat_bounds, feat_ids, n_feats);
            }
        }

        if (lane == 0)
            clause_outputs[idx] = (int8_t)out;
    }
}

/*
 * Clause outputs of the current sample during fit, one warp per clause. Use reservior sampling to select one matching
 * patch.
 *
 * Arguments:
 *   seed                    RNG key of the current sample
 *   X                       features of the batch
 *   e                       index of the current sample
 *   clause_drop_mask        1 for clauses dropped this epoch
 *   clause_position_bounds  per clause [y_lo, y_hi, x_lo, x_hi] of allowed patches
 *   clause_feat_bounds      per clause [lo, hi] of each bounded feature
 *   clause_feat_ids         per clause ids of its bounded features
 *   clause_n_feats          number of bounded features per clause
 *   has_contra              1 for clauses with a contradiction
 *   clause_len              number of included literals per clause
 *   clause_output           output, clause outputs
 *   selected_patch_ids      output, picked patch per firing clause (patches only)
 *   patch_weights           output, per clause and patch pick counts when TRACK_PATCH_WEIGHTS
 */
extern "C" __global__ void eval_fit(const ull seed, const FBOUND_T* __restrict__ X, const int e,
                                    const int8_t* __restrict__ clause_drop_mask,
                                    const PBOUND_T* __restrict__ clause_position_bounds,
                                    const FBOUND_T* __restrict__ clause_feat_bounds,
                                    const NFEAT_T* __restrict__ clause_feat_ids,
                                    const NFEAT_T* __restrict__ clause_n_feats, const int8_t* __restrict__ has_contra,
                                    const NLITS_T* __restrict__ clause_len, int8_t* __restrict__ clause_output,
                                    NPATCHES_T* __restrict__ selected_patch_ids, int* __restrict__ patch_weights) {
    const FBOUND_T* Xe = &X[sample_offset(e)];
    auto [warp, lane, warp_id, total_warps] = warp_grid();

    WARP_STRIDE_LOOP(clause, (ull)TOTAL_CLAUSES) {
        // trivial checks
        int out = (clause_drop_mask[clause] == 1) ? 0 : eval_trivial(clause, has_contra, clause_len);
        int selected_id = -1;

        const FBOUND_T* feat_bounds = &clause_feat_bounds[feat_bounds_offset(clause, 0, 0)];
        const NFEAT_T* feat_ids = &clause_feat_ids[feat_ids_offset(clause, 0)];
        const int n_feats = (int)clause_n_feats[clause];

        if (N_PATCHES > 1) {
            ull rng_k = rng_hash(seed, clause, 0xDEADBEEFULL);
            uint rng_counter = 0;

            if (out == 1) {
                // Empty clause: any patch.
                selected_id = (int)rand_int(rng_k, &rng_counter, N_PATCHES);
            } else if (out < 0) {
                const PBOUND_T* pos = &clause_position_bounds[pos_bounds_offset(clause, 0)];
                const int n_x = pos[3] - pos[2] + 1;
                const int n_window = (pos[1] - pos[0] + 1) * n_x;

                // Reservoir sampling, lane 0 keeps the pick.
                int count = 0;
                for (int base = 0; base < n_window; base += WARP_SIZE) {
                    uint ballot = match_patches_warp(warp, Xe, pos, feat_bounds, feat_ids, n_feats, base);
                    int n_matches = __popc(ballot);
                    count += n_matches;
                    if (lane == 0 && n_matches > 0 && (int)rand_int(rng_k, &rng_counter, (uint)count) < n_matches) {
                        // k-th set bit of the ballot.
                        int k = (int)rand_int(rng_k, &rng_counter, (uint)n_matches);
                        uint bits = ballot;
                        for (int j = 0; j < k; j++)
                            bits &= bits - 1;
                        int i = base + (__ffs(bits) - 1);
                        selected_id = (pos[0] + i / n_x) * N_PATCHES_X + pos[2] + i % n_x;
                    }
                }
                out = (selected_id >= 0); // valid on lane 0 only
            }
        } else {
            if (out < 0)
                out = match_features_warp(warp, Xe, feat_bounds, feat_ids, n_feats);
        }

        if (lane == 0) {
            clause_output[clause] = (int8_t)out;
            if (N_PATCHES > 1 && out == 1) {
                selected_patch_ids[clause] = (NPATCHES_T)selected_id;
                if (TRACK_PATCH_WEIGHTS)
                    patch_weights[patch_weights_offset(clause, selected_id)]++;
            }
        }
    }
}

/*
 * Class sums over chunks of clauses: chunk_sums[e, class, chunk] = weighted sum of firing clauses in the chunk.
 *
 * Arguments:
 *   clause_outputs  (N, TOTAL_CLAUSES) clause outputs
 *   clause_weights  clause weights per class
 *   N               number of samples
 *   chunks          number of chunks per class
 *   chunk_sums      output, (N, CLASSES, chunks)
 */
extern "C" __global__ void vote_over_clause_chunks(const int8_t* __restrict__ clause_outputs,
                                                   const float* __restrict__ clause_weights, const int N,
                                                   const int chunks, float* __restrict__ chunk_sums) {
    auto [warp, lane, warp_id, total_warps] = warp_grid();
    const int chunk_len = (CLAUSES_PER_CLASS + chunks - 1) / chunks;

    WARP_STRIDE_LOOP(w, (ull)N * CLASSES * chunks) {
        ull e = w / ((ull)CLASSES * chunks);
        ull class_id = (w / chunks) % CLASSES;
        ull chunk = w % chunks;
        int clause_start = (int)chunk * chunk_len;
        int clause_end = min(clause_start + chunk_len, (int)CLAUSES_PER_CLASS);

        const ull first_clause = COALESCED ? 0 : class_id * (ull)CLAUSES_PER_CLASS; // first global clause of the class
        const int8_t* outputs = &clause_outputs[clause_output_offset(e, first_clause)];
        const float* weights = &clause_weights[weight_offset(class_id, first_clause)];

        float s = 0.0f;
        for (int clause_id = clause_start + lane; clause_id < clause_end; clause_id += WARP_SIZE)
            if (outputs[clause_id])
                s += weights[clause_id];
        s = cg::reduce(warp, s, cg::plus<float>());
        if (lane == 0)
            chunk_sums[chunk_sums_offset(e, class_id, chunk, chunks)] = s;
    }
}

/*
 * Class sums from the chunk sums, added in chunk order so the result is deterministic.
 *
 * Arguments:
 *   chunk_sums  (N, CLASSES, chunks), from vote_over_clause_chunks
 *   N           number of samples
 *   chunks      number of chunks per class
 *   class_sums  output, (N, CLASSES)
 */
extern "C" __global__ void vote_combine_chunks(const float* __restrict__ chunk_sums, const int N, const int chunks,
                                               float* __restrict__ class_sums) {
    GRID_STRIDE_LOOP(idx, (ull)N * CLASSES) {
        ull e = idx / CLASSES, class_id = idx % CLASSES;
        float s = 0.0f;
        for (int chunk = 0; chunk < chunks; chunk++)
            s += chunk_sums[chunk_sums_offset(e, class_id, chunk, chunks)];
        class_sums[idx] = s;
    }
}
