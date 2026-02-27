#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <string.h>

#define EXPORT __attribute__((visibility("default")))

#define CLAUSES_PER_CLASS (CLAUSES / CLAUSE_BANKS)
#if ((LITERALS / 2) & 1)
    #define VECTORIZED_LIMIT 0
#else
    #define VECTORIZED_LIMIT (LITERALS & ~3)
#endif
#define S_INV (1.0f / S)
#define S_NEG_POLARITY_INV (1.0f / S_NEG_POLARITY)
#define Q_PROB (1.0f * Q / max(1, CLASSES - 1))
#define INT_SIZE 32
#define NUM_LITERAL_CHUNKS (((LITERALS - 1) / INT_SIZE) + 1)
#if ((LITERALS % INT_SIZE) != 0)
    #define FILTER (~(0xFFFFFFFF << (LITERALS % INT_SIZE)))
#else
    #define FILTER 0xFFFFFFFF
#endif

#if COALESCED == 0
    #define LOOP_CLASS_ID(class_id, clause) class_id = (ull)clause / CLAUSES_PER_CLASS;
#else
    #define LOOP_CLASS_ID(class_id, clause) for (class_id = 0; class_id < CLASSES; ++class_id)
#endif

#define CLIP(val, min, max) ((val < min) ? min : ((val > max) ? max : val))

typedef unsigned long long ull;

static double H[CLASSES];

EXPORT void set_H(double* h_vals) {
    for (int i = 0; i < CLASSES; ++i) {
        H[i] = h_vals[i];
    }
}

EXPORT void set_num_threads(int n) { omp_set_num_threads(n); }

static float rand_uniform(uint64_t* rng_state) {
    *rng_state = *rng_state * 1103515245 + 12345;
    return (float)(((*rng_state) >> 16) & 0x7FFF) / 32767.0f;
}

/***********INPUT ENCODING***********/
EXPORT void encode_batch(const int* X, unsigned int* encoded_X, const int N) {
#pragma omp parallel for schedule(guided)
    for (ull e_patch = 0; e_patch < (ull)PATCHES * N; ++e_patch) {
        ull e = e_patch / PATCHES;
        ull patch_id = e_patch % PATCHES;

        int patch_coordinate_y = patch_id / (DIM0 - PATCH_DIM0 + 1);
        int patch_coordinate_x = patch_id % (DIM0 - PATCH_DIM0 + 1);

        ull encX_offset = e * (ull)(PATCHES * NUM_LITERAL_CHUNKS) + patch_id * (ull)NUM_LITERAL_CHUNKS;
        unsigned int* patch_output = &encoded_X[encX_offset];

        memset(patch_output, 0, NUM_LITERAL_CHUNKS * sizeof(unsigned int));

#if APPEND_NEGATED
        for (int literal = LITERALS / 2; literal < LITERALS; ++literal) {
            int chunk_nr = literal / INT_SIZE;
            int chunk_pos = literal % INT_SIZE;
            patch_output[chunk_nr] |= (1u << chunk_pos);
        }
#endif

        for (int lit = 0; lit < patch_coordinate_y; ++lit) {
            int chunk_nr = lit / INT_SIZE;
            int chunk_pos = lit % INT_SIZE;
            patch_output[chunk_nr] |= (1u << chunk_pos);
#if APPEND_NEGATED
            int neg_chunk_nr = (lit + (LITERALS / 2)) / INT_SIZE;
            int neg_chunk_pos = (lit + (LITERALS / 2)) % INT_SIZE;
            patch_output[neg_chunk_nr] &= ~(1u << neg_chunk_pos);
#endif
        }

        for (int lit = 0; lit < patch_coordinate_x; ++lit) {
            int chunk_nr = (DIM1 - PATCH_DIM1 + lit) / INT_SIZE;
            int chunk_pos = (DIM1 - PATCH_DIM1 + lit) % INT_SIZE;
            patch_output[chunk_nr] |= (1u << chunk_pos);
#if APPEND_NEGATED
            int neg_chunk_nr = ((DIM1 - PATCH_DIM1 + lit) + (LITERALS / 2)) / INT_SIZE;
            int neg_chunk_pos = ((DIM1 - PATCH_DIM1 + lit) + (LITERALS / 2)) % INT_SIZE;
            patch_output[neg_chunk_nr] &= ~(1u << neg_chunk_pos);
#endif
        }

        for (ull p_y = patch_coordinate_y; p_y < patch_coordinate_y + PATCH_DIM1; ++p_y) {
            for (ull p_x = patch_coordinate_x; p_x < patch_coordinate_x + PATCH_DIM0; ++p_x) {
                for (int z = 0; z < DIM2; ++z) {
                    unsigned long long dense_idx =
                        e * (ull)(DIM0 * DIM1 * DIM2) + p_y * (ull)(DIM0 * DIM2) + p_x * (ull)DIM2 + z;

                    int rel_y = p_y - patch_coordinate_y;
                    int rel_x = p_x - patch_coordinate_x;
#if ENCODE_LOC
                    int patch_pos =
                        (DIM1 - PATCH_DIM1) + (DIM0 - PATCH_DIM0) + rel_y * PATCH_DIM0 * DIM2 + rel_x * DIM2 + z;
#else
                    int patch_pos = rel_y * PATCH_DIM0 * DIM2 + rel_x * DIM2 + z;
#endif
                    if (X[dense_idx] == 1) {
                        int chunk_nr = patch_pos / INT_SIZE;
                        int chunk_pos = patch_pos % INT_SIZE;
                        patch_output[chunk_nr] |= (1u << chunk_pos);
#if APPEND_NEGATED
                        int neg_chunk_nr = (patch_pos + (LITERALS / 2)) / INT_SIZE;
                        int neg_chunk_pos = (patch_pos + (LITERALS / 2)) % INT_SIZE;
                        patch_output[neg_chunk_nr] &= ~(1u << neg_chunk_pos);
#endif
                    } else if (X[dense_idx] == 0) {
#if APPEND_NEGATED
                        int neg_chunk_nr = (patch_pos + (LITERALS / 2)) / INT_SIZE;
                        int neg_chunk_pos = (patch_pos + (LITERALS / 2)) % INT_SIZE;
                        patch_output[neg_chunk_nr] &= ~(1u << neg_chunk_pos);
#endif
                    }
                }
            }
        }
    }
}

/***********CLAUSE PACKING***********/
EXPORT void pack_clauses(const unsigned int* global_ta_states, unsigned int* packed_clauses, int* num_includes) {
#pragma omp parallel for schedule(guided)
    for (ull clause = 0; clause < CLAUSES; ++clause) {
        const unsigned int* ta_state = &global_ta_states[clause * LITERALS];
        unsigned int* packed_clause = &packed_clauses[clause * NUM_LITERAL_CHUNKS];
        int total_count = 0;
        memset(packed_clause, 0, NUM_LITERAL_CHUNKS * sizeof(unsigned int));

        for (int li = 0; li < VECTORIZED_LIMIT; li += 4) {
            unsigned int x0 = ta_state[li];
            unsigned int x1 = ta_state[li + 1];
            unsigned int x2 = ta_state[li + 2];
            unsigned int x3 = ta_state[li + 3];

            if (x0 >= INCLUDE_TA_STATE) {
                packed_clause[li / INT_SIZE] |= (1u << (li % INT_SIZE));
                total_count++;
            }
            if (x1 >= INCLUDE_TA_STATE) {
                packed_clause[(li + 1) / INT_SIZE] |= (1u << ((li + 1) % INT_SIZE));
                total_count++;
            }
            if (x2 >= INCLUDE_TA_STATE) {
                packed_clause[(li + 2) / INT_SIZE] |= (1u << ((li + 2) % INT_SIZE));
                total_count++;
            }
            if (x3 >= INCLUDE_TA_STATE) {
                packed_clause[(li + 3) / INT_SIZE] |= (1u << ((li + 3) % INT_SIZE));
                total_count++;
            }
        }
        for (int li = VECTORIZED_LIMIT; li < LITERALS; ++li) {
            if (ta_state[li] >= INCLUDE_TA_STATE) {
                packed_clause[li / INT_SIZE] |= (1u << (li % INT_SIZE));
                total_count++;
            }
        }
        num_includes[clause] = total_count;
    }
}

/***********CLAUSE EVALUATION***********/
static inline int clause_match(const unsigned int* ta_state, const unsigned int* X, const unsigned int* literal_mask) {
    for (int chunk = 0; chunk < NUM_LITERAL_CHUNKS - 1; ++chunk)
        if ((ta_state[chunk] & (X[chunk] & literal_mask[chunk])) != (ta_state[chunk] & literal_mask[chunk])) return 0;
    if ((ta_state[NUM_LITERAL_CHUNKS - 1] &
         (X[NUM_LITERAL_CHUNKS - 1] & literal_mask[NUM_LITERAL_CHUNKS - 1] & FILTER)) !=
        (ta_state[NUM_LITERAL_CHUNKS - 1] & literal_mask[NUM_LITERAL_CHUNKS - 1] & FILTER))
        return 0;

    return 1;
}

/***********FAST EVALUATION***********/
EXPORT void fast_eval(const unsigned int* packed_ta_states, const int* num_includes,
                      const unsigned int* clause_drop_mask, const unsigned int* literal_mask,
                      const unsigned int* X_batch, unsigned int* clause_outputs, const int e) {
#pragma omp parallel for schedule(guided)
    for (ull clause_patch = 0; clause_patch < (ull)CLAUSES * (ull)PATCHES; ++clause_patch) {
        unsigned int* clause_output = &clause_outputs[clause_patch];

        ull clause = clause_patch / PATCHES;
        ull patch_id = clause_patch % PATCHES;

        if (clause_drop_mask[clause] == 1) {
            *clause_output = 0;
            continue;
        }

        if (num_includes[clause] == 0) {
            *clause_output = 1;
            continue;
        }

        *clause_output = clause_match(
            &packed_ta_states[clause * NUM_LITERAL_CHUNKS],
            &X_batch[(ull)e * (ull)(PATCHES * NUM_LITERAL_CHUNKS) + patch_id * NUM_LITERAL_CHUNKS], literal_mask);
    }
}

/***********SELECT ACTIVE CLAUSES AND CALCULATE CLASS SUMS***********/
EXPORT void select_active(uint64_t* rng_states, const float* clause_weights, const unsigned int* clause_outputs,
                          int* patch_weights, int* selected_patch_ids, float* positive_evidence,
                          float* negative_evidence) {
    memset(positive_evidence, 0, CLASSES * sizeof(float));
    memset(negative_evidence, 0, CLASSES * sizeof(float));

#pragma omp parallel
    {
        float local_pos[CLASSES] = {0};
        float local_neg[CLASSES] = {0};

#pragma omp for schedule(guided)
        for (ull clause = 0; clause < CLAUSES; ++clause) {
            int count = 0;
            int selected_id = -1;
            for (int patch_id = 0; patch_id < PATCHES; ++patch_id) {
                if (clause_outputs[clause * PATCHES + patch_id]) {
                    count++;
                    if (rand_uniform(&rng_states[omp_get_thread_num()]) < 1.0f / count) {
                        selected_id = patch_id;
                    }
                }
            }
            selected_patch_ids[clause] = selected_id;
            if (selected_id != -1) {
#pragma omp atomic
                patch_weights[clause * PATCHES + selected_id]++;
                ull class_id, rel_clause = clause % CLAUSES_PER_CLASS;
                LOOP_CLASS_ID(class_id, clause) {
                    float w = clause_weights[class_id * CLAUSES_PER_CLASS + rel_clause];
                    if (w >= 0)
                        local_pos[class_id] += w;
                    else
                        local_neg[class_id] += w;
                }
            }
        }

#pragma omp critical
        {
            for (int c = 0; c < CLASSES; ++c) {
                positive_evidence[c] += local_pos[c];
                negative_evidence[c] += local_neg[c];
            }
        }
    }
}

/***********FAST CLASS SUMS CALCULATION FOR INFERENCE***********/
EXPORT void calc_class_sums_infer_batch(const unsigned int* packed_ta_states, const float* clause_weights,
                                        const int* num_includes, const unsigned int* X_batch, const int N,
                                        float* class_sums_batch, const unsigned int* literal_mask) {
    memset(class_sums_batch, 0, N * CLASSES * sizeof(float));

#pragma omp parallel
    {
        float local_sums[N * CLASSES];
        memset(local_sums, 0, N * CLASSES * sizeof(float));

#pragma omp for schedule(guided)
        for (ull e_clause = 0; e_clause < (ull)N * (ull)CLAUSES; ++e_clause) {
            ull e = e_clause / CLAUSES;
            ull clause = e_clause % CLAUSES;
            if (num_includes[clause] == 0) continue;
            int clause_output = 0;
            for (int patch_id = 0; patch_id < PATCHES; ++patch_id) {
                if (clause_match(&packed_ta_states[clause * NUM_LITERAL_CHUNKS],
                                 &X_batch[e * (ull)(PATCHES * NUM_LITERAL_CHUNKS) + patch_id * NUM_LITERAL_CHUNKS],
                                 literal_mask)) {
                    clause_output = 1;
                    break;
                }
            }
            if (clause_output) {
                ull class_id, rel_clause = clause % CLAUSES_PER_CLASS;
                LOOP_CLASS_ID(class_id, clause) {
                    local_sums[e * CLASSES + class_id] += clause_weights[class_id * CLAUSES_PER_CLASS + rel_clause];
                }
            }
        }

#pragma omp critical
        {
            for (int e = 0; e < N; ++e) {
                for (int c = 0; c < CLASSES; ++c) {
                    class_sums_batch[e * CLASSES + c] += local_sums[e * CLASSES + c];
                }
            }
        }
    }
}

/***********TRANSFORM KERNELS***********/
EXPORT void transform(const unsigned int* packed_ta_states, const int* num_includes, const unsigned int* X_batch,
                      const int N, unsigned int* clause_outputs, const unsigned int* literal_mask) {
#pragma omp parallel for schedule(guided)
    for (ull e_clause = 0; e_clause < (ull)N * (ull)CLAUSES; ++e_clause) {
        ull e = e_clause / CLAUSES;
        ull clause = e_clause % CLAUSES;
        if (num_includes[clause] == 0) {
            clause_outputs[e * CLAUSES + clause] = 1;
            continue;
        }
        int clause_output = 0;
        for (int patch_id = 0; patch_id < PATCHES; ++patch_id) {
            if (clause_match(&packed_ta_states[clause * NUM_LITERAL_CHUNKS],
                             &X_batch[e * (ull)(PATCHES * NUM_LITERAL_CHUNKS) + patch_id * NUM_LITERAL_CHUNKS],
                             literal_mask)) {
                clause_output = 1;
                break;
            }
        }
        clause_outputs[e * CLAUSES + clause] = clause_output;
    }
}

EXPORT void transform_patchwise(const unsigned int* packed_ta_states, const int* num_includes,
                                const unsigned int* X_batch, const int N, unsigned int* clause_outputs,
                                const unsigned int* literal_mask) {
#pragma omp parallel for schedule(guided)
    for (ull e_clause_patch = 0; e_clause_patch < (ull)N * (ull)CLAUSES * (ull)PATCHES; ++e_clause_patch) {
        unsigned int* clause_output = &clause_outputs[e_clause_patch];

        ull e_clause = e_clause_patch / PATCHES;
        ull patch_id = e_clause_patch % PATCHES;

        ull e = e_clause / CLAUSES;
        ull clause = e_clause % CLAUSES;

        if (num_includes[clause] == 0) {
            *clause_output = 1;
            continue;
        }

        *clause_output = clause_match(&packed_ta_states[clause * NUM_LITERAL_CHUNKS],
                                      &X_batch[e * (ull)(PATCHES * NUM_LITERAL_CHUNKS) + patch_id * NUM_LITERAL_CHUNKS],
                                      literal_mask);
    }
}

/***********CLAUSE UPDATE HELPERS***********/
static inline void type1a_fb(uint64_t* rng_state, unsigned int* ta_state, const unsigned int* patch, const int sign,
                             const unsigned int* literal_mask) {
    float s_inv = (sign == 1) ? S_INV : S_NEG_POLARITY_INV;
    for (int li = 0; li < VECTORIZED_LIMIT; li += 4) {
        unsigned int ta_x = ta_state[li];
        unsigned int ta_y = ta_state[li + 1];
        unsigned int ta_z = ta_state[li + 2];
        unsigned int ta_w = ta_state[li + 3];

        unsigned int patch_x = (patch[li / INT_SIZE] >> (li % INT_SIZE)) & 1u;
        unsigned int patch_y = (patch[(li + 1) / INT_SIZE] >> ((li + 1) % INT_SIZE)) & 1u;
        unsigned int patch_z = (patch[(li + 2) / INT_SIZE] >> ((li + 2) % INT_SIZE)) & 1u;
        unsigned int patch_w = (patch[(li + 3) / INT_SIZE] >> ((li + 3) % INT_SIZE)) & 1u;

        unsigned int lit_up_x = (literal_mask[li / INT_SIZE] >> (li % INT_SIZE)) & 1u;
        unsigned int lit_up_y = (literal_mask[(li + 1) / INT_SIZE] >> ((li + 1) % INT_SIZE)) & 1u;
        unsigned int lit_up_z = (literal_mask[(li + 2) / INT_SIZE] >> ((li + 2) % INT_SIZE)) & 1u;
        unsigned int lit_up_w = (literal_mask[(li + 3) / INT_SIZE] >> ((li + 3) % INT_SIZE)) & 1u;

        ta_x += (lit_up_x == 1 && patch_x == 1 && ta_x < MAX_TA_STATE);
        ta_y += (lit_up_y == 1 && patch_y == 1 && ta_y < MAX_TA_STATE);
        ta_z += (lit_up_z == 1 && patch_z == 1 && ta_z < MAX_TA_STATE);
        ta_w += (lit_up_w == 1 && patch_w == 1 && ta_w < MAX_TA_STATE);

        ta_x -= (lit_up_x == 1 && patch_x == 0 && ta_x > 0 && rand_uniform(rng_state) <= s_inv);
        ta_y -= (lit_up_y == 1 && patch_y == 0 && ta_y > 0 && rand_uniform(rng_state) <= s_inv);
        ta_z -= (lit_up_z == 1 && patch_z == 0 && ta_z > 0 && rand_uniform(rng_state) <= s_inv);
        ta_w -= (lit_up_w == 1 && patch_w == 0 && ta_w > 0 && rand_uniform(rng_state) <= s_inv);

        ta_state[li] = ta_x;
        ta_state[li + 1] = ta_y;
        ta_state[li + 2] = ta_z;
        ta_state[li + 3] = ta_w;
    }

    for (int li = VECTORIZED_LIMIT; li < LITERALS; ++li) {
        unsigned int patch_bit = (patch[li / INT_SIZE] >> (li % INT_SIZE)) & 1u;
        unsigned int lit_up = (literal_mask[li / INT_SIZE] >> (li % INT_SIZE)) & 1u;
        if (lit_up == 1 && patch_bit == 1 && ta_state[li] < MAX_TA_STATE) {
            ta_state[li] += 1;
        } else if (lit_up == 1 && patch_bit == 0 && ta_state[li] > 0 && rand_uniform(rng_state) <= s_inv) {
            ta_state[li] -= 1;
        }
    }
}

static inline void type1b_fb(uint64_t* rng_state, unsigned int* ta_state, const int sign,
                             const unsigned int* literal_mask) {
    float s_inv = (sign == 1) ? S_INV : S_NEG_POLARITY_INV;
    for (int li = 0; li < VECTORIZED_LIMIT; li += 4) {
        unsigned int ta_x = ta_state[li];
        unsigned int ta_y = ta_state[li + 1];
        unsigned int ta_z = ta_state[li + 2];
        unsigned int ta_w = ta_state[li + 3];

        unsigned int lit_up_x = (literal_mask[li / INT_SIZE] >> (li % INT_SIZE)) & 1u;
        unsigned int lit_up_y = (literal_mask[(li + 1) / INT_SIZE] >> ((li + 1) % INT_SIZE)) & 1u;
        unsigned int lit_up_z = (literal_mask[(li + 2) / INT_SIZE] >> ((li + 2) % INT_SIZE)) & 1u;
        unsigned int lit_up_w = (literal_mask[(li + 3) / INT_SIZE] >> ((li + 3) % INT_SIZE)) & 1u;

        ta_x -= (lit_up_x == 1 && ta_x > 0 && rand_uniform(rng_state) <= s_inv);
        ta_y -= (lit_up_y == 1 && ta_y > 0 && rand_uniform(rng_state) <= s_inv);
        ta_z -= (lit_up_z == 1 && ta_z > 0 && rand_uniform(rng_state) <= s_inv);
        ta_w -= (lit_up_w == 1 && ta_w > 0 && rand_uniform(rng_state) <= s_inv);

        ta_state[li] = ta_x;
        ta_state[li + 1] = ta_y;
        ta_state[li + 2] = ta_z;
        ta_state[li + 3] = ta_w;
    }

    for (int li = VECTORIZED_LIMIT; li < LITERALS; ++li) {
        unsigned int lit_up = (literal_mask[li / INT_SIZE] >> (li % INT_SIZE)) & 1u;
        if (lit_up == 1 && ta_state[li] > 0 && rand_uniform(rng_state) <= s_inv) {
            ta_state[li] -= 1;
        }
    }
}

static inline void type2_fb(unsigned int* ta_state, const unsigned int* patch, const unsigned int* literal_mask) {
    for (int li = 0; li < VECTORIZED_LIMIT; li += 4) {
        unsigned int ta_x = ta_state[li];
        unsigned int ta_y = ta_state[li + 1];
        unsigned int ta_z = ta_state[li + 2];
        unsigned int ta_w = ta_state[li + 3];

        unsigned int patch_x = (patch[li / INT_SIZE] >> (li % INT_SIZE)) & 1u;
        unsigned int patch_y = (patch[(li + 1) / INT_SIZE] >> ((li + 1) % INT_SIZE)) & 1u;
        unsigned int patch_z = (patch[(li + 2) / INT_SIZE] >> ((li + 2) % INT_SIZE)) & 1u;
        unsigned int patch_w = (patch[(li + 3) / INT_SIZE] >> ((li + 3) % INT_SIZE)) & 1u;

        unsigned int lit_up_x = (literal_mask[li / INT_SIZE] >> (li % INT_SIZE)) & 1u;
        unsigned int lit_up_y = (literal_mask[(li + 1) / INT_SIZE] >> ((li + 1) % INT_SIZE)) & 1u;
        unsigned int lit_up_z = (literal_mask[(li + 2) / INT_SIZE] >> ((li + 2) % INT_SIZE)) & 1u;
        unsigned int lit_up_w = (literal_mask[(li + 3) / INT_SIZE] >> ((li + 3) % INT_SIZE)) & 1u;

        ta_x += (lit_up_x == 1 && patch_x == 0 && ta_x < INCLUDE_TA_STATE);
        ta_y += (lit_up_y == 1 && patch_y == 0 && ta_y < INCLUDE_TA_STATE);
        ta_z += (lit_up_z == 1 && patch_z == 0 && ta_z < INCLUDE_TA_STATE);
        ta_w += (lit_up_w == 1 && patch_w == 0 && ta_w < INCLUDE_TA_STATE);

        ta_state[li] = ta_x;
        ta_state[li + 1] = ta_y;
        ta_state[li + 2] = ta_z;
        ta_state[li + 3] = ta_w;
    }

    for (int li = VECTORIZED_LIMIT; li < LITERALS; ++li) {
        unsigned int patch_bit = (patch[li / INT_SIZE] >> (li % INT_SIZE)) & 1u;
        unsigned int lit_up = (literal_mask[li / INT_SIZE] >> (li % INT_SIZE)) & 1u;
        if (lit_up == 1 && patch_bit == 0 && ta_state[li] < INCLUDE_TA_STATE) {
            ta_state[li] += 1;
        }
    }
}

static inline double uprob_fun(double v, double y, double h, double g) {
    int tcs = (2 * y * h) - y;
    double prob = (y - v) / (2 * y);

    if ((y > 0) == (v > tcs)) {
        prob = pow((1.0 - h), (1.0 - g)) * pow(prob, g);
    } else {
        prob = 1.0 - (pow(h, 1.0 - g) * pow((1.0 - prob), g));
    }

    return prob;
}

EXPORT void evidence_to_update_prob(const float* positive_evidence, const float* negative_evidence, const int* targets,
                                    const double* g_pos, const double* g_neg, const int e, double* pprob,
                                    double* nprob) {
#pragma omp parallel for schedule(guided)
    for (ull class_id = 0; class_id < CLASSES; ++class_id) {
        int local_target = targets[e * CLASSES + class_id];
        if (local_target == 0) {
            pprob[class_id] = 0.0;
            nprob[class_id] = 0.0;
            continue;
        }

#if SPLIT_CLASS_SUM == 1
        double pos_ev = (double)CLIP(positive_evidence[class_id], 0, THRESH);
        double neg_ev = (double)CLIP(negative_evidence[class_id], -THRESH, 0);
        if (local_target == 1) {
            pprob[class_id] = (THRESH - pos_ev) / THRESH;
            nprob[class_id] = (0.0 - neg_ev) / THRESH;
        } else if (local_target == -1) {
            pprob[class_id] = (0.0 - pos_ev) / -THRESH;
            nprob[class_id] = (-THRESH - neg_ev) / -THRESH;
        }
#else
        double y = (double)THRESH * (double)local_target;
        double g = (local_target == 1) ? g_pos[class_id] : g_neg[class_id];
        double h = (local_target == 1) ? H[class_id] : (1.0 - H[class_id]);
        if (h == 1.0) h = 0.999999;
        if (h == 0.0) h = 0.000001;
        double class_sum = (double)CLIP(positive_evidence[class_id] + negative_evidence[class_id], -THRESH, THRESH);
        pprob[class_id] = uprob_fun(class_sum, y, h, g);
        nprob[class_id] = pprob[class_id];
#endif
    }
}

EXPORT void clause_update(uint64_t* rng_states, unsigned int* global_ta_states, float* clause_weights,
                          const int* selected_patch_ids, const int* num_includes, const unsigned int* clause_drop_mask,
                          const unsigned int* literal_mask, const unsigned int* X_batch, const int* targets,
                          const double* pprob, const double* nprob, const int e, int num_threads) {
#pragma omp parallel for schedule(guided)
    for (ull clause = 0; clause < CLAUSES; ++clause) {
        int thread_id = omp_get_thread_num();

        if (clause_drop_mask[clause] == 1) continue;

        unsigned int* ta_state = &global_ta_states[clause * LITERALS];
        int local_clause_output = selected_patch_ids[clause] > -1 ? 1 : 0;
        const unsigned int* X = &X_batch[(ull)e * (ull)(PATCHES * NUM_LITERAL_CHUNKS)];
        const unsigned int* patch =
            selected_patch_ids[clause] > -1 ? &X[selected_patch_ids[clause] * NUM_LITERAL_CHUNKS] : NULL;

        ull class_id, rel_clause = clause % CLAUSES_PER_CLASS;
        LOOP_CLASS_ID(class_id, clause) {
            int local_target = targets[e * CLASSES + class_id];
            if (local_target == 0) continue;

            float* local_weight = &clause_weights[class_id * CLAUSES_PER_CLASS + rel_clause];
            int sign = (*local_weight >= 0) - (*local_weight < 0);

            double update_prob = (sign == 1) ? pprob[class_id] : nprob[class_id];
            int should_update = (rand_uniform(&rng_states[thread_id]) <= update_prob);
            int clause_has_space = (num_includes[clause] <= MAX_INCLUDED_LITERALS);
            int t1 = (local_target * sign) > 0;

#if TYPE1A_FB
            if (should_update && t1 && local_clause_output && clause_has_space) {
                type1a_fb(&rng_states[thread_id], ta_state, patch, sign, literal_mask);
    #if WEIGHTED
                if (fabsf(*local_weight) < MAX_WEIGHT) (*local_weight) += sign * 1.0f;
    #endif
            }
#endif

#if TYPE1B_FB
            if (should_update && t1 && !(local_clause_output && clause_has_space)) {
                type1b_fb(&rng_states[thread_id], ta_state, sign, literal_mask);
            }
#endif

#if TYPE2_FB
            if (should_update && (local_target * sign) < 0 && local_clause_output) {
                type2_fb(ta_state, patch, literal_mask);
    #if WEIGHTED
                if (fabsf(*local_weight) < MAX_WEIGHT) (*local_weight) -= sign * 1.0f;
        #if ALLOW_POLARITY_CHANGE == 0
                if (sign == 1 && *local_weight < 0) *local_weight = 1;
                if (sign == -1 && *local_weight >= 0) *local_weight = -1;
        #endif
    #endif
    #if NEGATIVE_CLAUSES == 0
                if (*local_weight < 1) *local_weight = 1;
    #endif
            }
#endif
        }
    }
}
