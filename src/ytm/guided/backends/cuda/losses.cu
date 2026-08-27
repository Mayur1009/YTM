#ifdef IS_NEOVIM_CLANGD_ENV
#include "common.cu"
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
#endif

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

#define NEG_INF -1e30f

__device__ inline float safe_pow(float base, float exp) {
    float b = fmaxf(base, 0.0f);
    return (exp == 0.0f) ? 1.0f : powf(b, exp);
}

__device__ inline void compute_act(warp_t warp, const float* votes, float* y_hat) {
    int lane = warp.thread_rank();

#if ACT_FN == ACT_SOFTMAX
    float max_v = NEG_INF;
    for (int c = lane; c < CLASSES; c += WARP_SIZE) {
        float v = votes[c] / (float)CLAUSES_PER_CLASS;
        if (v > max_v)
            max_v = v;
    }
    max_v = cg::reduce(warp, max_v, cg::greater<float>());

    float sum_exp = 0.0f;
    for (int c = lane; c < CLASSES; c += WARP_SIZE) {
        float e = expf(votes[c] / (float)CLAUSES_PER_CLASS - max_v);
        y_hat[c] = e;
        sum_exp += e;
    }
    sum_exp = cg::reduce(warp, sum_exp, cg::plus<float>());

    for (int c = lane; c < CLASSES; c += WARP_SIZE)
        y_hat[c] /= sum_exp;
#elif ACT_FN == ACT_SIGMOID
    for (int c = lane; c < CLASSES; c += WARP_SIZE)
        y_hat[c] = 1.0f / (1.0f + expf(-votes[c] / (float)CLAUSES_PER_CLASS));
#else
    for (int c = lane; c < CLASSES; c += WARP_SIZE)
        y_hat[c] = votes[c];
#endif
}

__device__ inline float dact_of(float act) {
#if ACT_FN == ACT_SIGMOID
    return act * (1.0f - act);
#else
    return 1.0f;
#endif
}

__device__ inline void compute_ce(warp_t warp, const float* y, const float* y_hat, const float* class_weights,
                                  bool compute_loss, float* grad, float* loss) {
    int lane = warp.thread_rank();
#if ACT_FN == ACT_SOFTMAX
    float partial = 0.0f;
    for (int c = lane; c < CLASSES; c += WARP_SIZE)
        partial += y[c] * y_hat[c];
    float dot_val = cg::reduce(warp, partial, cg::plus<float>());
    float fw = safe_pow(1.0f - dot_val, LOSS_GAMMA);

    float acc = 0.0f;
    for (int c = lane; c < CLASSES; c += WARP_SIZE) {
        if (compute_loss)
            acc += class_weights[c] * y[c] * logf(y_hat[c] + LOSS_EPS);
        grad[c] = class_weights[c] * fw * (y[c] - y_hat[c]);
    }
    if (compute_loss) {
        acc = cg::reduce(warp, acc, cg::plus<float>());
        if (lane == 0)
            *loss = -acc * fw;
    }
#else
    float acc = 0.0f;
    for (int c = lane; c < CLASSES; c += WARP_SIZE) {
        float p_t = y[c] * y_hat[c] + (1.0f - y[c]) * (1.0f - y_hat[c]);
        float fw = safe_pow(1.0f - p_t, LOSS_GAMMA);
        if (compute_loss)
            acc += class_weights[c] * fw *
                   (y[c] * logf(y_hat[c] + LOSS_EPS) + (1.0f - y[c]) * logf(1.0f - y_hat[c] + LOSS_EPS));
        grad[c] = class_weights[c] * fw * (y[c] - y_hat[c]);
    }
    if (compute_loss) {
        acc = cg::reduce(warp, acc, cg::plus<float>());
        if (lane == 0)
            *loss = -acc;
    }
#endif
}

__device__ inline void compute_sce(warp_t warp, const float* y, const float* y_hat, const float* class_weights,
                                   bool compute_loss, float* grad, float* loss) {
    int lane = warp.thread_rank();
#if ACT_FN == ACT_SOFTMAX
    float partial = 0.0f;
    for (int c = lane; c < CLASSES; c += WARP_SIZE)
        partial += y_hat[c] * logf(y[c] + LOSS_EPS);
    float dot_val = cg::reduce(warp, partial, cg::plus<float>());

    float term_a = 0.0f, term_b = 0.0f;
    for (int c = lane; c < CLASSES; c += WARP_SIZE) {
        float logy_c = logf(y[c] + LOSS_EPS);
        if (compute_loss) {
            term_a += class_weights[c] * y[c] * logf(y_hat[c] + LOSS_EPS);
            term_b += class_weights[c] * y_hat[c] * logy_c;
        }
        grad[c] = class_weights[c] * (LOSS_ALPHA * (y[c] - y_hat[c]) + LOSS_BETA * y_hat[c] * (logy_c - dot_val));
    }
    if (compute_loss) {
        term_a = cg::reduce(warp, term_a, cg::plus<float>());
        term_b = cg::reduce(warp, term_b, cg::plus<float>());
        if (lane == 0)
            *loss = -LOSS_ALPHA * term_a - LOSS_BETA * term_b;
    }
#else
    float term_a = 0.0f, term_b = 0.0f;
    for (int c = lane; c < CLASSES; c += WARP_SIZE) {
        if (compute_loss) {
            term_a += class_weights[c] *
                      (y[c] * logf(y_hat[c] + LOSS_EPS) + (1.0f - y[c]) * logf(1.0f - y_hat[c] + LOSS_EPS));
            term_b += class_weights[c] *
                      (y_hat[c] * logf(y[c] + LOSS_EPS) + (1.0f - y_hat[c]) * logf(1.0f - y[c] + LOSS_EPS));
        }
        float lograt = logf((y[c] + LOSS_EPS) / (1.0f - y[c] + LOSS_EPS));
        grad[c] = class_weights[c] * (LOSS_ALPHA * (y[c] - y_hat[c]) + LOSS_BETA * lograt * y_hat[c] * (1.0f - y_hat[c]));
    }
    if (compute_loss) {
        term_a = cg::reduce(warp, term_a, cg::plus<float>());
        term_b = cg::reduce(warp, term_b, cg::plus<float>());
        if (lane == 0)
            *loss = -LOSS_ALPHA * term_a - LOSS_BETA * term_b;
    }
#endif
}

__device__ inline void compute_asl(warp_t warp, const float* y, const float* y_hat, bool compute_loss, float* grad,
                                   float* loss) {
    int lane = warp.thread_rank();
    float acc = 0.0f;
    for (int c = lane; c < CLASSES; c += WARP_SIZE) {
        float p = fminf(fmaxf(y_hat[c], LOSS_EPS), 1.0f - LOSS_EPS);
        float pm = LOSS_CLIP > 0.0f ? fminf(fmaxf(p - LOSS_CLIP, 0.0f), 1.0f) : p;
        bool active_neg = LOSS_CLIP > 0.0f ? ((p - LOSS_CLIP) > 0.0f) : true;
        pm = fminf(fmaxf(pm, LOSS_EPS), 1.0f - LOSS_EPS);
        float log_p = logf(p);
        float log_1mpm = logf(1.0f - pm);

        if (compute_loss) {
            float loss_pos = powf(1.0f - p, LOSS_GAMMA_POS) * log_p;
            float loss_neg = powf(pm, LOSS_GAMMA_NEG) * log_1mpm;
            acc += y[c] * loss_pos + (1.0f - y[c]) * loss_neg;
        }

        float grad_pos =
            powf(1.0f - p, LOSS_GAMMA_POS + 1.0f) - LOSS_GAMMA_POS * powf(1.0f - p, LOSS_GAMMA_POS) * p * log_p;
        float grad_neg = (LOSS_GAMMA_NEG * powf(pm, LOSS_GAMMA_NEG - 1.0f) * log_1mpm -
                          powf(pm, LOSS_GAMMA_NEG) / (1.0f - pm)) *
                         p * (1.0f - p);
        grad_neg = active_neg ? grad_neg : 0.0f;
        grad[c] = y[c] * grad_pos + (1.0f - y[c]) * grad_neg;
    }
    if (compute_loss) {
        acc = cg::reduce(warp, acc, cg::plus<float>());
        if (lane == 0)
            *loss = -acc;
    }
}

__device__ inline void compute_mse(warp_t warp, const float* y, const float* y_hat, const float* class_weights,
                                   bool compute_loss, float* grad, float* loss) {
    int lane = warp.thread_rank();
    float acc = 0.0f;
    for (int c = lane; c < CLASSES; c += WARP_SIZE) {
        float d = y[c] - y_hat[c];
        if (compute_loss)
            acc += class_weights[c] * d * d;
        grad[c] = 2.0f * class_weights[c] * d * dact_of(y_hat[c]);
    }
    if (compute_loss) {
        acc = cg::reduce(warp, acc, cg::plus<float>());
        if (lane == 0)
            *loss = acc;
    }
}

__device__ inline void compute_mae(warp_t warp, const float* y, const float* y_hat, const float* class_weights,
                                   bool compute_loss, float* grad, float* loss) {
    int lane = warp.thread_rank();
    float acc = 0.0f;
    for (int c = lane; c < CLASSES; c += WARP_SIZE) {
        float d = y[c] - y_hat[c];
        float s = (d > 0.0f) - (d < 0.0f);
        if (compute_loss)
            acc += class_weights[c] * fabsf(d);
        grad[c] = class_weights[c] * s * dact_of(y_hat[c]);
    }
    if (compute_loss) {
        acc = cg::reduce(warp, acc, cg::plus<float>());
        if (lane == 0)
            *loss = acc;
    }
}

__device__ inline void compute_huber(warp_t warp, const float* y, const float* y_hat, const float* class_weights,
                                     bool compute_loss, float* grad, float* loss) {
    int lane = warp.thread_rank();
    float acc = 0.0f;
    for (int c = lane; c < CLASSES; c += WARP_SIZE) {
        float r = y[c] - y_hat[c];
        if (compute_loss) {
            float ar = fabsf(r);
            acc += class_weights[c] * (ar <= LOSS_DELTA ? 0.5f * r * r : LOSS_DELTA * (ar - 0.5f * LOSS_DELTA));
        }
        float rc = fminf(fmaxf(r, -LOSS_DELTA), LOSS_DELTA);
        grad[c] = class_weights[c] * rc * dact_of(y_hat[c]);
    }
    if (compute_loss) {
        acc = cg::reduce(warp, acc, cg::plus<float>());
        if (lane == 0)
            *loss = acc;
    }
}

__device__ inline void compute_tversky(warp_t warp, const float* y, const float* y_hat, const float* class_weights,
                                       bool compute_loss, float* grad, float* loss) {
    int lane = warp.thread_rank();
    float partial_tp = 0.0f, partial_fp = 0.0f, partial_fn = 0.0f;
    for (int c = lane; c < CLASSES; c += WARP_SIZE) {
        partial_tp += class_weights[c] * y[c] * y_hat[c];
        partial_fp += class_weights[c] * (1.0f - y[c]) * y_hat[c];
        partial_fn += class_weights[c] * y[c] * (1.0f - y_hat[c]);
    }
    float tp = cg::reduce(warp, partial_tp, cg::plus<float>());
    float fp = cg::reduce(warp, partial_fp, cg::plus<float>());
    float fn_ = cg::reduce(warp, partial_fn, cg::plus<float>());

    float N = tp + LOSS_EPS;
    float D = tp + LOSS_ALPHA * fp + LOSS_BETA * fn_ + LOSS_EPS;
    float T = N / D;
    float fw = LOSS_GAMMA * powf(1.0f - T, LOSS_GAMMA - 1.0f);

    for (int c = lane; c < CLASSES; c += WARP_SIZE) {
        float coeff = LOSS_ALPHA + y[c] * (1.0f - LOSS_ALPHA - LOSS_BETA);
        grad[c] = fw * class_weights[c] * (y[c] * D - N * coeff) / (D * D) * dact_of(y_hat[c]);
    }
    if (compute_loss && lane == 0)
        *loss = powf(1.0f - T, LOSS_GAMMA);
}

extern "C" __global__ void compute_act_loss_grad(const float* votes, const float* y, const float* class_weights,
                                                  int compute_loss_flag, float* y_hat, float* grad, float* loss) {
    bool compute_loss = compute_loss_flag != 0;
    auto warp = cg::tiled_partition<WARP_SIZE>(cg::this_thread_block());
    compute_act(warp, votes, y_hat);
    warp.sync();

#if LOSS_FN == LOSS_CE
    compute_ce(warp, y, y_hat, class_weights, compute_loss, grad, loss);
#elif LOSS_FN == LOSS_MSE
    compute_mse(warp, y, y_hat, class_weights, compute_loss, grad, loss);
#elif LOSS_FN == LOSS_MAE
    compute_mae(warp, y, y_hat, class_weights, compute_loss, grad, loss);
#elif LOSS_FN == LOSS_SCE
    compute_sce(warp, y, y_hat, class_weights, compute_loss, grad, loss);
#elif LOSS_FN == LOSS_ASL
    compute_asl(warp, y, y_hat, compute_loss, grad, loss);
#elif LOSS_FN == LOSS_TVERSKY
    compute_tversky(warp, y, y_hat, class_weights, compute_loss, grad, loss);
#elif LOSS_FN == LOSS_HUBER
    compute_huber(warp, y, y_hat, class_weights, compute_loss, grad, loss);
#endif
}

extern "C" __global__ void apply_act_batch(const float* votes, int n_samples, float* y_hat) {
    auto warp = cg::tiled_partition<WARP_SIZE>(cg::this_thread_block());
    auto grid = cg::this_grid();
    ull warp_id = grid.thread_rank() / warp.size();
    ull total_warps = grid.size() / warp.size();

    for (ull e = warp_id; e < (ull)n_samples; e += total_warps)
        compute_act(warp, &votes[e * (ull)CLASSES], &y_hat[e * (ull)CLASSES]);
}
