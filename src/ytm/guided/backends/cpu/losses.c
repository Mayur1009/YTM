#ifdef IS_NEOVIM_CLANGD_ENV
#include "common.c"
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

#include <math.h>

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

static inline float safe_pow(float base, float exp) {
    float b = (base < 0.0f) ? 0.0f : base;
    return (exp == 0.0f) ? 1.0f : powf(b, exp);
}

static inline float vote_at(const float* votes, int c, int vote_diff_ind, float vote_diff_delta) {
    return votes[c] + (c == vote_diff_ind ? vote_diff_delta : 0.0f);
}

static inline void compute_act(const float* votes, float* y_hat, int vote_diff_ind, float vote_diff_delta) {
    int clauses_per_class = CLAUSES_PER_CLASS;
#if ACT_FN == ACT_SOFTMAX
    float max_v = NEG_INF;
    for (int c = 0; c < CLASSES; c++) {
        float v = vote_at(votes, c, vote_diff_ind, vote_diff_delta) / (float)clauses_per_class;
        if (v > max_v)
            max_v = v;
    }
    float sum_exp = 0.0f;
    for (int c = 0; c < CLASSES; c++) {
        float raw = vote_at(votes, c, vote_diff_ind, vote_diff_delta);
        y_hat[c] = expf((raw / (float)clauses_per_class) - max_v);
        sum_exp += y_hat[c];
    }
    for (int c = 0; c < CLASSES; c++)
        y_hat[c] /= sum_exp;
#elif ACT_FN == ACT_SIGMOID
    for (int c = 0; c < CLASSES; c++) {
        float raw = vote_at(votes, c, vote_diff_ind, vote_diff_delta);
        y_hat[c] = 1.0f / (1.0f + expf(-raw / (float)clauses_per_class));
    }
#else
    for (int c = 0; c < CLASSES; c++)
        y_hat[c] = vote_at(votes, c, vote_diff_ind, vote_diff_delta);
#endif
}

static inline float dact_of(float act) {
#if ACT_FN == ACT_SIGMOID
    return act * (1.0f - act);
#else
    return 1.0f;
#endif
}

static inline void compute_ce(const float* y, const float* y_hat, const float* class_weights, float* grad,
                              float* loss) {
#if ACT_FN == ACT_SOFTMAX
    float dot_val = 0.0f;
    for (int c = 0; c < CLASSES; c++)
        dot_val += y[c] * y_hat[c];
    float fw = safe_pow(1.0f - dot_val, LOSS_GAMMA);

    float acc = 0.0f;
    for (int c = 0; c < CLASSES; c++) {
        if (loss)
            acc += class_weights[c] * y[c] * logf(y_hat[c] + LOSS_EPS);
        if (grad)
            grad[c] = class_weights[c] * fw * (y[c] - y_hat[c]);
    }
    if (loss)
        *loss = -acc * fw;
#else
    float acc = 0.0f;
    for (int c = 0; c < CLASSES; c++) {
        float p_t = y[c] * y_hat[c] + (1.0f - y[c]) * (1.0f - y_hat[c]);
        float fw = safe_pow(1.0f - p_t, LOSS_GAMMA);
        if (loss)
            acc += class_weights[c] * fw *
                   (y[c] * logf(y_hat[c] + LOSS_EPS) + (1.0f - y[c]) * logf(1.0f - y_hat[c] + LOSS_EPS));
        if (grad)
            grad[c] = class_weights[c] * fw * (y[c] - y_hat[c]);
    }
    if (loss)
        *loss = -acc;
#endif
}

static inline void compute_sce(const float* y, const float* y_hat, const float* class_weights, float* grad,
                               float* loss) {
#if ACT_FN == ACT_SOFTMAX
    float logy[CLASSES];
    for (int c = 0; c < CLASSES; c++)
        logy[c] = logf(y[c] + LOSS_EPS);

    float dot_val = 0.0f;
    for (int c = 0; c < CLASSES; c++)
        dot_val += y_hat[c] * logy[c];

    float term_a = 0.0f, term_b = 0.0f;
    for (int c = 0; c < CLASSES; c++) {
        if (loss) {
            term_a += class_weights[c] * y[c] * logf(y_hat[c] + LOSS_EPS);
            term_b += class_weights[c] * y_hat[c] * logy[c];
        }
        if (grad)
            grad[c] = class_weights[c] * (LOSS_ALPHA * (y[c] - y_hat[c]) + LOSS_BETA * y_hat[c] * (logy[c] - dot_val));
    }
    if (loss)
        *loss = -LOSS_ALPHA * term_a - LOSS_BETA * term_b;
#else
    float term_a = 0.0f, term_b = 0.0f;
    for (int c = 0; c < CLASSES; c++) {
        if (loss) {
            term_a += class_weights[c] *
                      (y[c] * logf(y_hat[c] + LOSS_EPS) + (1.0f - y[c]) * logf(1.0f - y_hat[c] + LOSS_EPS));
            term_b += class_weights[c] *
                      (y_hat[c] * logf(y[c] + LOSS_EPS) + (1.0f - y_hat[c]) * logf(1.0f - y[c] + LOSS_EPS));
        }
        if (grad) {
            float lograt = logf((y[c] + LOSS_EPS) / (1.0f - y[c] + LOSS_EPS));
            grad[c] =
                class_weights[c] * (LOSS_ALPHA * (y[c] - y_hat[c]) + LOSS_BETA * lograt * y_hat[c] * (1.0f - y_hat[c]));
        }
    }
    if (loss)
        *loss = -LOSS_ALPHA * term_a - LOSS_BETA * term_b;
#endif
}

static inline void compute_asl(const float* y, const float* y_hat, float* grad, float* loss) {
    float acc = 0.0f;
    for (int c = 0; c < CLASSES; c++) {
        float p = fminf(fmaxf(y_hat[c], LOSS_EPS), 1.0f - LOSS_EPS);
        float pm = LOSS_CLIP > 0.0f ? fminf(fmaxf(p - LOSS_CLIP, 0.0f), 1.0f) : p;
        bool active_neg = LOSS_CLIP > 0.0f ? ((p - LOSS_CLIP) > 0.0f) : true;
        pm = fminf(fmaxf(pm, LOSS_EPS), 1.0f - LOSS_EPS);
        float log_p = logf(p);
        float log_1mpm = logf(1.0f - pm);

        if (loss) {
            float loss_pos = powf(1.0f - p, LOSS_GAMMA_POS) * log_p;
            float loss_neg = powf(pm, LOSS_GAMMA_NEG) * log_1mpm;
            acc += y[c] * loss_pos + (1.0f - y[c]) * loss_neg;
        }

        if (grad) {
            float grad_pos =
                powf(1.0f - p, LOSS_GAMMA_POS + 1.0f) - LOSS_GAMMA_POS * powf(1.0f - p, LOSS_GAMMA_POS) * p * log_p;
            float grad_neg = (LOSS_GAMMA_NEG * powf(pm, LOSS_GAMMA_NEG - 1.0f) * log_1mpm -
                              powf(pm, LOSS_GAMMA_NEG) / (1.0f - pm)) *
                             p * (1.0f - p);
            grad_neg = active_neg ? grad_neg : 0.0f;
            grad[c] = y[c] * grad_pos + (1.0f - y[c]) * grad_neg;
        }
    }
    if (loss)
        *loss = -acc;
}

static inline void compute_mse(const float* y, const float* y_hat, const float* class_weights, float* grad,
                               float* loss) {
    float acc = 0.0f;
    for (int c = 0; c < CLASSES; c++) {
        float d = y[c] - y_hat[c];
        if (loss)
            acc += class_weights[c] * d * d;
        if (grad)
            grad[c] = 2.0f * class_weights[c] * d * dact_of(y_hat[c]);
    }
    if (loss)
        *loss = acc;
}

static inline void compute_mae(const float* y, const float* y_hat, const float* class_weights, float* grad,
                               float* loss) {
    float acc = 0.0f;
    for (int c = 0; c < CLASSES; c++) {
        float d = y[c] - y_hat[c];
        float s = (d > 0.0f) - (d < 0.0f);
        if (loss)
            acc += class_weights[c] * fabsf(d);
        if (grad)
            grad[c] = class_weights[c] * s * dact_of(y_hat[c]);
    }
    if (loss)
        *loss = acc;
}

static inline void compute_huber(const float* y, const float* y_hat, const float* class_weights, float* grad,
                                 float* loss) {
    float acc = 0.0f;
    for (int c = 0; c < CLASSES; c++) {
        float r = y[c] - y_hat[c];
        if (loss) {
            float ar = fabsf(r);
            acc += class_weights[c] * (ar <= LOSS_DELTA ? 0.5f * r * r : LOSS_DELTA * (ar - 0.5f * LOSS_DELTA));
        }
        if (grad) {
            float rc = fminf(fmaxf(r, -LOSS_DELTA), LOSS_DELTA);
            grad[c] = class_weights[c] * rc * dact_of(y_hat[c]);
        }
    }
    if (loss)
        *loss = acc;
}

static inline void compute_tversky(const float* y, const float* y_hat, const float* class_weights, float* grad,
                                   float* loss) {
    float tp = 0.0f, fp = 0.0f, fn_ = 0.0f;
    for (int c = 0; c < CLASSES; c++) {
        tp += class_weights[c] * y[c] * y_hat[c];
        fp += class_weights[c] * (1.0f - y[c]) * y_hat[c];
        fn_ += class_weights[c] * y[c] * (1.0f - y_hat[c]);
    }
    float N = tp + LOSS_EPS;
    float D = tp + LOSS_ALPHA * fp + LOSS_BETA * fn_ + LOSS_EPS;
    float T = N / D;

    if (grad) {
        float fw = LOSS_GAMMA * safe_pow(1.0f - T, LOSS_GAMMA - 1.0f);
        for (int c = 0; c < CLASSES; c++) {
            float coeff = LOSS_ALPHA + y[c] * (1.0f - LOSS_ALPHA - LOSS_BETA);
            grad[c] = fw * class_weights[c] * (y[c] * D - N * coeff) / (D * D) * dact_of(y_hat[c]);
        }
    }
    if (loss)
        *loss = safe_pow(1.0f - T, LOSS_GAMMA);
}

void compute_act_loss_grad(const float* votes, const float* y, const float* class_weights, float* y_hat, float* grad,
                           float* loss, int vote_diff_ind, float vote_diff_delta) {
    compute_act(votes, y_hat, vote_diff_ind, vote_diff_delta);

#if LOSS_FN == LOSS_CE
    compute_ce(y, y_hat, class_weights, grad, loss);
#elif LOSS_FN == LOSS_MSE
    compute_mse(y, y_hat, class_weights, grad, loss);
#elif LOSS_FN == LOSS_MAE
    compute_mae(y, y_hat, class_weights, grad, loss);
#elif LOSS_FN == LOSS_SCE
    compute_sce(y, y_hat, class_weights, grad, loss);
#elif LOSS_FN == LOSS_ASL
    compute_asl(y, y_hat, grad, loss);
#elif LOSS_FN == LOSS_TVERSKY
    compute_tversky(y, y_hat, class_weights, grad, loss);
#elif LOSS_FN == LOSS_HUBER
    compute_huber(y, y_hat, class_weights, grad, loss);
#endif
}

void apply_act_batch(const float* votes, int n_samples, float* y_hat) {
#pragma omp parallel for schedule(static)
    for (int e = 0; e < n_samples; e++)
        compute_act(&votes[(ull)e * CLASSES], &y_hat[(ull)e * CLASSES], -1, 0.0f);
}
