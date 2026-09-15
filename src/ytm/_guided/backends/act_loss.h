#ifdef IS_NEOVIM_CLANGD_ENV
#include "../../_core/backends/common.h"
#include "../../_core/backends/cpu.h"

#define ACT_SOFTMAX 0
#define ACT_SIGMOID 1
#define ACT_IDENTITY 2
#define ACT_FN ACT_SOFTMAX

#define LOSS_CE 0
#define LOSS_MSE 1
#define LOSS_MAE 2
#define LOSS_SCE 3
#define LOSS_ASL 4
#define LOSS_TVERSKY 5
#define LOSS_HUBER 6
#define LOSS_FN LOSS_CE
#endif

#if NEGATIVE_CLAUSES
#define NORM ((float)CLAUSES_PER_CLASS / 2.0f)
#else
#define NORM ((float)CLAUSES_PER_CLASS)
#endif

INLINE_FN float dact_of(float act) {
#if ACT_FN == ACT_SIGMOID
    return act * (1.0f - act);
#else
    return 1.0f;
#endif
}

INLINE_FN void _softmax(const float* votes, float* y_hat) {
    float max_v = -INFINITY;
    for (int c = 0; c < CLASSES; c++) {
        float v = votes[c] / NORM;
        if (v > max_v)
            max_v = v;
    }
    float sum_exp = 0.0f;
    for (int c = 0; c < CLASSES; c++) {
        float e = expf((votes[c] / NORM) - max_v);
        y_hat[c] = e;
        sum_exp += e;
    }
    for (int c = 0; c < CLASSES; c++)
        y_hat[c] /= sum_exp;
}

INLINE_FN float _sigmoid(float x) { return 1.0f / (1.0f + expf(-x / NORM)); }

INLINE_FN float _identity(float x) { return x; }

INLINE_FN float _ce(float a, float b) { return a * logf(b + LOSS_EPS); }

INLINE_FN float _bce(float a, float b) { return a * logf(b + LOSS_EPS) + (1.0f - a) * logf(1.0f - b + LOSS_EPS); }

INLINE_FN float _ce_grad(float a, float b) { return a - b; }

INLINE_FN void compute_ce(const float* y, const float* y_hat, float* grad, float* loss) {
#if ACT_FN == ACT_SOFTMAX
    if (loss) {
        float loss_sum = 0.0f;
        for (int c = 0; c < CLASSES; c++)
            loss_sum += _ce(y[c], y_hat[c]);
        *loss = -loss_sum;
    }

    if (grad) {
        for (int c = 0; c < CLASSES; c++)
            grad[c] = _ce_grad(y[c], y_hat[c]);
    }
#else
    float loss_sum = 0.0f;
    for (int c = 0; c < CLASSES; c++) {
        if (loss)
            loss_sum += _bce(y[c], y_hat[c]);
        if (grad)
            grad[c] = _ce_grad(y[c], y_hat[c]);
    }
    if (loss)
        *loss = -loss_sum;
#endif
}

INLINE_FN void compute_sce(const float* y, const float* y_hat, float* grad, float* loss) {
#if ACT_FN == ACT_SOFTMAX
    if (loss) {
        float term_ce = 0.0f, term_rce = 0.0f;
        for (int c = 0; c < CLASSES; c++) {
            term_ce += _ce(y[c], y_hat[c]);
            term_rce += _ce(y_hat[c], y[c]);
        }
        *loss = -LOSS_ALPHA * term_ce - LOSS_BETA * term_rce;
    }

    if (grad) {
        float dot_val = 0.0f;
        for (int c = 0; c < CLASSES; c++)
            dot_val += y_hat[c] * logf(y[c] + LOSS_EPS);
        for (int c = 0; c < CLASSES; c++) {
            float logy = logf(y[c] + LOSS_EPS);
            grad[c] = LOSS_ALPHA * _ce_grad(y[c], y_hat[c]) + LOSS_BETA * y_hat[c] * (logy - dot_val);
        }
    }
#else
    float term_ce = 0.0f, term_rce = 0.0f;
    for (int c = 0; c < CLASSES; c++) {
        if (loss) {
            term_ce += _bce(y[c], y_hat[c]);
            term_rce += _bce(y_hat[c], y[c]);
        }
        if (grad) {
            float lograt = logf((y[c] + LOSS_EPS) / (1.0f - y[c] + LOSS_EPS));
            grad[c] = LOSS_ALPHA * _ce_grad(y[c], y_hat[c]) + LOSS_BETA * lograt * y_hat[c] * (1.0f - y_hat[c]);
        }
    }
    if (loss)
        *loss = -LOSS_ALPHA * term_ce - LOSS_BETA * term_rce;
#endif
}

INLINE_FN void compute_asl(const float* y, const float* y_hat, float* grad, float* loss) {
    float loss_sum = 0.0f;
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
            loss_sum += y[c] * loss_pos + (1.0f - y[c]) * loss_neg;
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
        *loss = -loss_sum;
}

INLINE_FN void compute_mse(const float* y, const float* y_hat, float* grad, float* loss) {
    float loss_sum = 0.0f;
    for (int c = 0; c < CLASSES; c++) {
        float d = y[c] - y_hat[c];
        if (loss)
            loss_sum += d * d;
        if (grad)
            grad[c] = 2.0f * d * dact_of(y_hat[c]);
    }
    if (loss)
        *loss = loss_sum;
}

INLINE_FN void compute_mae(const float* y, const float* y_hat, float* grad, float* loss) {
    float loss_sum = 0.0f;
    for (int c = 0; c < CLASSES; c++) {
        float d = y[c] - y_hat[c];
        float s = (d > 0.0f) - (d < 0.0f);
        if (loss)
            loss_sum += fabsf(d);
        if (grad)
            grad[c] = s * dact_of(y_hat[c]);
    }
    if (loss)
        *loss = loss_sum;
}

INLINE_FN void compute_huber(const float* y, const float* y_hat, float* grad, float* loss) {
    float loss_sum = 0.0f;
    for (int c = 0; c < CLASSES; c++) {
        float r = y[c] - y_hat[c];
        if (loss) {
            float ar = fabsf(r);
            loss_sum += (ar <= LOSS_DELTA ? 0.5f * r * r : LOSS_DELTA * (ar - 0.5f * LOSS_DELTA));
        }
        if (grad) {
            float rc = fminf(fmaxf(r, -LOSS_DELTA), LOSS_DELTA);
            grad[c] = rc * dact_of(y_hat[c]);
        }
    }
    if (loss)
        *loss = loss_sum;
}

INLINE_FN void compute_tversky(const float* y, const float* y_hat, float* grad, float* loss) {
    float tp = 0.0f, fp = 0.0f, fn_ = 0.0f;
    for (int c = 0; c < CLASSES; c++) {
        tp += y[c] * y_hat[c];
        fp += (1.0f - y[c]) * y_hat[c];
        fn_ += y[c] * (1.0f - y_hat[c]);
    }
    float N = tp + LOSS_EPS;
    float D = tp + LOSS_ALPHA * fp + LOSS_BETA * fn_ + LOSS_EPS;
    float T = N / D;

    if (grad) {
        for (int c = 0; c < CLASSES; c++) {
            float coeff = LOSS_ALPHA + y[c] * (1.0f - LOSS_ALPHA - LOSS_BETA);
            grad[c] = (y[c] * D - N * coeff) / (D * D) * dact_of(y_hat[c]);
        }
    }
    if (loss)
        *loss = 1.0f - T;
}

INLINE_FN void loss_gradient_impl(const float* y_hat, const float* y, float* grad, float* loss) {
#if LOSS_FN == LOSS_CE
    compute_ce(y, y_hat, grad, loss);
#elif LOSS_FN == LOSS_MSE
    compute_mse(y, y_hat, grad, loss);
#elif LOSS_FN == LOSS_MAE
    compute_mae(y, y_hat, grad, loss);
#elif LOSS_FN == LOSS_SCE
    compute_sce(y, y_hat, grad, loss);
#elif LOSS_FN == LOSS_ASL
    compute_asl(y, y_hat, grad, loss);
#elif LOSS_FN == LOSS_TVERSKY
    compute_tversky(y, y_hat, grad, loss);
#elif LOSS_FN == LOSS_HUBER
    compute_huber(y, y_hat, grad, loss);
#endif
}
