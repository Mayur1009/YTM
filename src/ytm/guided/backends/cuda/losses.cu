#ifdef IS_NEOVIM_CLANGD_ENV
#include "activations.cu"
#include "common.cu"
#endif
#pragma once

__device__ inline float safe_pow(float base, float exp) {
    float b = fmaxf(base, 0.0f);
    return (exp == 0.0f) ? 1.0f : powf(b, exp);
}

__device__ inline float _ce(float a, float b, float w) { return w * a * logf(b + LOSS_EPS); }

__device__ inline float _bce(float a, float b, float w) {
    return w * (a * logf(b + LOSS_EPS) + (1.0f - a) * logf(1.0f - b + LOSS_EPS));
}

__device__ inline float _ce_grad(float a, float b, float w) { return w * (a - b); }

__device__ inline void compute_ce(const float* y, const float* y_hat, const float* class_weights, float* grad,
                                  float* loss) {
#if ACT_FN == ACT_SOFTMAX
    float p_t = 0.0f;
    for (int c = 0; c < CLASSES; c++)
        p_t += y[c] * y_hat[c];
    float fw = safe_pow(1.0f - p_t, LOSS_GAMMA);

    if (loss) {
        float loss_sum = 0.0f;
        for (int c = 0; c < CLASSES; c++) {
            loss_sum += _ce(y[c], y_hat[c], class_weights[c]);
        }
        *loss = -loss_sum * fw;
    }

    if (grad) {
        for (int c = 0; c < CLASSES; c++)
            grad[c] = fw * _ce_grad(y[c], y_hat[c], class_weights[c]);
    }
#else
    float loss_sum = 0.0f;
    for (int c = 0; c < CLASSES; c++) {
        float p_t = y[c] * y_hat[c] + (1.0f - y[c]) * (1.0f - y_hat[c]);
        float fw = safe_pow(1.0f - p_t, LOSS_GAMMA);
        if (loss)
            loss_sum += fw * _bce(y[c], y_hat[c], class_weights[c]);
        if (grad)
            grad[c] = fw * _ce_grad(y[c], y_hat[c], class_weights[c]);
    }
    if (loss)
        *loss = -loss_sum;
#endif
}

__device__ inline void compute_sce(const float* y, const float* y_hat, const float* class_weights, float* grad,
                                   float* loss) {
#if ACT_FN == ACT_SOFTMAX
    if(loss) {
        float term_ce = 0.0f, term_rce = 0.0f;
        for (int c = 0; c < CLASSES; c++) {
            term_ce += _ce(y[c], y_hat[c], class_weights[c]);
            term_rce += _ce(y_hat[c], y[c], class_weights[c]);
        }
        *loss = -LOSS_ALPHA * term_ce - LOSS_BETA * term_rce;
    }

    if (grad) {
        float dot_val = 0.0f;
        for (int c = 0; c < CLASSES; c++)
            dot_val += y_hat[c] * logf(y[c] + LOSS_EPS);
        for (int c = 0; c < CLASSES; c++) {
            float logy = logf(y[c] + LOSS_EPS);
            grad[c] = LOSS_ALPHA * _ce_grad(y[c], y_hat[c], class_weights[c]) +
                      class_weights[c] * LOSS_BETA * y_hat[c] * (logy - dot_val);
        }
    }
#else
    float term_ce = 0.0f, term_rce = 0.0f;
    for (int c = 0; c < CLASSES; c++) {
        if (loss) {
            term_ce += _bce(y[c], y_hat[c], class_weights[c]);
            term_rce += _bce(y_hat[c], y[c], class_weights[c]);
        }
        if (grad) {
            float lograt = logf((y[c] + LOSS_EPS) / (1.0f - y[c] + LOSS_EPS));
            grad[c] = LOSS_ALPHA * _ce_grad(y[c], y_hat[c], class_weights[c]) +
                      class_weights[c] * LOSS_BETA * lograt * y_hat[c] * (1.0f - y_hat[c]);
        }
    }
    if (loss)
        *loss = -LOSS_ALPHA * term_ce - LOSS_BETA * term_rce;
#endif
}

__device__ inline void compute_asl(const float* y, const float* y_hat, float* grad, float* loss) {
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
            float grad_neg =
                (LOSS_GAMMA_NEG * powf(pm, LOSS_GAMMA_NEG - 1.0f) * log_1mpm - powf(pm, LOSS_GAMMA_NEG) / (1.0f - pm)) *
                p * (1.0f - p);
            grad_neg = active_neg ? grad_neg : 0.0f;
            grad[c] = y[c] * grad_pos + (1.0f - y[c]) * grad_neg;
        }
    }
    if (loss)
        *loss = -loss_sum;
}

__device__ inline void compute_mse(const float* y, const float* y_hat, const float* class_weights, float* grad,
                                   float* loss) {
    float loss_sum = 0.0f;
    for (int c = 0; c < CLASSES; c++) {
        float d = y[c] - y_hat[c];
        if (loss)
            loss_sum += class_weights[c] * d * d;
        if (grad)
            grad[c] = 2.0f * class_weights[c] * d * dact_of(y_hat[c]);
    }
    if (loss)
        *loss = loss_sum;
}

__device__ inline void compute_mae(const float* y, const float* y_hat, const float* class_weights, float* grad,
                                   float* loss) {
    float loss_sum = 0.0f;
    for (int c = 0; c < CLASSES; c++) {
        float d = y[c] - y_hat[c];
        float s = (d > 0.0f) - (d < 0.0f);
        if (loss)
            loss_sum += class_weights[c] * fabsf(d);
        if (grad)
            grad[c] = class_weights[c] * s * dact_of(y_hat[c]);
    }
    if (loss)
        *loss = loss_sum;
}

__device__ inline void compute_huber(const float* y, const float* y_hat, const float* class_weights, float* grad,
                                     float* loss) {
    float loss_sum = 0.0f;
    for (int c = 0; c < CLASSES; c++) {
        float r = y[c] - y_hat[c];
        if (loss) {
            float ar = fabsf(r);
            loss_sum += class_weights[c] * (ar <= LOSS_DELTA ? 0.5f * r * r : LOSS_DELTA * (ar - 0.5f * LOSS_DELTA));
        }
        if (grad) {
            float rc = fminf(fmaxf(r, -LOSS_DELTA), LOSS_DELTA);
            grad[c] = class_weights[c] * rc * dact_of(y_hat[c]);
        }
    }
    if (loss)
        *loss = loss_sum;
}

__device__ inline void compute_tversky(const float* y, const float* y_hat, const float* class_weights, float* grad,
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

__device__ inline void loss_gradient_impl(const float* y_hat, const float* y, const float* class_weights, float* grad,
                                          float* loss) {
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

extern "C" __global__ void loss_gradient(const float* y_hat, const float* y, const float* class_weights, float* grad,
                                         float* loss) {
    /*
     * Calculate loss and gradient. Fully serial.
     */
    ull tid = (ull)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0) {
        loss_gradient_impl(y_hat, y, class_weights, grad, loss);
    }
}
