#ifdef IS_NEOVIM_CLANGD_ENV
#include "common.c"
#endif

#include <math.h>

#if NEGATIVE_CLAUSES
#define NORM ((float)CLAUSES_PER_CLASS / 2.0f)
#else
#define NORM ((float)CLAUSES_PER_CLASS)
#endif

static inline float dact_of(float act) {
#if ACT_FN == ACT_SIGMOID
    return act * (1.0f - act);
#else
    return 1.0f;
#endif
}

static inline void _softmax(const float* votes, float* y_hat) {
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

static inline float _sigmoid(float x) { return 1.0f / (1.0f + expf(-x / NORM)); }

static inline float _identity(float x) { return x; }

void votes_activation(const float* votes, float* y_hat) {
#if ACT_FN == ACT_SOFTMAX
    _softmax(votes, y_hat);
#elif ACT_FN == ACT_SIGMOID
    for (int c = 0; c < CLASSES; c++)
        y_hat[c] = _sigmoid(votes[c]);
#else
    for (int c = 0; c < CLASSES; c++)
        y_hat[c] = _identity(votes[c]);
#endif
}

void votes_activation_batch(const float* votes, int n_samples, float* y_hat) {
#if ACT_FN == ACT_SOFTMAX
#pragma omp parallel for schedule(static)
    for (int e = 0; e < n_samples; e++)
        _softmax(&votes[(ull)e * CLASSES], &y_hat[(ull)e * CLASSES]);
#else
    ull total = (ull)n_samples * (ull)CLASSES;
#pragma omp parallel for schedule(static)
    for (ull idx = 0; idx < total; idx++)
#if ACT_FN == ACT_SIGMOID
        y_hat[idx] = _sigmoid(votes[idx]);
#else
        y_hat[idx] = _identity(votes[idx]);
#endif
#endif
}
