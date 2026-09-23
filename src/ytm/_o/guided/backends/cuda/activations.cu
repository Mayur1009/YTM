#ifdef IS_NEOVIM_CLANGD_ENV
#include "common.cu"
#endif
#pragma once

#if NEGATIVE_CLAUSES
#define NORM ((float)CLAUSES_PER_CLASS / 2.0f)
#else
#define NORM ((float)CLAUSES_PER_CLASS)
#endif

__device__ inline float dact_of(float act) {
#if ACT_FN == ACT_SIGMOID
    return act * (1.0f - act);
#else
    return 1.0f;
#endif
}

__device__ inline void _softmax(const float* votes, float* y_hat) {
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

__device__ inline float _sigmoid(float x) { return 1.0f / (1.0f + expf(-x / NORM)); }

__device__ inline float _identity(float x) { return x; }

extern "C" __global__ void votes_activation(const float* votes, float* y_hat) {
    /*
     * Apply activation function to the clause votes.
     * Parallel across the vector, except for softmax
     */
    ull tid = (ull)blockIdx.x * blockDim.x + threadIdx.x;
    ull stride = (ull)blockDim.x * gridDim.x;

#if ACT_FN == ACT_SOFTMAX
    if (tid == 0)
        _softmax(votes, y_hat);
#elif ACT_FN == ACT_SIGMOID
    for (ull c = tid; c < (ull)CLASSES; c += stride)
        y_hat[c] = _sigmoid(votes[c]);
#else
    for (ull c = tid; c < (ull)CLASSES; c += stride)
        y_hat[c] = _identity(votes[c]);
#endif
}

extern "C" __global__ void votes_activation_batch(const float* votes, int n_samples, float* y_hat) {
    /*
     * Apply activation function to a batch of votes in parallel.
     */
    ull tid = (ull)blockIdx.x * blockDim.x + threadIdx.x;
    ull stride = (ull)blockDim.x * gridDim.x;

#if ACT_FN == ACT_SOFTMAX
    for (ull e = tid; e < (ull)n_samples; e += stride)
        _softmax(&votes[e * (ull)CLASSES], &y_hat[e * (ull)CLASSES]);
#else
    ull total = (ull)n_samples * (ull)CLASSES;
    for (ull idx = tid; idx < total; idx += stride)
#if ACT_FN == ACT_SIGMOID
        y_hat[idx] = _sigmoid(votes[idx]);
#else
        y_hat[idx] = _identity(votes[idx]);
#endif
#endif
}
