#ifdef IS_NEOVIM_CLANGD_ENV
#include "../../_core/backends/common.h"
#include "../../_core/backends/cpu.h"

#define ACT_SOFTMAX 0
#define ACT_SIGMOID 1
#define ACT_IDENTITY 2
#define ACT_FN ACT_SOFTMAX
#endif

#define NORM ((float)CLAUSES_PER_CLASS / (NEGATIVE_CLAUSES ? 2.0f : 1.0f))

INLINE_FN void compute_act(const float* RESTRICT votes, float* RESTRICT y_hat) {
    if (ACT_FN == ACT_SIGMOID) {
        for (int c = 0; c < CLASSES; c++)
            y_hat[c] = 1.0f / (1.0f + expf(-votes[c] / NORM));
        return;
    }
    if (ACT_FN != ACT_SOFTMAX) {
        for (int c = 0; c < CLASSES; c++)
            y_hat[c] = votes[c] / NORM;
        return;
    }
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
