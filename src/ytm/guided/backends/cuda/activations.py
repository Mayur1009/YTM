import cupy as cp

_softmax_max = cp.ReductionKernel(
    "float32 v",
    "float32 out",
    "v",
    "max(a, b)",
    "out = a",
    "-1e30",
    "softmax_max",
)
_softmax_sum_exp = cp.ReductionKernel(
    "float32 v, float32 max_v",
    "float32 out",
    "exp(v - max_v)",
    "a + b",
    "out = a",
    "0",
    "softmax_sum_exp",
)
_softmax_normalize = cp.ElementwiseKernel(
    "float32 v, float32 max_v, float32 sum_exp",
    "float32 y_hat",
    "y_hat = exp(v - max_v) / sum_exp",
    "softmax_normalize",
)


def softmax(v, axis=-1):
    max_v = _softmax_max(v)
    sum_exp = _softmax_sum_exp(v, max_v)
    return _softmax_normalize(v, max_v, sum_exp)


_sigmoid = cp.ElementwiseKernel(
    "float32 v",
    "float32 y_hat",
    "y_hat = 1.0f / (1.0f + exp(-v))",
    "sigmoid",
)


def sigmoid(v):
    return _sigmoid(v)
