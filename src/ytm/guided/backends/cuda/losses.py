import cupy as cp
import numpy as np

_ce_softmax_grad = cp.ElementwiseKernel(
    "float32 y, float32 y_hat, float32 lw, float32 fw",
    "float32 grad",
    "grad = lw * fw * (y - y_hat)",
    "ce_softmax_grad",
)
_ce_softmax_loss_term = cp.ElementwiseKernel(
    "float32 y, float32 y_hat, float32 lw, float32 eps",
    "float32 term",
    "term = lw * y * log(y_hat + eps)",
    "ce_softmax_loss_term",
)
_ce_grad = cp.ElementwiseKernel(
    "float32 y, float32 y_hat, float32 lw, float32 gamma",
    "float32 grad",
    """
    float p_t = y * y_hat + (1.0f - y) * (1.0f - y_hat);
    float fw = powf(1.0f - p_t, gamma);
    grad = lw * fw * (y - y_hat);
    """,
    "ce_grad",
)
_ce_loss_term = cp.ElementwiseKernel(
    "float32 y, float32 y_hat, float32 lw, float32 gamma, float32 eps",
    "float32 term",
    """
    float p_t = y * y_hat + (1.0f - y) * (1.0f - y_hat);
    float fw = powf(1.0f - p_t, gamma);
    term = lw * fw * (y * log(y_hat + eps) + (1.0f - y) * log(1.0f - y_hat + eps));
    """,
    "ce_loss_term",
)


def build_ce(lw, gamma, eps, act_fn):
    gamma = np.float32(gamma)
    eps = np.float32(eps)

    if act_fn == "softmax":

        def _loss_fn(y, y_hat):
            fw = (1.0 - cp.dot(y, y_hat)) ** gamma
            return float(-cp.sum(_ce_softmax_loss_term(y, y_hat, lw, eps)) * fw)

        def _grad_fn(y, y_hat, grad):
            fw = (1.0 - cp.dot(y, y_hat)) ** gamma
            _ce_softmax_grad(y, y_hat, lw, fw, grad)
    else:

        def _loss_fn(y, y_hat):
            return float(-cp.sum(_ce_loss_term(y, y_hat, lw, gamma, eps)))

        def _grad_fn(y, y_hat, grad):
            _ce_grad(y, y_hat, lw, gamma, grad)

    return _loss_fn, _grad_fn


_asl_grad = cp.ElementwiseKernel(
    "float32 y, float32 y_hat, float32 gamma_pos, float32 gamma_neg, float32 clip_val, float32 eps",
    "float32 grad",
    """
    float p = fminf(fmaxf(y_hat, eps), 1.0f - eps);
    float pm = clip_val > 0.0f ? fminf(fmaxf(p - clip_val, 0.0f), 1.0f) : p;
    bool active_neg = clip_val > 0.0f ? ((p - clip_val) > 0.0f) : true;
    pm = fminf(fmaxf(pm, eps), 1.0f - eps);

    float grad_pos = powf(1.0f - p, gamma_pos + 1.0f) - gamma_pos * powf(1.0f - p, gamma_pos) * p * logf(p);
    float grad_neg = (gamma_neg * powf(pm, gamma_neg - 1.0f) * logf(1.0f - pm) - powf(pm, gamma_neg) / (1.0f - pm)) *
                     p * (1.0f - p);
    grad_neg = active_neg ? grad_neg : 0.0f;

    grad = y * grad_pos + (1.0f - y) * grad_neg;
    """,
    "asl_grad",
)
_asl_loss_term = cp.ElementwiseKernel(
    "float32 y, float32 y_hat, float32 gamma_pos, float32 gamma_neg, float32 clip_val, float32 eps",
    "float32 term",
    """
    float p = fminf(fmaxf(y_hat, eps), 1.0f - eps);
    float pm = clip_val > 0.0f ? fminf(fmaxf(p - clip_val, 0.0f), 1.0f) : p;
    pm = fminf(fmaxf(pm, eps), 1.0f - eps);

    float loss_pos = powf(1.0f - p, gamma_pos) * logf(p);
    float loss_neg = powf(pm, gamma_neg) * logf(1.0f - pm);
    term = y * loss_pos + (1.0f - y) * loss_neg;
    """,
    "asl_loss_term",
)


def build_asl(gamma_pos, gamma_neg, clip, eps):
    gamma_pos = np.float32(gamma_pos)
    gamma_neg = np.float32(gamma_neg)
    clip = np.float32(clip)
    eps = np.float32(eps)

    def _loss_fn(y, y_hat):
        return float(-cp.sum(_asl_loss_term(y, y_hat, gamma_pos, gamma_neg, clip, eps)))

    def _grad_fn(y, y_hat, grad):
        _asl_grad(y, y_hat, gamma_pos, gamma_neg, clip, eps, grad)

    return _loss_fn, _grad_fn


_sce_log = cp.ElementwiseKernel("float32 y, float32 eps", "float32 logy", "logy = log(y + eps)", "sce_log")
_sce_softmax_grad = cp.ElementwiseKernel(
    "float32 y, float32 y_hat, float32 lw, float32 logy, float32 dot_val, float32 alpha, float32 beta",
    "float32 grad",
    "grad = lw * (alpha * (y - y_hat) + beta * y_hat * (logy - dot_val))",
    "sce_softmax_grad",
)
_sce_softmax_loss_term = cp.ElementwiseKernel(
    "float32 y, float32 y_hat, float32 lw, float32 logy, float32 eps",
    "float32 term_a, float32 term_b",
    """
    term_a = lw * y * logy;
    term_b = lw * y_hat * log(y + eps);
    """,
    "sce_softmax_loss_term",
)
_sce_grad = cp.ElementwiseKernel(
    "float32 y, float32 y_hat, float32 lw, float32 alpha, float32 beta, float32 eps",
    "float32 grad",
    """
    float lograt = log((y + eps) / (1.0f - y + eps));
    grad = lw * (alpha * (y - y_hat) + beta * lograt * y_hat * (1.0f - y_hat));
    """,
    "sce_grad",
)
_sce_loss_term = cp.ElementwiseKernel(
    "float32 y, float32 y_hat, float32 lw, float32 eps",
    "float32 term_a, float32 term_b",
    """
    term_a = lw * (y * log(y_hat + eps) + (1.0f - y) * log(1.0f - y_hat + eps));
    term_b = lw * (y_hat * log(y + eps) + (1.0f - y_hat) * log(1.0f - y + eps));
    """,
    "sce_loss_term",
)


def build_sce(lw, alpha, beta, eps, act_fn):
    alpha = np.float32(alpha)
    beta = np.float32(beta)
    eps = np.float32(eps)

    if act_fn == "softmax":

        def _loss_fn(y, y_hat):
            logy = _sce_log(y, eps)
            term_a, term_b = _sce_softmax_loss_term(y, y_hat, lw, logy, eps)
            return float(-alpha * cp.sum(term_a) - beta * cp.sum(term_b))

        def _grad_fn(y, y_hat, grad):
            logy = _sce_log(y, eps)
            dot_val = cp.dot(y_hat, logy)
            _sce_softmax_grad(y, y_hat, lw, logy, dot_val, alpha, beta, grad)
    else:

        def _loss_fn(y, y_hat):
            term_a, term_b = _sce_loss_term(y, y_hat, lw, eps)
            return float(-alpha * cp.sum(term_a) - beta * cp.sum(term_b))

        def _grad_fn(y, y_hat, grad):
            _sce_grad(y, y_hat, lw, alpha, beta, eps, grad)

    return _loss_fn, _grad_fn


_mse_grad = cp.ElementwiseKernel(
    "float32 y, float32 y_hat, float32 lw, float32 dact",
    "float32 grad",
    "grad = 2.0f * lw * (y - y_hat) * dact",
    "mse_grad",
)
_mse_loss_term = cp.ElementwiseKernel(
    "float32 y, float32 y_hat, float32 lw",
    "float32 term",
    "term = lw * (y - y_hat) * (y - y_hat)",
    "mse_loss_term",
)


def build_mse(lw, dact_fn):
    def _loss_fn(y, y_hat):
        return float(cp.sum(_mse_loss_term(y, y_hat, lw)))

    def _grad_fn(y, y_hat, grad):
        dact = dact_fn(y_hat)
        _mse_grad(y, y_hat, lw, dact, grad)

    return _loss_fn, _grad_fn


_mae_grad = cp.ElementwiseKernel(
    "float32 y, float32 y_hat, float32 lw, float32 dact",
    "float32 grad",
    """
    float d = y - y_hat;
    float s = (d > 0.0f) - (d < 0.0f);
    grad = lw * s * dact;
    """,
    "mae_grad",
)
_mae_loss_term = cp.ElementwiseKernel(
    "float32 y, float32 y_hat, float32 lw",
    "float32 term",
    "term = lw * fabsf(y - y_hat)",
    "mae_loss_term",
)


def build_mae(lw, dact_fn):
    def _loss_fn(y, y_hat):
        return float(cp.sum(_mae_loss_term(y, y_hat, lw)))

    def _grad_fn(y, y_hat, grad):
        dact = dact_fn(y_hat)
        _mae_grad(y, y_hat, lw, dact, grad)

    return _loss_fn, _grad_fn


_huber_grad = cp.ElementwiseKernel(
    "float32 y, float32 y_hat, float32 lw, float32 delta, float32 dact",
    "float32 grad",
    """
    float r = fminf(fmaxf(y - y_hat, -delta), delta);
    grad = lw * r * dact;
    """,
    "huber_grad",
)
_huber_loss_term = cp.ElementwiseKernel(
    "float32 y, float32 y_hat, float32 lw, float32 delta",
    "float32 term",
    """
    float r = y - y_hat;
    float ar = fabsf(r);
    term = lw * (ar <= delta ? 0.5f * r * r : delta * (ar - 0.5f * delta));
    """,
    "huber_loss_term",
)


def build_huber(lw, delta, dact_fn):
    delta = np.float32(delta)

    def _loss_fn(y, y_hat):
        return float(cp.sum(_huber_loss_term(y, y_hat, lw, delta)))

    def _grad_fn(y, y_hat, grad):
        dact = dact_fn(y_hat)
        _huber_grad(y, y_hat, lw, delta, dact, grad)

    return _loss_fn, _grad_fn


_tversky_tp = cp.ReductionKernel(
    "float32 y, float32 yhat, float32 lw", "float32 out", "lw * y * yhat", "a + b", "out = a", "0", "tversky_tp"
)
_tversky_fp = cp.ReductionKernel(
    "float32 y, float32 yhat, float32 lw",
    "float32 out",
    "lw * (1.0f - y) * yhat",
    "a + b",
    "out = a",
    "0",
    "tversky_fp",
)
_tversky_fn = cp.ReductionKernel(
    "float32 y, float32 yhat, float32 lw",
    "float32 out",
    "lw * y * (1.0f - yhat)",
    "a + b",
    "out = a",
    "0",
    "tversky_fn",
)
_tversky_scalar = cp.ElementwiseKernel(
    "float32 tp, float32 fp, float32 fn_, float32 alpha, float32 beta, float32 gamma, float32 eps",
    "float32 fw, float32 N, float32 D",
    """
    N = tp + eps;
    D = tp + alpha * fp + beta * fn_ + eps;
    float T = N / D;
    fw = gamma * powf(1.0f - T, gamma - 1.0f);
    """,
    "tversky_scalar",
)
_tversky_grad = cp.ElementwiseKernel(
    "float32 y, float32 lw, float32 fw, float32 N, float32 D, float32 alpha, float32 beta, float32 dact",
    "float32 grad",
    """
    float coeff = alpha + y * (1.0f - alpha - beta);
    grad = fw * lw * (y * D - N * coeff) / (D * D) * dact;
    """,
    "tversky_grad",
)


def build_tversky(lw, alpha, beta, gamma, eps, dact_fn):
    alpha = np.float32(alpha)
    beta = np.float32(beta)
    gamma = np.float32(gamma)
    eps = np.float32(eps)

    def _loss_fn(y, y_hat):
        tp = _tversky_tp(y, y_hat, lw)
        fp = _tversky_fp(y, y_hat, lw)
        fn_ = _tversky_fn(y, y_hat, lw)
        N = tp + eps
        D = tp + alpha * fp + beta * fn_ + eps
        T = N / D
        return float((1.0 - T) ** gamma)

    def _grad_fn(y, y_hat, grad):
        tp = _tversky_tp(y, y_hat, lw)
        fp = _tversky_fp(y, y_hat, lw)
        fn_ = _tversky_fn(y, y_hat, lw)
        fw, N, D = _tversky_scalar(tp, fp, fn_, alpha, beta, gamma, eps)
        dact = dact_fn(y_hat)
        _tversky_grad(y, lw, fw, N, D, alpha, beta, dact, grad)

    return _loss_fn, _grad_fn
