import numpy as np


def build_ce(lw, gamma, eps, act_fn):
    if act_fn == "softmax":

        def _loss_fn(y, y_hat):
            fw = (1.0 - np.dot(y, y_hat)) ** gamma
            return -np.sum(lw * y * np.log(y_hat + eps)) * fw

        def _grad_fn(y, y_hat, grad):
            fw = (1.0 - np.dot(y, y_hat)) ** gamma
            grad[:] = lw * fw * (y - y_hat)
    else:

        def _loss_fn(y, y_hat):
            p_t = y * y_hat + (1.0 - y) * (1.0 - y_hat)
            fw = (1.0 - p_t) ** gamma
            return -np.sum(lw * fw * (y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps)))

        def _grad_fn(y, y_hat, grad):
            p_t = y * y_hat + (1.0 - y) * (1.0 - y_hat)
            fw = (1.0 - p_t) ** gamma
            grad[:] = lw * fw * (y - y_hat)

    return _loss_fn, _grad_fn


def build_asl(gamma_pos, gamma_neg, clip, eps):
    def _loss_fn(y, y_hat):
        p = np.clip(y_hat, eps, 1.0 - eps)
        pm = np.clip(p - clip, 0.0, 1.0) if clip > 0 else p
        pm = np.clip(pm, eps, 1.0 - eps)

        loss_pos = (1.0 - p) ** gamma_pos * np.log(p)
        loss_neg = (pm**gamma_neg) * np.log(1.0 - pm)
        return -np.sum(y * loss_pos + (1.0 - y) * loss_neg)

    def _grad_fn(y, y_hat, grad):
        p = np.clip(y_hat, eps, 1.0 - eps)
        pm = np.clip(p - clip, 0.0, 1.0) if clip > 0 else p
        active_neg = (p - clip) > 0 if clip > 0 else np.ones_like(p, dtype=bool)
        pm = np.clip(pm, eps, 1.0 - eps)

        grad_pos = (1.0 - p) ** (gamma_pos + 1.0) - gamma_pos * (1.0 - p) ** gamma_pos * p * np.log(p)
        grad_neg = (gamma_neg * pm ** (gamma_neg - 1.0) * np.log(1.0 - pm) - pm**gamma_neg / (1.0 - pm)) * p * (1.0 - p)
        grad_neg = np.where(active_neg, grad_neg, 0.0)

        grad[:] = y * grad_pos + (1.0 - y) * grad_neg

    return _loss_fn, _grad_fn


def build_sce(lw, alpha, beta, eps, act_fn):
    if act_fn == "softmax":

        def _loss_fn(y, y_hat):
            return -alpha * np.sum(lw * y * np.log(y_hat + eps)) - beta * np.sum(lw * y_hat * np.log(y + eps))

        def _grad_fn(y, y_hat, grad):
            logy = np.log(y + eps)
            grad[:] = lw * (alpha * (y - y_hat) + beta * y_hat * (logy - np.dot(y_hat, logy)))
    else:

        def _loss_fn(y, y_hat):
            return -alpha * np.sum(lw * (y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps))) - beta * np.sum(
                lw * (y_hat * np.log(y + eps) + (1 - y_hat) * np.log(1 - y + eps))
            )

        def _grad_fn(y, y_hat, grad):
            grad[:] = lw * (alpha * (y - y_hat) + beta * np.log((y + eps) / (1.0 - y + eps)) * y_hat * (1.0 - y_hat))

    return _loss_fn, _grad_fn


def build_mse(lw, dact_fn):
    def _loss_fn(y, y_hat):
        return np.sum(lw * (y - y_hat) ** 2)

    def _grad_fn(y, y_hat, grad):
        grad[:] = 2 * lw * (y - y_hat) * dact_fn(y_hat)

    return _loss_fn, _grad_fn


def build_mae(lw, dact_fn):
    def _loss_fn(y, y_hat):
        return np.sum(lw * np.abs(y - y_hat))

    def _grad_fn(y, y_hat, grad):
        grad[:] = lw * np.sign(y - y_hat) * dact_fn(y_hat)

    return _loss_fn, _grad_fn


def build_huber(lw, delta, dact_fn):
    def _loss_fn(y, y_hat):
        r = y - y_hat
        return np.sum(lw * np.where(np.abs(r) <= delta, 0.5 * r**2, delta * (np.abs(r) - 0.5 * delta)))

    def _grad_fn(y, y_hat, grad):
        grad[:] = lw * np.clip(y - y_hat, -delta, delta) * dact_fn(y_hat)

    return _loss_fn, _grad_fn


def build_tversky(lw, alpha, beta, gamma, eps, dact_fn):
    def _loss_fn(y, y_hat):
        tp = np.dot(lw * y, y_hat)
        fp = np.dot(lw * (1.0 - y), y_hat)
        fn = np.dot(lw * y, 1.0 - y_hat)
        T = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
        return (1.0 - T) ** gamma

    def _grad_fn(y, y_hat, grad):
        tp = np.dot(lw * y, y_hat)
        fp = np.dot(lw * (1.0 - y), y_hat)
        fn = np.dot(lw * y, 1.0 - y_hat)
        N = tp + eps
        D = tp + alpha * fp + beta * fn + eps
        T = N / D
        fw = gamma * (1.0 - T) ** (gamma - 1.0)
        coeff = alpha + y * (1.0 - alpha - beta)
        grad[:] = fw * lw * (y * D - N * coeff) / D**2 * dact_fn(y_hat)

    return _loss_fn, _grad_fn
