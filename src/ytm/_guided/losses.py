import numpy as np

_fun_template = (
    "INLINE_FN void compute_loss(const float* restrict y, const float* restrict y_hat, float* restrict grad, float* restrict loss)"
)


def _elementwise_body(loss_expr: str, grad_expr: str) -> str:
    return f"""\
{_fun_template} {{
    if (loss) {{
        float loss_sum = 0.0f;
        for (int c = 0; c < CLASSES; c++)
            loss_sum += {loss_expr};
        *loss = loss_sum;
    }}
    if (grad) {{
        for (int c = 0; c < CLASSES; c++)
            grad[c] = {grad_expr};
    }}
}}
"""


class ActLoss:
    act: str
    _src: str = ""

    @property
    def src(self) -> str:
        return self._src

    @src.setter
    def src(self, value: str) -> None:
        self._src = value


class SoftmaxCE(ActLoss):
    act = "softmax"

    def __init__(self, eps: float = 1e-7, weights: list[float] | np.ndarray | None = None):
        self.eps = eps
        self.weights = None if weights is None else np.asarray(weights, dtype=np.float32)
        if self.weights is None:
            self.src = _elementwise_body(
                loss_expr=f"-y[c] * logf(y_hat[c] + {eps}f)",
                grad_expr="y[c] - y_hat[c]",
            )
        else:
            weights_init = ", ".join(f"{w}f" for w in self.weights)
            self.src = f"""\
static const float _weights[CLASSES] = {{{weights_init}}};

{_fun_template} {{
    float loss_sum = 0.0f;
    float W = 0.0f;
    for (int c = 0; c < CLASSES; c++) {{
        if (loss)
            loss_sum += _weights[c] * y[c] * logf(y_hat[c] + {eps}f);
        if (grad)
            W += _weights[c] * y[c];
    }}
    if (loss)
        *loss = -loss_sum;
    if (grad) {{
        for (int c = 0; c < CLASSES; c++)
            grad[c] = _weights[c] * y[c] - y_hat[c] * W;
    }}
}}
"""


class SigmoidBCE(ActLoss):
    act = "sigmoid"

    def __init__(self, eps: float = 1e-7, weights: list[float] | np.ndarray | None = None):
        self.eps = eps
        self.weights = None if weights is None else np.asarray(weights, dtype=np.float32)
        if self.weights is None:
            self.src = _elementwise_body(
                loss_expr=f"-(y[c] * logf(y_hat[c] + {eps}f) + (1.0f - y[c]) * logf(1.0f - y_hat[c] + {eps}f))",
                grad_expr="y[c] - y_hat[c]",
            )
        else:
            weights_init = ", ".join(f"{w}f" for w in self.weights)
            self.src = f"static const float _weights[CLASSES] = {{{weights_init}}};\n\n" + _elementwise_body(
                loss_expr=f"-_weights[c] * (y[c] * logf(y_hat[c] + {eps}f) + (1.0f - y[c]) * logf(1.0f - y_hat[c] + {eps}f))",
                grad_expr="_weights[c] * (y[c] - y_hat[c])",
            )


class MSE(ActLoss):
    act = "identity"

    def __init__(self):
        self.src = _elementwise_body(
            loss_expr="(y[c] - y_hat[c]) * (y[c] - y_hat[c])",
            grad_expr="2.0f * (y[c] - y_hat[c])",
        )


class MAE(ActLoss):
    act = "identity"

    def __init__(self):
        self.src = _elementwise_body(
            loss_expr="fabsf(y[c] - y_hat[c])",
            grad_expr="((y[c] - y_hat[c]) > 0.0f) - ((y[c] - y_hat[c]) < 0.0f)",
        )


class Huber(ActLoss):
    act = "identity"

    def __init__(self, delta: float = 1.0):
        self.delta = delta
        self.src = _elementwise_body(
            loss_expr=f"(fabsf(y[c] - y_hat[c]) <= {delta}f ? 0.5f * (y[c] - y_hat[c]) * (y[c] - y_hat[c]) : {delta}f * (fabsf(y[c] - y_hat[c]) - 0.5f * {delta}f))",
            grad_expr=f"fminf(fmaxf(y[c] - y_hat[c], -{delta}f), {delta}f)",
        )


class ASL(ActLoss):
    act = "sigmoid"

    def __init__(self, gamma_pos: float = 0.0, gamma_neg: float = 4.0, clip: float = 0.05, eps: float = 1e-7):
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps
        self.src = f"""\
{_fun_template} {{
    float loss_sum = 0.0f;
    for (int c = 0; c < CLASSES; c++) {{
        float p = fminf(fmaxf(y_hat[c], {eps}f), 1.0f - {eps}f);
        float pm = {clip}f > 0.0f ? fminf(fmaxf(p - {clip}f, 0.0f), 1.0f) : p;
        bool active_neg = {clip}f > 0.0f ? ((p - {clip}f) > 0.0f) : true;
        pm = fminf(fmaxf(pm, {eps}f), 1.0f - {eps}f);
        float log_p = logf(p);
        float log_1mpm = logf(1.0f - pm);

        if (loss) {{
            float loss_pos = powf(1.0f - p, {gamma_pos}f) * log_p;
            float loss_neg = powf(pm, {gamma_neg}f) * log_1mpm;
            loss_sum += y[c] * loss_pos + (1.0f - y[c]) * loss_neg;
        }}

        if (grad) {{
            float grad_pos = powf(1.0f - p, {gamma_pos}f + 1.0f) - {gamma_pos}f * powf(1.0f - p, {gamma_pos}f) * p * log_p;
            float grad_neg = ({gamma_neg}f * powf(pm, {gamma_neg}f - 1.0f) * log_1mpm - powf(pm, {gamma_neg}f) / (1.0f - pm)) * p * (1.0f - p);
            grad_neg = active_neg ? grad_neg : 0.0f;
            grad[c] = y[c] * grad_pos + (1.0f - y[c]) * grad_neg;
        }}
    }}
    if (loss)
        *loss = -loss_sum;
}}
"""


class SCE(ActLoss):
    act = "softmax"

    def __init__(self, alpha: float = 1.0, beta: float = 1.0, eps: float = 1e-4, eps_hat: float = 1e-7):
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.eps_hat = eps_hat
        self.src = f"""\
{_fun_template} {{
    float term_ce = 0.0f, term_rce = 0.0f;
    for (int c = 0; c < CLASSES; c++) {{
        term_ce += y[c] * logf(y_hat[c] + {eps_hat}f);
        term_rce += y_hat[c] * logf(y[c] + {eps}f);
    }}
    if (loss)
        *loss = -{alpha}f * term_ce - {beta}f * term_rce;
    if (grad) {{
        for (int c = 0; c < CLASSES; c++)
            grad[c] = {alpha}f * (y[c] - y_hat[c]) + {beta}f * y_hat[c] * (logf(y[c] + {eps}f) - term_rce);
    }}
}}
"""


class Tversky(ActLoss):
    act = "sigmoid"

    def __init__(self, alpha: float = 0.5, beta: float = 0.5, eps: float = 1e-7):
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.src = f"""\
{_fun_template} {{
    float tp = 0.0f, fp = 0.0f, fn_ = 0.0f;
    for (int c = 0; c < CLASSES; c++) {{
        tp += y[c] * y_hat[c];
        fp += (1.0f - y[c]) * y_hat[c];
        fn_ += y[c] * (1.0f - y_hat[c]);
    }}
    float N = tp + {eps}f;
    float D = tp + {alpha}f * fp + {beta}f * fn_ + {eps}f;

    if (loss)
        *loss = 1.0f - N / D;
    if (grad) {{
        for (int c = 0; c < CLASSES; c++) {{
            float coeff = {alpha}f + y[c] * (1.0f - {alpha}f - {beta}f);
            grad[c] = (y[c] * D - N * coeff) / (D * D) * y_hat[c] * (1.0f - y_hat[c]);
        }}
    }}
}}
"""


class FocalBCE(ActLoss):
    act = "sigmoid"

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, eps: float = 1e-7):
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.src = _elementwise_body(
            loss_expr=(
                f"-(y[c] * {alpha}f * powf(1.0f - y_hat[c], {gamma}f) * logf(y_hat[c] + {eps}f) "
                f"+ (1.0f - y[c]) * (1.0f - {alpha}f) * powf(y_hat[c], {gamma}f) * logf(1.0f - y_hat[c] + {eps}f))"
            ),
            grad_expr=(
                f"y[c] * {alpha}f * (powf(1.0f - y_hat[c], {gamma}f + 1.0f) - {gamma}f * powf(1.0f - y_hat[c], {gamma}f) * y_hat[c] * logf(y_hat[c] + {eps}f)) "
                f"+ (1.0f - y[c]) * (1.0f - {alpha}f) * ({gamma}f * powf(y_hat[c], {gamma}f) * (1.0f - y_hat[c]) * logf(1.0f - y_hat[c] + {eps}f) - powf(y_hat[c], {gamma}f + 1.0f))"
            ),
        )


class FocalCE(ActLoss):
    act = "softmax"

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, eps: float = 1e-7):
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.src = f"""\
{_fun_template} {{
    float loss_sum = 0.0f;
    float g[CLASSES];
    float S = 0.0f;
    for (int c = 0; c < CLASSES; c++) {{
        float p = y_hat[c];
        if (loss)
            loss_sum += {alpha}f * y[c] * powf(1.0f - p, {gamma}f) * logf(p + {eps}f);
        if (grad) {{
            g[c] = {alpha}f * y[c] * ({gamma}f * powf(1.0f - p, {gamma}f - 1.0f) * logf(p + {eps}f) - powf(1.0f - p, {gamma}f) / (p + {eps}f));
            S += g[c] * p;
        }}
    }}
    if (loss)
        *loss = -loss_sum;
    if (grad) {{
        for (int c = 0; c < CLASSES; c++)
            grad[c] = y_hat[c] * (S - g[c]);
    }}
}}
"""
