import numpy as np

_fun_template = (
    "INLINE_FN void compute_loss(const float* RESTRICT y, const float* RESTRICT y_hat, float* RESTRICT grad, float* RESTRICT loss)"
)


def _c_float(x: float) -> str:
    return f"{float(x)!r}f"


def _c_array(values: np.ndarray) -> str:
    return ", ".join(_c_float(v) for v in values)


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
    """Base class for an output activation paired with a loss.

    A subclass sets ``act`` (``"softmax"``, ``"sigmoid"`` or ``"identity"``) and ``src``, the C
    source of ``compute_loss(y, y_hat, grad, loss)``. Given targets ``y`` and activations
    ``y_hat`` for one sample, it writes the summed loss to ``loss`` and, per class, the negative
    gradient of the loss with respect to the pre-activation to ``grad``. Either pointer can be
    NULL, then that output is skipped.
    """

    act: str
    src: str = ""


class SoftmaxCE(ActLoss):
    """Softmax with cross-entropy, for single-label classification.

    Parameters
    ----------
    eps : float, default=1e-7
        Added inside the log, must be > 0.

    weights : array-like of float, default=None
        Per-class weights, one per class. None weights every class by 1.
    """

    act = "softmax"

    def __init__(self, eps: float = 1e-7, weights: list[float] | np.ndarray | None = None):
        assert eps > 0, f"eps must be > 0, got {eps}"
        self.eps = eps
        self.weights = None if weights is None else np.asarray(weights, dtype=np.float32)
        if self.weights is None:
            self.src = _elementwise_body(
                loss_expr=f"-y[c] * logf(y_hat[c] + {_c_float(eps)})",
                grad_expr="y[c] - y_hat[c]",
            )
        else:
            self.src = f"""\
static const float _weights[CLASSES] = {{{_c_array(self.weights)}}};

{_fun_template} {{
    float loss_sum = 0.0f;
    float W = 0.0f;
    for (int c = 0; c < CLASSES; c++) {{
        if (loss)
            loss_sum += _weights[c] * y[c] * logf(y_hat[c] + {_c_float(eps)});
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
    """Sigmoid with binary cross-entropy per class, for binary and multi-label classification.

    Parameters
    ----------
    eps : float, default=1e-7
        Added inside the logs, must be > 0.

    pos_weights : array-like of float, default=None
        Per-class weights of the positive term, one per class. None weights it by 1.

    neg_weights : array-like of float, default=None
        Per-class weights of the negative term, one per class. None weights it by 1.
    """

    act = "sigmoid"

    def __init__(
        self,
        eps: float = 1e-7,
        pos_weights: list[float] | np.ndarray | None = None,
        neg_weights: list[float] | np.ndarray | None = None,
    ):
        assert eps > 0, f"eps must be > 0, got {eps}"
        self.eps = eps
        self.pos_weights = None if pos_weights is None else np.asarray(pos_weights, dtype=np.float32)
        self.neg_weights = None if neg_weights is None else np.asarray(neg_weights, dtype=np.float32)
        if self.pos_weights is not None and self.neg_weights is not None:
            assert self.pos_weights.shape == self.neg_weights.shape, (
                f"pos_weights and neg_weights must have the same length, got {len(self.pos_weights)} and {len(self.neg_weights)}"
            )

        if self.pos_weights is None and self.neg_weights is None:
            self.src = _elementwise_body(
                loss_expr=f"-(y[c] * logf(y_hat[c] + {_c_float(eps)}) + (1.0f - y[c]) * logf(1.0f - y_hat[c] + {_c_float(eps)}))",
                grad_expr="y[c] - y_hat[c]",
            )
        else:
            pos_weights = np.ones_like(self.neg_weights) if self.pos_weights is None else self.pos_weights
            neg_weights = np.ones_like(self.pos_weights) if self.neg_weights is None else self.neg_weights
            decls = f"""\
static const float _pos_weights[CLASSES] = {{{_c_array(pos_weights)}}};
static const float _neg_weights[CLASSES] = {{{_c_array(neg_weights)}}};

"""
            self.src = decls + _elementwise_body(
                loss_expr=f"-(_pos_weights[c] * y[c] * logf(y_hat[c] + {_c_float(eps)}) + _neg_weights[c] * (1.0f - y[c]) * logf(1.0f - y_hat[c] + {_c_float(eps)}))",
                grad_expr="_pos_weights[c] * y[c] * (1.0f - y_hat[c]) - _neg_weights[c] * (1.0f - y[c]) * y_hat[c]",
            )


class MSE(ActLoss):
    """Identity with squared error, for regression."""

    act = "identity"

    def __init__(self):
        self.src = _elementwise_body(
            loss_expr="(y[c] - y_hat[c]) * (y[c] - y_hat[c])",
            grad_expr="2.0f * (y[c] - y_hat[c])",
        )


class MAE(ActLoss):
    """Identity with absolute error, for regression."""

    act = "identity"

    def __init__(self):
        self.src = _elementwise_body(
            loss_expr="fabsf(y[c] - y_hat[c])",
            grad_expr="((y[c] - y_hat[c]) > 0.0f) - ((y[c] - y_hat[c]) < 0.0f)",
        )


class Huber(ActLoss):
    """Identity with Huber loss, for regression. Squared error near 0, absolute error beyond ``delta``.

    Parameters
    ----------
    delta : float, default=1.0
        Error size where the loss switches from squared to absolute, must be > 0.
    """

    act = "identity"

    def __init__(self, delta: float = 1.0):
        assert delta > 0, f"delta must be > 0, got {delta}"
        self.delta = delta
        d = _c_float(delta)
        self.src = _elementwise_body(
            loss_expr=f"(fabsf(y[c] - y_hat[c]) <= {d} ? 0.5f * (y[c] - y_hat[c]) * (y[c] - y_hat[c]) : {d} * (fabsf(y[c] - y_hat[c]) - 0.5f * {d}))",
            grad_expr=f"fminf(fmaxf(y[c] - y_hat[c], -{d}), {d})",
        )


class ASL(ActLoss):
    """Sigmoid with asymmetric loss, for multi-label classification with many negatives.

    Parameters
    ----------
    gamma_pos : float, default=0.0
        Focusing exponent of the positive term, must be >= 0.

    gamma_neg : float, default=4.0
        Focusing exponent of the negative term, must be >= 0.

    clip : float, default=0.05
        Probability margin shifted off the negatives, in ``[0, 1)``. 0 turns it off.

    eps : float, default=1e-7
        Clamps probabilities to ``[eps, 1 - eps]``, must be > 0.
    """

    act = "sigmoid"

    def __init__(self, gamma_pos: float = 0.0, gamma_neg: float = 4.0, clip: float = 0.05, eps: float = 1e-7):
        assert gamma_pos >= 0 and gamma_neg >= 0, f"gamma_pos and gamma_neg must be >= 0, got {gamma_pos} and {gamma_neg}"
        assert 0 <= clip < 1, f"clip must be in [0, 1), got {clip}"
        assert eps > 0, f"eps must be > 0, got {eps}"
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps
        gp, gn, cl, ep = _c_float(gamma_pos), _c_float(gamma_neg), _c_float(clip), _c_float(eps)
        self.src = f"""\
{_fun_template} {{
    float loss_sum = 0.0f;
    for (int c = 0; c < CLASSES; c++) {{
        float p = fminf(fmaxf(y_hat[c], {ep}), 1.0f - {ep});
        float pm = {cl} > 0.0f ? fminf(fmaxf(p - {cl}, 0.0f), 1.0f) : p;
        bool active_neg = {cl} > 0.0f ? ((p - {cl}) > 0.0f) : true;
        pm = fminf(fmaxf(pm, {ep}), 1.0f - {ep});
        float log_p = logf(p);
        float log_1mpm = logf(1.0f - pm);

        if (loss) {{
            float loss_pos = powf(1.0f - p, {gp}) * log_p;
            float loss_neg = powf(pm, {gn}) * log_1mpm;
            loss_sum += y[c] * loss_pos + (1.0f - y[c]) * loss_neg;
        }}

        if (grad) {{
            float grad_pos = powf(1.0f - p, {gp} + 1.0f) - {gp} * powf(1.0f - p, {gp}) * p * log_p;
            float grad_neg = ({gn} * powf(pm, {gn} - 1.0f) * log_1mpm - powf(pm, {gn}) / (1.0f - pm)) * p * (1.0f - p);
            grad_neg = active_neg ? grad_neg : 0.0f;
            grad[c] = y[c] * grad_pos + (1.0f - y[c]) * grad_neg;
        }}
    }}
    if (loss)
        *loss = -loss_sum;
}}
"""


class SCE(ActLoss):
    """Softmax with symmetric cross-entropy (cross-entropy plus reverse cross-entropy), for noisy labels.

    Parameters
    ----------
    alpha : float, default=1.0
        Weight of the cross-entropy term, must be >= 0.

    beta : float, default=1.0
        Weight of the reverse cross-entropy term, must be >= 0.

    eps : float, default=1e-4
        Added to the targets inside the reverse term's log, must be > 0.

    eps_hat : float, default=1e-7
        Added to the predictions inside the cross-entropy log, must be > 0.
    """

    act = "softmax"

    def __init__(self, alpha: float = 1.0, beta: float = 1.0, eps: float = 1e-4, eps_hat: float = 1e-7):
        assert alpha >= 0 and beta >= 0, f"alpha and beta must be >= 0, got {alpha} and {beta}"
        assert eps > 0 and eps_hat > 0, f"eps and eps_hat must be > 0, got {eps} and {eps_hat}"
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.eps_hat = eps_hat
        a, b, ep, eh = _c_float(alpha), _c_float(beta), _c_float(eps), _c_float(eps_hat)
        self.src = f"""\
{_fun_template} {{
    float term_ce = 0.0f, term_rce = 0.0f;
    for (int c = 0; c < CLASSES; c++) {{
        term_ce += y[c] * logf(y_hat[c] + {eh});
        term_rce += y_hat[c] * logf(y[c] + {ep});
    }}
    if (loss)
        *loss = -{a} * term_ce - {b} * term_rce;
    if (grad) {{
        for (int c = 0; c < CLASSES; c++)
            grad[c] = {a} * (y[c] - y_hat[c]) + {b} * y_hat[c] * (logf(y[c] + {ep}) - term_rce);
    }}
}}
"""


class Tversky(ActLoss):
    """Sigmoid with Tversky loss over the classes of a sample, for multi-label classification.

    Parameters
    ----------
    alpha : float, default=0.5
        Weight of false positives, must be >= 0.

    beta : float, default=0.5
        Weight of false negatives, must be >= 0.

    eps : float, default=1e-7
        Added to the numerator and denominator, must be > 0.
    """

    act = "sigmoid"

    def __init__(self, alpha: float = 0.5, beta: float = 0.5, eps: float = 1e-7):
        assert alpha >= 0 and beta >= 0, f"alpha and beta must be >= 0, got {alpha} and {beta}"
        assert eps > 0, f"eps must be > 0, got {eps}"
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        a, b, ep = _c_float(alpha), _c_float(beta), _c_float(eps)
        self.src = f"""\
{_fun_template} {{
    float tp = 0.0f, fp = 0.0f, fn_ = 0.0f;
    for (int c = 0; c < CLASSES; c++) {{
        tp += y[c] * y_hat[c];
        fp += (1.0f - y[c]) * y_hat[c];
        fn_ += y[c] * (1.0f - y_hat[c]);
    }}
    float N = tp + {ep};
    float D = tp + {a} * fp + {b} * fn_ + {ep};

    if (loss)
        *loss = 1.0f - N / D;
    if (grad) {{
        for (int c = 0; c < CLASSES; c++) {{
            float coeff = {a} + y[c] * (1.0f - {a} - {b});
            grad[c] = (y[c] * D - N * coeff) / (D * D) * y_hat[c] * (1.0f - y_hat[c]);
        }}
    }}
}}
"""


class FocalBCE(ActLoss):
    """Sigmoid with focal binary cross-entropy per class, for imbalanced binary and multi-label classification.

    Parameters
    ----------
    alpha : float, default=0.25
        Weight of the positive term, in ``[0, 1]``. The negative term gets ``1 - alpha``.

    gamma : float, default=2.0
        Focusing exponent, must be >= 0. 0 gives weighted binary cross-entropy.

    eps : float, default=1e-7
        Added inside the logs, must be > 0.
    """

    act = "sigmoid"

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, eps: float = 1e-7):
        assert 0 <= alpha <= 1, f"alpha must be in [0, 1], got {alpha}"
        assert gamma >= 0, f"gamma must be >= 0, got {gamma}"
        assert eps > 0, f"eps must be > 0, got {eps}"
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        a, g, ep = _c_float(alpha), _c_float(gamma), _c_float(eps)
        self.src = _elementwise_body(
            loss_expr=(
                f"-(y[c] * {a} * powf(1.0f - y_hat[c], {g}) * logf(y_hat[c] + {ep}) "
                f"+ (1.0f - y[c]) * (1.0f - {a}) * powf(y_hat[c], {g}) * logf(1.0f - y_hat[c] + {ep}))"
            ),
            grad_expr=(
                f"y[c] * {a} * (powf(1.0f - y_hat[c], {g} + 1.0f) - {g} * powf(1.0f - y_hat[c], {g}) * y_hat[c] * logf(y_hat[c] + {ep})) "
                f"+ (1.0f - y[c]) * (1.0f - {a}) * ({g} * powf(y_hat[c], {g}) * (1.0f - y_hat[c]) * logf(1.0f - y_hat[c] + {ep}) - powf(y_hat[c], {g} + 1.0f))"
            ),
        )


class FocalCE(ActLoss):
    """Softmax with focal cross-entropy, for imbalanced single-label classification.

    Parameters
    ----------
    alpha : float, default=1.0
        Overall weight, must be >= 0.

    gamma : float, default=2.0
        Focusing exponent, must be >= 0. 0 gives cross-entropy.

    eps : float, default=1e-7
        Added inside the log and floors ``1 - p``, must be > 0.
    """

    act = "softmax"

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, eps: float = 1e-7):
        assert alpha >= 0, f"alpha must be >= 0, got {alpha}"
        assert gamma >= 0, f"gamma must be >= 0, got {gamma}"
        assert eps > 0, f"eps must be > 0, got {eps}"
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        a, gm, ep = _c_float(alpha), _c_float(gamma), _c_float(eps)
        self.src = f"""\
{_fun_template} {{
    float loss_sum = 0.0f;
    float g[CLASSES];
    float sum = 0.0f;
    for (int c = 0; c < CLASSES; c++) {{
        float p = y_hat[c];
        float q = fmaxf(1.0f - p, {ep});
        if (loss)
            loss_sum += {a} * y[c] * powf(q, {gm}) * logf(p + {ep});
        if (grad) {{
            g[c] = {a} * y[c] * ({gm} * powf(q, {gm} - 1.0f) * logf(p + {ep}) - powf(q, {gm}) / (p + {ep}));
            sum += g[c] * p;
        }}
    }}
    if (loss)
        *loss = -loss_sum;
    if (grad) {{
        for (int c = 0; c < CLASSES; c++)
            grad[c] = y_hat[c] * (sum - g[c]);
    }}
}}
"""
