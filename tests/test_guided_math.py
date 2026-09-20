"""Guided activations and losses against torch / scipy references and autograd."""

import numpy as np
import pytest

from .support import activation, guided, loss_and_grad, set_clauses, set_weights

torch = pytest.importorskip("torch")
F = torch.nn.functional
scipy_special = pytest.importorskip("scipy.special")

from ytm._guided.backends.act_loss import ASL, MAE, MSE, SCE, FocalBCE, FocalCE, Huber, SigmoidBCE, SoftmaxCE, Tversky

C = 4
Z = np.array([-3.5, -1.0, 0.5, 2.0])


def test_softmax_activation_matches_scipy(device):
    """Votes are divided by NORM before softmax; forgetting it (or dividing twice) shifts every probability."""
    tm = guided("multi", device, n_classes=C, act_loss=SoftmaxCE())
    votes = Z * 2.0  # NORM = n_clauses / 2 = 2 -> z = Z
    assert np.allclose(activation(tm, votes), scipy_special.softmax(Z), atol=1e-6)


def test_sigmoid_activation_matches_scipy(device):
    """Sigmoid activation must equal expit(votes / NORM); a wrong NORM or a softmax mix-up changes every probability."""
    tm = guided("multi", device, n_classes=C, act_loss=SigmoidBCE())
    assert np.allclose(activation(tm, Z * 2.0), scipy_special.expit(Z), atol=1e-6)


def test_identity_activation_is_votes_over_norm(device):
    """Identity activation must still divide by NORM; skipping it makes MSE/MAE/Huber act on unnormalised votes."""
    tm = guided("multi", device, n_classes=C, act_loss=MSE())
    assert np.allclose(activation(tm, Z * 2.0), Z, atol=1e-6)


def test_model_score_applies_the_activation_and_raw_votes_does_not(device):
    """`score` returns y_hat, `raw_votes` the pre-activation sum (guided-raw-votes-decision)."""
    tm = guided("multi", device, n_classes=C, act_loss=SoftmaxCE())
    set_clauses(tm, {})  # all clauses empty, fire everywhere
    w = np.zeros((C, 4), dtype=np.float32)
    w[:, 0] = Z * 2.0
    set_weights(tm, w)
    X = np.zeros((1, 4), dtype=int)
    assert np.allclose(tm.raw_votes(X, force_repack=True), [Z * 2.0], atol=1e-6)
    assert np.allclose(tm.score(X, force_repack=True), [scipy_special.softmax(Z)], atol=1e-6)


def _t(a):
    return torch.tensor(np.asarray(a), dtype=torch.float64)


def _act(name, z):
    return {"softmax": lambda: torch.softmax(z, 0), "sigmoid": lambda: torch.sigmoid(z), "identity": lambda: z}[name]()


# own transcriptions of act_loss.py as written (no library reference exists): consistency of loss and gradient only
def _asl(z, y, gp=0.0, gn=4.0, clip=0.05, eps=1e-7):
    p = torch.sigmoid(z)
    pm = torch.clamp(p - clip, min=0.0)
    return -(y * (1 - p) ** gp * torch.log(p.clamp(min=eps)) + (1 - y) * pm**gn * torch.log((1 - pm).clamp(min=eps))).sum()


def _sce(z, y, a=1.0, b=1.0, eps=1e-4, eps_hat=1e-7):
    p = torch.softmax(z, 0)
    return a * -(y * torch.log(p + eps_hat)).sum() + b * -(p * torch.log(y + eps)).sum()


def _tversky(z, y, a=0.5, b=0.5, eps=1e-7):
    p = torch.sigmoid(z)
    tp, fp, fn = (y * p).sum(), ((1 - y) * p).sum(), (y * (1 - p)).sum()
    return 1 - (tp + eps) / (tp + a * fp + b * fn + eps)


def _focal_ce(z, y, alpha=1.0, gamma=2.0, eps=1e-7):
    p = torch.softmax(z, 0)
    return -(alpha * y * (1 - p) ** gamma * torch.log(p + eps)).sum()


def _bce_pos_neg(z, y, pw, nw):
    return -(pw * y * F.logsigmoid(z) + nw * (1 - y) * F.logsigmoid(-z)).sum()


PW = np.array([1.0, 2.0, 3.0, 0.5])
NW = np.array([0.5, 1.5, 1.0, 2.0])
ONE_HOT = np.array([0.0, 0.0, 1.0, 0.0])
MULTI = np.array([1.0, 0.0, 1.0, 0.0])
REAL = np.array([0.3, -0.7, 1.2, 0.0])

# id, loss, act, target, torch loss of (z, y), library-backed?
CASES = [
    ("softmax_ce", SoftmaxCE(), "softmax", ONE_HOT, lambda z, y: F.cross_entropy(z[None], y[None], reduction="sum"), True),
    (
        "softmax_ce_weighted",
        SoftmaxCE(weights=PW),
        "softmax",
        ONE_HOT,
        lambda z, y: F.cross_entropy(z[None], y[None], weight=_t(PW), reduction="sum"),
        True,
    ),
    ("sigmoid_bce", SigmoidBCE(), "sigmoid", MULTI, lambda z, y: F.binary_cross_entropy_with_logits(z, y, reduction="sum"), True),
    (
        "sigmoid_bce_pos",
        SigmoidBCE(pos_weights=PW),
        "sigmoid",
        MULTI,
        lambda z, y: F.binary_cross_entropy_with_logits(z, y, pos_weight=_t(PW), reduction="sum"),
        True,
    ),
    (
        "sigmoid_bce_pos_neg",
        SigmoidBCE(pos_weights=PW, neg_weights=NW),
        "sigmoid",
        MULTI,
        lambda z, y: _bce_pos_neg(z, y, _t(PW), _t(NW)),
        False,
    ),
    ("mse", MSE(), "identity", REAL, lambda z, y: F.mse_loss(z, y, reduction="sum"), True),
    ("mae", MAE(), "identity", REAL, lambda z, y: F.l1_loss(z, y, reduction="sum"), True),
    ("huber", Huber(delta=0.5), "identity", REAL, lambda z, y: F.huber_loss(z, y, reduction="sum", delta=0.5), True),
    ("focal_bce", FocalBCE(alpha=0.25, gamma=2.0), "sigmoid", MULTI, None, True),  # torchvision, see below
    ("asl", ASL(), "sigmoid", MULTI, _asl, False),
    ("sce", SCE(), "softmax", ONE_HOT, _sce, False),
    ("tversky", Tversky(), "sigmoid", MULTI, _tversky, False),
    ("focal_ce", FocalCE(), "softmax", ONE_HOT, _focal_ce, False),  # red: act_loss.py:261 `float S` collides with the S macro
]


def _focal_bce_ref(z, y):
    tvo = pytest.importorskip("torchvision.ops")
    return tvo.sigmoid_focal_loss(z, y, alpha=0.25, gamma=2.0, reduction="sum")


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_loss_value_and_gradient_match_reference(device, case):
    """Loss equals the reference loss; grad equals -dL/dz by autograd. Library-backed cases are marked in CASES; the rest
    only prove the hand-derived gradient is consistent with the loss as written, not that the loss matches its paper."""
    _name, loss, act, y, ref, _lib = case
    ref = ref or _focal_bce_ref
    tm = guided("multi", device, n_classes=C, act_loss=loss)
    z = _t(Z).requires_grad_(True)
    yt = _t(y)
    ref_loss = ref(z, yt)
    (ref_grad,) = torch.autograd.grad(ref_loss, z)

    y_hat = _act(act, z).detach().numpy()
    got_loss, got_grad = loss_and_grad(tm, y_hat, y)

    assert got_loss == pytest.approx(float(ref_loss.detach()), rel=2e-4, abs=1e-5)
    assert np.allclose(got_grad, -ref_grad.numpy(), rtol=2e-3, atol=2e-5)


def test_huber_loss_matches_scipy():
    """scipy.special.huber is an independent implementation of the same piecewise loss."""
    tm = guided("multi", "cpu:1", n_classes=C, act_loss=Huber(delta=0.5))
    loss, _ = loss_and_grad(tm, Z, REAL)
    assert loss == pytest.approx(float(scipy_special.huber(0.5, REAL - Z).sum()), rel=1e-5)


def test_saturated_bce_loss_is_bounded_by_eps_and_the_gradient_is_still_exact(device):
    """Torch's logits BCE grows linearly in |z|; ours takes log(y_hat + eps), so it caps at -log(eps) per class.
    The gradient y - y_hat never touches the log and stays exact."""
    tm = guided("multi", device, n_classes=2, act_loss=SigmoidBCE())
    loss, grad = loss_and_grad(tm, [0.0, 1.0], [1.0, 0.0])
    assert loss == pytest.approx(2 * -np.log(1e-7), rel=1e-3)
    assert np.array_equal(grad, [1.0, -1.0])


# The FocalCE cases below are red because act_loss.py:261 declares a local `float S`, which collides with the config macro
# `#define S <s>f` (config.py:221), so the generated C source does not compile.
# Once that is fixed, FocalCE(gamma=0.5) at the `split` votes pattern (EXTREME_VOTES[1]) is still expected to stay red:
# powf(1 - p, gamma - 1) = powf(0, -0.5) = inf, then 0 * inf = NaN in g[c] poisons S += g[c] * p.
ALL_LOSSES = [
    pytest.param(SoftmaxCE(), ONE_HOT, id="softmax_ce"),
    pytest.param(SoftmaxCE(weights=PW), ONE_HOT, id="softmax_ce_weighted"),
    pytest.param(SigmoidBCE(), MULTI, id="sigmoid_bce"),
    pytest.param(SigmoidBCE(pos_weights=PW, neg_weights=NW), MULTI, id="sigmoid_bce_pos_neg"),
    pytest.param(MSE(), REAL, id="mse"),
    pytest.param(MAE(), REAL, id="mae"),
    pytest.param(Huber(), REAL, id="huber"),
    pytest.param(ASL(), MULTI, id="asl"),
    pytest.param(SCE(), ONE_HOT, id="sce"),
    pytest.param(Tversky(), MULTI, id="tversky"),
    pytest.param(FocalBCE(), MULTI, id="focal_bce"),
    pytest.param(FocalCE(), ONE_HOT, id="focal_ce"),
    pytest.param(FocalCE(gamma=0.5), ONE_HOT, id="focal_ce_gamma0.5"),
    pytest.param(ASL(gamma_pos=0.5), MULTI, id="asl_gamma_pos0.5"),
]
EXTREME_VOTES = [
    np.zeros(C),
    np.array([1e13, -1e13, 0.0, 0.0]),
    np.full(C, -1e13),
    np.full(C, 1e13),
    np.array([1e13, 1e13, -1e13, -1e13]),
]


@pytest.mark.parametrize("loss, y", ALL_LOSSES)
@pytest.mark.parametrize("votes", EXTREME_VOTES, ids=["zero", "split", "all_neg", "all_pos", "mixed"])
def test_activation_loss_and_gradient_stay_finite_for_finite_votes(device, loss, y, votes):
    """Votes reach about max_weight * n_clauses (1e13 at defaults). Every act x loss must stay finite there
    (softmax max-shift, expf overflow, powf(0, gamma-1), 1/(D*D)). NaN or inf here would silently poison every weight."""
    tm = guided("multi", device, n_classes=C, act_loss=loss)
    y_hat = activation(tm, votes)
    got_loss, grad = loss_and_grad(tm, y_hat, y)
    assert np.all(np.isfinite(y_hat)) and np.isfinite(got_loss) and np.all(np.isfinite(grad))
