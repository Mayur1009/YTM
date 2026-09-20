import numpy as np

from .support import FB_T1A, FB_T1B, FB_T2, discrete, fill, guided, host, make_buffers, set_weights, update_weights

X1 = [[0, 0, 0, 0]]


def _discrete_update(device, weights, fb, **kw):
    kw.setdefault("allow_polarity_change", True)  # discrete BinaryTM defaults it to False
    tm = discrete("binary", device, n_clauses=len(weights), **kw)
    set_weights(tm, [weights])
    buf = make_buffers(tm, X1, [[0.0]])
    update_weights(tm, buf, np.array(fb).reshape(-1, 1))
    return list(host(tm, tm.dev.clause_weights)[0])


def test_discrete_weight_update_rules(device):
    """Rule for allow_polarity_change=True (forced by the helper): T2 flips polarity when |w - sign| < 1, T1A grows |w|, T1B is a no-op."""
    # Weights [1, 3, -1, 2, -2, 4].
    out = _discrete_update(device, [1, 3, -1, 2, -2, 4], [FB_T2, FB_T2, FB_T2, FB_T1A, FB_T1A, FB_T1B])
    assert out == [-1, 2, 1, 3, -3, 4]


def test_discrete_weight_never_reaches_max_weight_through_type1a(device):
    """`|nw| < MAX_WEIGHT` uses a strict bound; an off-by-one would let weights reach MAX_WEIGHT."""
    assert _discrete_update(device, [4, -4], [FB_T1A, FB_T1A], max_weight=5.0) == [4, -4]
    assert _discrete_update(device, [3, -3], [FB_T1A, FB_T1A], max_weight=5.0) == [4, -4]


def test_discrete_type2_keeps_polarity_when_polarity_change_is_off(device):
    """allow_polarity_change=False: w=+1 under T2 goes to 0, which must be pushed back to +1, and -1 back to -1."""
    assert _discrete_update(device, [1, -1, 3], [FB_T2, FB_T2, FB_T2], allow_polarity_change=False) == [1, -1, 2]


def test_discrete_positive_only_weights_clip_at_one(device):
    """negative_clauses=False uses `clip(nw, 1, MAX)`: a T2 on w=1 stays at 1, never 0 or negative."""
    assert _discrete_update(device, [1, 3], [FB_T2, FB_T2], negative_clauses=False) == [1, 2]


def _guided_update(device, weights, output, grad, **kw):
    tm = guided("multi", device, n_classes=2, lr=0.5, fb_signal="grad", **kw)
    set_weights(tm, weights)
    buf = make_buffers(tm, [[0, 0, 0, 0]], [[1.0, 0.0]])
    fill(tm, buf.clause_output, output)
    fill(tm, buf.grad, grad)
    update_weights(tm, buf)
    return host(tm, tm.dev.clause_weights)


W = [[1, 1, -1, 4], [1, 1, -1, -4]]


def test_guided_weight_update_is_lr_times_grad_on_firing_clauses_only(device):
    """Clause 1 does not fire and must be untouched; the others move by lr*grad = [+1.0, -0.5] per class."""
    out = _guided_update(device, W, [1, 0, 1, 1], [2.0, -1.0])
    assert np.allclose(out, [[2, 1, 0, 5], [0.5, 1, -1.5, -4.5]])


def test_guided_weight_update_keeps_polarity_when_change_is_off(device):
    """allow_polarity_change=False clamps a weight that would cross zero to +-0.0001 instead of flipping it."""
    out = _guided_update(device, W, [1, 0, 1, 1], [2.0, -1.0], allow_polarity_change=False)
    assert np.allclose(out, [[2, 1, -0.0001, 5], [0.5, 1, -1.5, -4.5]], atol=1e-7)


def test_guided_weight_update_clips_at_max_weight(device):
    """A huge lr*grad must land exactly on +-max_weight, not overflow or wrap."""
    out = _guided_update(device, W, [1, 1, 1, 1], [1e9, -1e9], max_weight=10.0)
    assert np.allclose(out, [[10, 10, 10, 10], [-10, -10, -10, -10]])
