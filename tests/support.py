"""Hand-set state and one-stage fit driver, identical on cpu and cuda.

A test overwrites `ta_states` / `clause_weights` with values it derived by hand, pushes one sample
through a single stage of the fit path, and reads the result back. It only touches device methods
that already exist.
"""

import contextlib
from ctypes import POINTER, c_float

import numpy as np

from ytm._core.utils import Feedback

FB_NONE, FB_T1A, FB_T1B, FB_T2 = (int(f) for f in (Feedback.NONE, Feedback.T1A, Feedback.T1B, Feedback.T2))

_float_p = POINTER(c_float)


def is_cuda(tm) -> bool:
    return tm.device_config.kind == "cuda"


def _ctx(tm):
    return tm.dev.cuda_dev if is_cuda(tm) else contextlib.nullcontext()


def host(tm, arr) -> np.ndarray:
    return np.array(tm.dev._to_host(arr))


def discrete(kind: str, device: str = "cpu:1", **kw):
    from ytm.discrete import BinaryTM, MultiClassTM

    cfg: dict = {"n_clauses": 4, "T": 10.0, "s": 1.0, "dim": (4, 1, 1), "seed": 1}
    if kind == "multi":
        cfg["n_classes"] = 2
    cfg.update(kw)
    return (BinaryTM if kind == "binary" else MultiClassTM)(**cfg, device=device)


def guided(kind: str, device: str = "cpu:1", **kw):
    from ytm.guided import BinaryTM, MultiClassTM

    cfg: dict = {"n_clauses": 4, "s": 1.0, "dim": (4, 1, 1), "seed": 1}
    if kind == "multi":
        cfg["n_classes"] = 3
    cfg.update(kw)
    return (BinaryTM if kind == "binary" else MultiClassTM)(**cfg, device=device)


def fill(tm, arr, values) -> None:
    """Overwrite a device array in place with `values`, whatever the backend."""
    with _ctx(tm):
        arr[...] = tm.dev.xp.asarray(np.asarray(values), dtype=arr.dtype).reshape(arr.shape)


def set_ta_states(tm, states) -> None:
    fill(tm, tm.dev.ta_states, states)
    with _ctx(tm):
        tm.dev.packed_clauses.is_clause_synced.fill(0)


def set_clauses(tm, includes_by_clause: dict[int, list[int]], state: int | None = None) -> None:
    """Every literal excluded, except `includes_by_clause[c]` which sit exactly at `include_state`.

    `state` overrides the excluded value (default `include_state - 1`).
    """
    cfg = tm.config
    states = np.full((cfg._total_clauses, cfg._n_literals), cfg._include_state - 1 if state is None else state)
    for clause, lits in includes_by_clause.items():
        states[clause, lits] = cfg._include_state
    set_ta_states(tm, states)


def set_weights(tm, weights) -> None:
    fill(tm, tm.dev.clause_weights, weights)


def make_buffers(tm, X, Y):
    """Fit buffers for `X` (raw, will be prepared) and `Y` (one row per sample, already encoded)."""
    dev, cfg = tm.dev, tm.config
    Xp = tm._prepare_X(np.asarray(X))
    Yf = np.ascontiguousarray(Y, dtype=np.float32)
    mask = np.zeros(cfg._total_clauses, dtype=np.int8)
    guided_model = hasattr(cfg, "lr")

    if not is_cuda(tm):
        if guided_model:
            return dev._fit_allocs(Xp, Yf, mask, lr=cfg.lr, lambda_=cfg.lambda_)
        return dev._fit_allocs(Xp, Yf, mask, np.ones((Xp.shape[0], cfg.n_classes), dtype=np.float32))

    import cupy as cp

    with _ctx(tm):
        if guided_model:
            buf = dev._fit_allocs(cp.asarray(mask), lr=cfg.lr, lambda_=cfg.lambda_)
        else:
            buf = dev._fit_allocs(cp.asarray(mask))
            buf.label_probs = cp.ones((Xp.shape[0], cfg.n_classes), dtype=cp.float32)
        buf.X = cp.asarray(Xp, dtype=cfg._fbound_dtype)
        buf.Y = cp.asarray(Yf)
    return buf


def fb_ids_from_dense(tm, fb) -> np.ndarray:
    """Global clause ids with any feedback in a dense `feedback_type`, sorted: the list `decide` builds, in order."""
    cfg = tm.config
    has_fb = np.asarray(fb) != FB_NONE
    if has_fb.ndim == 1:  # guided delta_l: one entry per global clause
        ids = np.flatnonzero(has_fb)
    elif cfg._total_clauses == cfg._n_clauses:  # coalesced: row is the clause, columns are classes
        ids = np.flatnonzero(has_fb.any(axis=1))
    else:  # one bank per class: row is the clause within its bank, column is the bank
        rel, cls = np.nonzero(has_fb)
        ids = cls * cfg._n_clauses + rel
    return np.sort(ids).astype(np.uint32)


def fb_counts_from_dense(tm, fb, fb_type) -> np.ndarray:
    """Per global clause, how many classes gave `fb_type` in a dense `feedback_type`."""
    cfg = tm.config
    is_type = np.asarray(fb) == fb_type
    if is_type.ndim == 1:  # guided delta_l: one entry per global clause
        return is_type.astype(np.int64)
    if cfg._total_clauses == cfg._n_clauses:  # coalesced: row is the clause, columns are classes
        return is_type.sum(axis=1)
    return is_type.T.reshape(-1).astype(np.int64)  # one bank per class: global id is bank * n_clauses + row


def fill_fb_list(tm, buf, ids, fb) -> None:
    """Fill the feedback lists with clauses `ids`, their counts taken from the dense host array `fb`."""
    ids = np.asarray(ids, dtype=np.uint32)
    fb = np.asarray(fb)
    # Split into the T1b list and the T1a/T2 list, each with per clause counts over classes.
    n1a, n1b, n2 = (fb_counts_from_dense(tm, fb, t)[ids] for t in (FB_T1A, FB_T1B, FB_T2))
    t1b = n1b > 0
    fill(tm, buf.t1b_cid[: t1b.sum()], ids[t1b])
    fill(tm, buf.t1b_cnt[: t1b.sum()], n1b[t1b])
    fill(tm, buf.t1b_n, [t1b.sum()])
    t12 = (n1a + n2) > 0
    fill(tm, buf.t1a_t2_cid[: t12.sum()], ids[t12])
    fill(tm, buf.t1a_cnt[: t12.sum()], n1a[t12])
    fill(tm, buf.t2_cnt[: t12.sum()], n2[t12])
    fill(tm, buf.t1a_t2_n, [t12.sum()])


def _set_feedback(tm, buf, fb, ids=None) -> None:
    """Put a dense feedback array on the device: the per class types and the lists."""
    fill(tm, buf.feedback_type, fb)
    fb = host(tm, buf.feedback_type)
    fill_fb_list(tm, buf, fb_ids_from_dense(tm, fb) if ids is None else ids, fb)


def apply_feedback(tm, buf, fb, e: int = 0, key: int = 1, ids=None) -> None:
    """Pack, evaluate (so patch selection and clause outputs exist), then apply the given feedback.

    The feedback list comes from `fb` unless `ids` overrides it.
    """
    dev = tm.dev
    with _ctx(tm):
        dev.pack_clauses(force_repack=True)
        dev._fit_eval(buf, e, key)
        _set_feedback(tm, buf, fb, ids)
        dev._fit_apply_fb(buf, e, key)


def decide_feedback(tm, buf, votes, e: int = 0, key: int = 1) -> np.ndarray:
    """Evaluate, overwrite the votes with `votes`, let the device decide feedback, return it."""
    dev = tm.dev
    with _ctx(tm):
        dev.pack_clauses(force_repack=True)
        dev._fit_eval(buf, e, key)
        fill(tm, buf.votes, votes)
        dev._fit_decide_fb(buf, e, key)
    return host(tm, buf.feedback_type)


def update_weights(tm, buf, fb=None, ids=None) -> None:
    dev = tm.dev
    with _ctx(tm):
        if fb is not None:
            _set_feedback(tm, buf, fb, ids)
        dev._fit_update_weights(buf)


def activation(tm, votes) -> np.ndarray:
    """`votes -> y_hat` through the compiled activation, no clauses involved."""
    votes = np.ascontiguousarray(votes, dtype=np.float32)
    y_hat = np.zeros_like(votes)
    dev = tm.dev
    if not is_cuda(tm):
        dev.lib.votes_activation(votes.ctypes.data_as(_float_p), y_hat.ctypes.data_as(_float_p))
        return y_hat

    import cupy as cp

    with _ctx(tm):
        v, y = cp.asarray(votes), cp.zeros_like(cp.asarray(votes))
        dev.cu.votes_activation(1, (v, y))
        return y.get()


def loss_and_grad(tm, y_hat, y) -> tuple[float, np.ndarray]:
    """`compute_loss` through the compiled kernel. Returns (loss, grad) with grad = -dL/dz."""
    y_hat = np.ascontiguousarray(y_hat, dtype=np.float32)
    y = np.ascontiguousarray(y, dtype=np.float32)
    grad, loss = np.zeros_like(y_hat), np.zeros(1, dtype=np.float32)
    dev = tm.dev
    if not is_cuda(tm):
        dev.lib.loss_gradient(*(a.ctypes.data_as(_float_p) for a in (y_hat, y, grad, loss)))
        return float(loss[0]), grad

    import cupy as cp

    with _ctx(tm):
        a, b, g, l = cp.asarray(y_hat), cp.asarray(y), cp.zeros_like(cp.asarray(y_hat)), cp.zeros(1, dtype=cp.float32)
        dev.cu.loss_gradient(1, (a, b, g, l))
        return float(l.get()[0]), g.get()
