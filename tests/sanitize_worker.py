"""One small workload over every C entry point; launched under a sanitizer by test_sanitize.py."""

import json
import sys

import numpy as np

SAN = ["-fsanitize=address,undefined,float-divide-by-zero,float-cast-overflow", "-fno-sanitize-recover=all", "-fno-omit-frame-pointer"]


def _flags():
    from ytm._core._device_checks import DEFAULT_COMPILE_FLAGS, resolve_link_flags, select_compiler

    base = [f for f in DEFAULT_COMPILE_FLAGS if not f.startswith(("-O", "-march", "-mtune"))]
    return base + ["-O1", "-g"] + SAN + resolve_link_flags(select_compiler())


def build(spec: dict, device: str):
    kw = {k: tuple(v) if isinstance(v, list) else v for k, v in spec["kw"].items()}
    extra = {"compile_flags": _flags()} if device.startswith("cpu") else {}
    if spec["backend"] == "discrete":
        from ytm._discrete import BinaryTM, MultiClassTM

        if spec.get("kind") == "binary":
            kw.pop("n_classes", None)
            return BinaryTM(**kw, T=10.0, device=device, **extra)
        return MultiClassTM(**kw, T=10.0, device=device, **extra)
    from ytm._guided import MultiClassTM

    return MultiClassTM(**kw, device=device, **extra)


def _prepared(tm, X):
    """What BaseTM._prepare_X returns; used only for the bypass below (feat_mins are 0 so no shift)."""
    cfg = tm.config
    return np.asarray(X.reshape(X.shape[0], *cfg._dim), dtype=cfg._fbound_dtype, order="C")


def _bypass(cfg) -> bool:
    """True when BaseTM._prepare_X would raise: one patch that is smaller than the image (known bug, src/ytm/_core/base.py:39)."""
    return cfg._n_patches == 1 and tuple(cfg._patch_dim) != (cfg._dim[0], cfg._dim[1])


def _fit_dev(tm, backend: str, X, Y, batch_size: int) -> None:
    """tm.fit minus _prepare_X: builds the same device-level arguments as _discrete/_guided BaseTM._fit."""
    cfg = tm.config
    Xp = _prepared(tm, X)
    onehot = np.zeros((Y.shape[0], cfg.n_classes), dtype=np.int8)
    for i in range(cfg.n_classes):
        onehot[:, i] = Y == i if cfg.n_classes > 1 else Y
    if backend == "guided":
        tm.dev.fit_epoch(Xp, np.asarray(onehot, dtype=np.float32, order="C"), 0.0, batch_size, None, None)
        return
    enc = np.asarray((onehot.astype(np.float32) * 2 - 1) * cfg._T_max, dtype=np.float32, order="C")
    label_probs = np.full_like(enc, cfg.q / max(1, cfg.n_classes - 1), dtype=np.float32)
    label_probs[enc == cfg._T_max] = 1.0
    tm.dev.fit_epoch(Xp, enc, 0.0, batch_size, np.asarray(label_probs, dtype=np.float32, order="C"))


def main() -> None:
    spec, device = json.loads(sys.argv[1]), sys.argv[2]
    tm = build(spec, device)
    cfg = tm.config
    rng = np.random.default_rng(0)
    n = 7
    X = rng.integers(0, int(np.max(cfg._feat_maxs)) + 1, size=(n, int(np.prod(cfg._dim))), dtype=np.int32)
    # A one-class (binary) model takes Y in {0, 1} with both values present; its wac/wic target class is always 0.
    Y = np.arange(n) % 2 if cfg.n_classes == 1 else rng.integers(0, cfg.n_classes, n)
    target = np.zeros(n, dtype=np.int64) if cfg.n_classes == 1 else Y
    if _bypass(cfg):
        # tm.fit/predict/score/wac all call _prepare_X, which raises here (known source bug); go to the device directly.
        for _ in range(2):
            _fit_dev(tm, spec["backend"], X, Y, 3)
        Xp = _prepared(tm, X)
        tm.dev.calc_class_sums(Xp, 3)
        tm.dev.calc_class_sums(Xp)
        tm.dev.wac(Xp, target, +1, 3)
    else:
        for _ in range(2):
            tm.fit(X, Y, batch_size=3)
        tm.predict(X, batch_size=3)
        tm.score(X)
        tm.wac(X, target, +1, batch_size=3)
    for c in range(cfg.n_classes):
        tm.wic(c, +1)
    tm.get_clauses()
    if cfg._n_patches > 1:
        tm.transform_patchwise(X)
    print("OK")


if __name__ == "__main__":
    main()
