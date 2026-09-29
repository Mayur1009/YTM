"""One small workload over every C entry point; launched under a sanitizer by test_sanitize.py."""

import json
import sys

import numpy as np

def build(spec: dict, device: str):
    kw = {k: tuple(v) if isinstance(v, list) else v for k, v in spec["kw"].items()}
    if spec["backend"] == "discrete":
        from ytm.discrete import BinaryTM, MultiClassTM

        if spec.get("kind") == "binary":
            kw.pop("n_classes", None)
            return BinaryTM(**kw, T=10.0, device=device)
        return MultiClassTM(**kw, T=10.0, device=device)
    from ytm.guided import MultiClassTM

    return MultiClassTM(**kw, device=device)


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
