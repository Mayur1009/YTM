import argparse

from ytm.tm import MultiClassTM
from ytm.utils import Profiler

from ..data import load_mnist
from ..log import RunLog


def train_model(tm, X_train, Y_train, X_test, Y_test, epochs: int, log: RunLog, expname: str):
    for epoch in range(1, epochs + 1):
        with Profiler() as fit_p:
            tm.fit(X_train, Y_train)
        with Profiler() as pred_p:
            pred, _ = tm.predict(X_test)
        fit, predict = fit_p.get_profile(), pred_p.get_profile()
        acc = (pred == Y_test).mean()
        log.write(epoch, fit, predict, {"acc": acc})
        print(f"{expname} epoch {epoch}/{epochs}  fit {fit.elapsed:.2f}s  predict {predict.elapsed:.2f}s  acc {acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch", type=lambda v: None if v.lower() == "none" else (int(v), int(v)), default="10")
    parser.add_argument("--coalesced", type=int, choices=(0, 1), default=1)
    parser.add_argument("--n_clauses", type=int, default=500)
    parser.add_argument("--T", type=float, default=1000)
    parser.add_argument("--s", type=float, default=10.0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    expname = "mnist_conv" if args.patch else "mnist_flat"

    tm = MultiClassTM(
        n_clauses=args.n_clauses,
        T=args.T,
        s=args.s,
        dim=(28, 28, 1),
        n_classes=10,
        patch_dim=args.patch,
        coalesced=bool(args.coalesced),
        seed=args.seed,
        device=args.device,
    )
    log = RunLog("ytm", expname, {**vars(args), "dim": (28, 28, 1), "n_classes": 10})

    train_model(tm, *load_mnist(), epochs=args.epochs, log=log, expname=expname)
