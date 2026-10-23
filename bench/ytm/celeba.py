import argparse

from sklearn.metrics import f1_score

from ytm.tm import MultiOutputTM
from ytm.utils import Profiler

from ..data import load_celeba
from ..log import RunLog


def train_model(tm, X_train, Y_train, X_test, Y_test, epochs: int, log: RunLog, expname: str):
    for epoch in range(1, epochs + 1):
        with Profiler() as fit_p:
            tm.fit(X_train, Y_train)
        with Profiler() as pred_p:
            pred, _ = tm.predict(X_test)
        fit, predict = fit_p.get_profile(), pred_p.get_profile()
        acc = f1_score(Y_test, pred, average="macro", zero_division=0)
        log.write(epoch, fit, predict, {"acc": acc})
        print(f"{expname} epoch {epoch}/{epochs}  fit {fit.elapsed:.2f}s  predict {predict.elapsed:.2f}s  acc {acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch", type=int, default=3)
    parser.add_argument("--levels", type=int, default=8)
    parser.add_argument("--n_clauses", type=int, default=80000)
    parser.add_argument("--T", type=float, default=80000)
    parser.add_argument("--s", type=float, default=30.0)
    parser.add_argument("--q", type=float, default=10.0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    expname = "celeba_conv"
    dim = (64, 64, 3)

    tm = MultiOutputTM(
        n_clauses=args.n_clauses,
        T=args.T,
        s=args.s,
        q=args.q,
        dim=dim,
        n_classes=40,
        feat_maxs=args.levels,
        patch_dim=(args.patch, args.patch),
        seed=args.seed,
        device=args.device,
    )
    log = RunLog("ytm", expname, {**vars(args), "coalesced": 1, "dim": dim, "n_classes": 40})

    train_model(tm, *load_celeba(args.levels), epochs=args.epochs, log=log, expname=expname)
