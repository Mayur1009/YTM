import argparse

from ytm.tm import MultiClassTM
from ytm.utils import Profiler

from ..data import load_ag_news
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
    parser.add_argument("--max_features", type=int, default=2000)
    parser.add_argument("--coalesced", type=int, choices=(0, 1), default=1)
    parser.add_argument("--n_clauses", type=int, default=8000)
    parser.add_argument("--T", type=float, default=40000)
    parser.add_argument("--s", type=float, default=2.0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    expname = "ag_news"

    tm = MultiClassTM(
        n_clauses=args.n_clauses,
        T=args.T,
        s=args.s,
        dim=(args.max_features, 1, 1),
        n_classes=4,
        coalesced=bool(args.coalesced),
        seed=args.seed,
        device=args.device,
    )
    log = RunLog("ytm", expname, {**vars(args), "dim": (args.max_features, 1, 1), "n_classes": 4})

    train_model(tm, *load_ag_news(args.max_features), epochs=args.epochs, log=log, expname=expname)
