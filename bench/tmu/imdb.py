import argparse

import numpy as np
from tmu.models.classification.coalesced_classifier import TMCoalescedClassifier
from tmu.models.classification.vanilla_classifier import TMClassifier

from ytm.utils import Profiler

from ..data import load_imdb
from ..log import RunLog


def prepare(X_train, Y_train, X_test, Y_test):
    return X_train.astype(np.uint32), Y_train.astype(np.uint32), X_test.astype(np.uint32), Y_test.astype(np.uint32)


def train_model(tm, X_train, Y_train, X_test, Y_test, epochs: int, log: RunLog, expname: str):
    for epoch in range(1, epochs + 1):
        with Profiler() as fit_p:
            tm.fit(X_train, Y_train)
        with Profiler() as pred_p:
            pred = tm.predict(X_test)
        fit, predict = fit_p.get_profile(), pred_p.get_profile()
        acc = (pred == Y_test).mean()
        log.write(epoch, fit, predict, {"acc": acc})
        print(f"{expname} epoch {epoch}/{epochs}  fit {fit.elapsed:.2f}s  predict {predict.elapsed:.2f}s  acc {acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_features", type=int, default=12000)
    parser.add_argument("--coalesced", type=int, choices=(0, 1), default=0)
    parser.add_argument("--n_clauses", type=int, default=5000)
    parser.add_argument("--T", type=float, default=10000)
    parser.add_argument("--s", type=float, default=1.1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--platform", default="CPU")
    args = parser.parse_args()

    expname = "imdb"

    TM = TMCoalescedClassifier if args.coalesced else TMClassifier
    tm = TM(args.n_clauses, args.T, args.s, platform=args.platform, weighted_clauses=True, seed=args.seed)
    log = RunLog("tmu", expname, {**vars(args), "weighted_clauses": True})

    train_model(tm, *prepare(*load_imdb(args.max_features)), epochs=args.epochs, log=log, expname=expname)
