import argparse

import numpy as np
from PySparseCoalescedTsetlinMachineCUDA.tm import MultiOutputTsetlinMachine
from sklearn.metrics import f1_score

from ytm.utils import Profiler

from ..data import load_go_emotions
from ..log import RunLog
from .pysparse import PySparseTM, to_csr


def prepare(X_train, Y_train, X_test, Y_test):
    return to_csr(X_train, Y_train.astype(np.uint32), X_test, Y_test.astype(np.uint32))


def train_model(tm, X_train, Y_train, X_test, Y_test, epochs: int, log: RunLog, expname: str):
    for epoch in range(1, epochs + 1):
        with Profiler() as fit_p:
            tm.fit(X_train, Y_train)
        with Profiler() as pred_p:
            pred = tm.predict(X_test)
        fit, predict = fit_p.get_profile(), pred_p.get_profile()
        acc = f1_score(Y_test, pred, average="macro", zero_division=0)
        log.write(epoch, fit, predict, {"acc": acc})
        print(f"{expname} epoch {epoch}/{epochs}  fit {fit.elapsed:.2f}s  predict {predict.elapsed:.2f}s  acc {acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_features", type=int, default=10000)
    parser.add_argument("--n_clauses", type=int, default=22400)
    parser.add_argument("--T", type=float, default=25000)
    parser.add_argument("--s", type=float, default=2.0)
    parser.add_argument("--q", type=float, default=1.5)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    expname = "go_emotions"

    tm = PySparseTM(MultiOutputTsetlinMachine(args.n_clauses, args.T, args.s, q=args.q))
    log = RunLog("pysparse", expname, {**vars(args), "coalesced": 1, "platform": "CUDA"})

    train_model(tm, *prepare(*load_go_emotions(args.max_features)), epochs=args.epochs, log=log, expname=expname)
