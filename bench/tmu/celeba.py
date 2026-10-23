import argparse

import numpy as np
from PySparseCoalescedTsetlinMachineCUDA.tm import MultiOutputConvolutionalTsetlinMachine2D
from sklearn.metrics import f1_score

from ytm.utils import Profiler

from ..data import load_celeba, thermometer
from ..log import RunLog
from .pysparse import PySparseTM, to_csr


def prepare(X_train, Y_train, X_test, Y_test, levels: int):
    X_train, X_test = thermometer(X_train, levels), thermometer(X_test, levels)
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
    parser.add_argument("--patch", type=int, default=3)
    parser.add_argument("--levels", type=int, default=8)
    parser.add_argument("--n_clauses", type=int, default=80000)
    parser.add_argument("--T", type=float, default=80000)
    parser.add_argument("--s", type=float, default=30.0)
    parser.add_argument("--q", type=float, default=10.0)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    expname = "celeba_conv"
    dim = (64, 64, 3 * args.levels)

    tm = PySparseTM(MultiOutputConvolutionalTsetlinMachine2D(args.n_clauses, args.T, args.s, dim, (args.patch, args.patch), q=args.q))
    log = RunLog("pysparse", expname, {**vars(args), "coalesced": 1, "platform": "CUDA", "dim": dim})

    data = prepare(*load_celeba(args.levels), levels=args.levels)
    train_model(tm, *data, epochs=args.epochs, log=log, expname=expname)
