import argparse

import numpy as np
from sklearn.datasets import make_friedman1
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from ytm.discrete.regressor import RegressionTM
from ytm.utils import print_table


def train(tm: RegressionTM, X_train, Y_train, X_test, Y_test, epochs=1):
    for epoch in range(epochs):
        tm.fit(X_train, Y_train)
        pred_train, _ = tm.predict(X_train)
        pred_test, _ = tm.predict(X_test)
        print_table(
            f"Epoch {epoch + 1}/{epochs}",
            {
                "Train": {"MAE": f"{np.mean(np.abs(pred_train - Y_train)):.4f}", "R2": f"{r2_score(Y_train, pred_train):.4f}", "MSE": f"{mean_squared_error(Y_train, pred_train):.4f}"},
                "Test":  {"MAE": f"{np.mean(np.abs(pred_test - Y_test)):.4f}", "R2": f"{r2_score(Y_test, pred_test):.4f}", "MSE": f"{mean_squared_error(Y_test, pred_test):.4f}"},
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clauses", type=int, default=2000)
    parser.add_argument("--T", type=int, default=3000)
    parser.add_argument("--s", type=float, default=2.0)
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--seed", type=lambda x: None if x == "None" else int(x), default=42)
    parser.add_argument("--n_threads", type=int, default=8)
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()

    X, Y = make_friedman1(n_samples=args.n_samples, n_features=10, noise=0.1, random_state=args.seed)
    X = ((X / X.max()) * 300.0).astype(np.int32)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=args.seed)

    tm = RegressionTM(
        n_clauses=args.n_clauses,
        T=args.T,
        s=args.s,
        dim=(10, 1, 1),
        y_range=(float(Y.min()), float(Y.max())),
        feat_mins=X.min(),
        feat_maxs=X.max(),
        device=args.device,
        n_threads=args.n_threads,
        seed=args.seed,
    )

    train(tm, X_train, Y_train, X_test, Y_test, epochs=args.epochs)
