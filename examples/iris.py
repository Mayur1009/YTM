"""Train discrete or guided TM on Iris (4 numeric features, 3 classes).

Usage:
    python iris.py {discrete,guided} [options]

Examples:
    python iris.py guided                                      # 2 clauses per class
    python iris.py discrete --T 10
    python iris.py guided --n_clauses 4 --coalesced 1          # 4 clauses shared by all classes

Options:
    - python iris.py <discrete/guided> --help shows all options.
    - Common:    --epochs --bins --strategy uniform|quantile --n_clauses --coalesced 0|1 --s --max_includes
                 --boost_tp_inc 0|1 --boost_tp_dec 0|1 --seed --device cpu:N|cuda:N --print
    - discrete:  --T
    - guided:    --lr --lambda
"""

import argparse

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer

from ytm.discrete import MultiClassTM as DiscreteTM
from ytm.guided import MultiClassTM as GuidedTM
from ytm.utils import Timer, print_table


def load(bins: int, strategy: str, seed: int):
    X, Y = load_iris(return_X_y=True)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y.astype(np.uint8), test_size=0.3, stratify=Y, random_state=seed)
    d = KBinsDiscretizer(n_bins=bins, encode="ordinal", strategy=strategy)
    X_train, X_test = d.fit_transform(X_train).astype(np.int32), d.transform(X_test).astype(np.int32)
    return X_train, Y_train, X_test, Y_test, d.n_bins_ - 1


def train_model(tm, xtrain, ytrain, xtest, ytest, epochs: int, model_name: str):
    for epoch in range(epochs):
        with (fit_timer := Timer()):
            loss = tm.fit(xtrain, ytrain)

        with (test_timer := Timer()):
            test_pred, _ = tm.predict(xtest)

        with (train_timer := Timer()):
            train_pred, _ = tm.predict(xtrain)

        train_log = {
            "Acc": f"{(train_pred == ytrain).mean() * 100:.2f}%",
            "Eval Time": f"{train_timer.elapsed:.2f}s",
            "Fit Time": f"{fit_timer.elapsed:.2f}s",
        }
        test_log = {"Acc": f"{(test_pred == ytest).mean() * 100:.2f}%", "Eval Time": f"{test_timer.elapsed:.2f}s"}

        if loss is not None:  # Only loss-guided TM returns loss
            train_log["Loss"] = f"{loss:.4f}"

        print_table(
            f"{model_name} epoch {epoch + 1}/{epochs}",
            {
                "Train": train_log,
                "Test": test_log,
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TM Iris dataset")
    # Subcommand to select discrete or guided
    models = parser.add_subparsers(dest="model", required=True)

    # Common args
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--epochs", type=int, default=200)
    common.add_argument("--bins", type=int, default=8, help="discretization levels per feature")
    common.add_argument("--strategy", choices=["uniform", "quantile"], default="quantile")
    common.add_argument("--n_clauses", type=int, default=2, help="per class, or total with --coalesced 1")
    common.add_argument("--coalesced", type=lambda v: bool(int(v)), default=False, metavar="0|1")
    common.add_argument("--s", type=float, default=1.5)
    common.add_argument("--max_includes", type=int, default=4, help="max literals per clause")
    common.add_argument("--boost_tp_inc", type=lambda v: bool(int(v)), default=False, metavar="0|1")
    common.add_argument("--boost_tp_dec", type=lambda v: bool(int(v)), default=False, metavar="0|1")
    common.add_argument("--seed", type=int, default=10)
    common.add_argument("--device", type=str, default="cpu:1", help="cpu:N_THREADS or cuda:GPU_ID")
    common.add_argument("--print", dest="print_clauses", action="store_true", help="print the learned clauses after training")

    # Discrete args
    discrete = models.add_parser("discrete", parents=[common])
    discrete.add_argument("--T", type=float, default=10)

    # Guided args
    guided = models.add_parser("guided", parents=[common])
    guided.add_argument("--lr", type=float, default=0.03)
    guided.add_argument("--lambda", dest="lambda_", type=float, default=1.0)

    # Parse args
    args = parser.parse_args()

    # Args to dict
    params = vars(args)
    model_name, epochs = params.pop("model"), params.pop("epochs")
    bins, strategy, show_clauses = params.pop("bins"), params.pop("strategy"), params.pop("print_clauses")

    X_train, Y_train, X_test, Y_test, feat_maxs = load(bins, strategy, params["seed"])

    # Model Initialization
    TM = DiscreteTM if model_name == "discrete" else GuidedTM
    tm = TM(**params, dim=(X_train.shape[1], 1, 1), n_classes=3, feat_maxs=feat_maxs)

    # Training
    train_model(tm, X_train, Y_train, X_test, Y_test, epochs=epochs, model_name=model_name)

    if show_clauses:
        # Bounds are in bin units, 0 .. bins - 1
        tm.print_clauses(["sepal_len", "sepal_wid", "petal_len", "petal_wid"], sort="w0,len")
