"""Train discrete or guided TM on Iris (4 numeric features, 3 classes).

Features discretized per feature into --bins levels (sklearn KBinsDiscretizer), thermometer encoded by the TM.
Random stratified 70/30 train/test split.

Usage:
    python iris.py {discrete,guided} [options]

Examples:
    python iris.py discrete
    python iris.py discrete --bins 4 --strategy uniform
    python iris.py guided --lr 0.5 --epochs 200

Options:
    - python iris.py <discrete/guided> --help shows all options.
    - Common:    --epochs --bins --strategy uniform|quantile --n_clauses --s --max_includes --seed --device cpu:N|cuda:N
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


def load(bins: int, strategy: str, seed: int):
    X, Y = load_iris(return_X_y=True)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y.astype(np.uint8), test_size=0.3, stratify=Y, random_state=seed)
    d = KBinsDiscretizer(n_bins=bins, encode="ordinal", strategy=strategy)
    X_train, X_test = d.fit_transform(X_train).astype(np.int32), d.transform(X_test).astype(np.int32)
    return X_train, Y_train, X_test, Y_test, d.n_bins_ - 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TM Iris dataset")
    # Subcommand to select discrete or guided
    models = parser.add_subparsers(dest="model", required=True)

    # Common args
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--epochs", type=int, default=100)
    common.add_argument("--bins", type=int, default=8, help="discretization levels per feature")
    common.add_argument("--strategy", choices=["uniform", "quantile"], default="quantile")
    common.add_argument("--n_clauses", type=int, default=20)
    common.add_argument("--s", type=float, default=3.0)
    common.add_argument("--max_includes", type=int, default=None, help="max literals per clause (default: no limit)")
    common.add_argument("--seed", type=int, default=10)
    common.add_argument("--device", type=str, default="cpu:1", help="cpu:N_THREADS or cuda:GPU_ID")

    # Discrete args
    discrete = models.add_parser("discrete", parents=[common])
    discrete.add_argument("--T", type=float, default=10)

    # Guided args
    guided = models.add_parser("guided", parents=[common])
    guided.add_argument("--lr", type=float, default=1.0)
    guided.add_argument("--lambda", dest="lambda_", type=float, default=1.0)

    # Parse args
    args = parser.parse_args()

    # Args to dict
    params = vars(args)
    model_name, epochs = params.pop("model"), params.pop("epochs")
    bins, strategy = params.pop("bins"), params.pop("strategy")

    X_train, Y_train, X_test, Y_test, feat_maxs = load(bins, strategy, params["seed"])

    # Model Initialization
    TM = DiscreteTM if model_name == "discrete" else GuidedTM
    tm = TM(**params, dim=(X_train.shape[1], 1, 1), n_classes=3, feat_maxs=feat_maxs)

    # Training
    for epoch in range(epochs):
        loss = tm.fit(X_train, Y_train)
        train_acc = (tm.predict(X_train)[0] == Y_train).mean()
        test_acc = (tm.predict(X_test)[0] == Y_test).mean()
        loss_str = f"  loss {loss:.4f}" if loss is not None else ""  # Only loss-guided TM returns loss
        print(f"{model_name} epoch {epoch + 1}/{epochs}  train acc {train_acc * 100:.2f}%  test acc {test_acc * 100:.2f}%{loss_str}")
