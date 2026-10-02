"""Train discrete or guided TM on Fashion-MNIST dataset.

Usage:
    python fmnist.py {discrete,guided} [options]

Examples:
    python fmnist.py discrete                                  # 3x3 conv, cpu:1
    python fmnist.py discrete --patch none --T 5000            # flat, no conv
    python fmnist.py guided --lr 0.5 --device cuda:0 --epochs 20

Options:
    - python fmnist.py <discrete/guided> --help shows all options.
    - Common:    --epochs --clause_drop_p --levels --n_clauses --s --patch H,W|none --seed --device cpu:N|cuda:N
    - discrete:  --T
    - guided:    --lr --lambda
"""

import argparse

import numpy as np
from datasets import load_dataset

from ytm.discrete import MultiClassTM as DiscreteTM
from ytm.guided import MultiClassTM as GuidedTM
from ytm.utils import Timer, print_table


def load_fmnist(levels: int):
    ds = load_dataset("zalando-datasets/fashion_mnist")
    # Pixels 0-255 -> levels 0-levels, thermometer encoded by the TM.
    X_train, X_test = ((levels * np.array(ds[split]["image"], dtype=np.float32) / 255.0).astype(np.int32) for split in ("train", "test"))
    Y_train, Y_test = (np.array(ds[split]["label"], dtype=np.uint8) for split in ("train", "test"))
    return X_train, Y_train, X_test, Y_test


def train_model(tm, xtrain, ytrain, xtest, ytest, epochs: int, clause_drop_p: float, model_name: str):
    for epoch in range(epochs):
        with (fit_timer := Timer()):
            loss = tm.fit(xtrain, ytrain, clause_drop_p=clause_drop_p)

        with (test_timer := Timer()):
            test_pred, _ = tm.predict(xtest)

        with (train_timer := Timer()):
            train_pred, _ = tm.predict(xtrain)

        train_log = {
            "Acc": f"{(train_pred == ytrain).mean() * 100:.4f}%",
            "Eval Time": f"{train_timer.elapsed:.2f}s",
            "Fit Time": f"{fit_timer.elapsed:.2f}s",
        }
        test_log = {"Acc": f"{(test_pred == ytest).mean() * 100:.4f}%", "Eval Time": f"{test_timer.elapsed:.2f}s"}

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
    parser = argparse.ArgumentParser(description="TM Fashion-MNIST dataset")
    # Subcommand to select discrete or guided
    models = parser.add_subparsers(dest="model", required=True)

    # Common args
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--epochs", type=int, default=10)
    common.add_argument("--clause_drop_p", type=float, default=0.0)
    common.add_argument("--levels", type=int, default=8, help="thermometer levels")
    common.add_argument("--n_clauses", type=int, default=40000)
    common.add_argument("--s", type=float, default=10.0)
    common.add_argument(
        "--patch",
        dest="patch_dim",
        type=lambda v: None if v.lower() == "none" else tuple(map(int, v.split(","))),
        default="3,3",
        help="H,W or none",
    )
    common.add_argument("--seed", type=int, default=10)
    common.add_argument("--device", type=str, default="cpu:1", help="cpu:N_THREADS or cuda:GPU_ID")

    # Discrete args
    discrete = models.add_parser("discrete", parents=[common])
    discrete.add_argument("--T", type=float, default=15000)

    # Guided args
    guided = models.add_parser("guided", parents=[common])
    guided.add_argument("--lr", type=float, default=0.5)
    guided.add_argument("--lambda", dest="lambda_", type=float, default=3.0)

    # Parse args
    args = parser.parse_args()

    # Args to dict
    params = vars(args)
    model_name, epochs, clause_drop_p = params.pop("model"), params.pop("epochs"), params.pop("clause_drop_p")
    levels = params.pop("levels")

    # Model Initialization
    TM = DiscreteTM if model_name == "discrete" else GuidedTM
    tm = TM(
        **params,
        dim=(28, 28, 1),
        n_classes=10,
        feat_maxs=levels,
    )

    # Training
    train_model(tm, *load_fmnist(levels), epochs=epochs, clause_drop_p=clause_drop_p, model_name=model_name)
