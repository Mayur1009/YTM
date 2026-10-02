"""Train discrete or guided TM on MNIST dataset.

Usage:
    python mnist.py {discrete,guided} [options]

Examples:
    python mnist.py discrete                                  # 10x10 conv, cpu:1
    python mnist.py discrete --patch none --T 5000            # flat, no conv
    python mnist.py guided --lr 0.5 --device cuda:0 --epochs 20

Options:
    - python mnist.py <discrete/guided> --help shows all options.
    - Common:      --epochs --n_clauses --s --patch H,W|none --seed --device cpu:N|cuda:N
    - discrete:  --T
    - guided:    --lr --lambda
"""

import argparse

import numpy as np
from datasets import load_dataset

from ytm.discrete import MultiClassTM as DiscreteTM
from ytm.guided import MultiClassTM as GuidedTM
from ytm.utils import Timer, print_table


def load_mnist():
    ds = load_dataset("ylecun/mnist")
    X_train, X_test = ((np.array(ds[split]["image"]) > 75).astype(np.uint8) for split in ("train", "test"))
    Y_train, Y_test = (np.array(ds[split]["label"], dtype=np.uint8) for split in ("train", "test"))
    return X_train, Y_train, X_test, Y_test


def train_model(tm, xtrain, ytrain, xtest, ytest, epochs: int, model_name: str):
    for epoch in range(epochs):
        with (fit_timer := Timer()):
            loss = tm.fit(xtrain, ytrain)

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
    parser = argparse.ArgumentParser(description="TM MNIST dataset")
    # Subcommand to select discrete or guided
    models = parser.add_subparsers(dest="model", required=True)

    # Commmon args
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--epochs", type=int, default=10)
    common.add_argument("--n_clauses", type=int, default=500)
    common.add_argument("--s", type=float, default=10.0)
    common.add_argument(
        "--patch",
        dest="patch_dim",
        type=lambda v: None if v.lower() == "none" else tuple(map(int, v.split(","))),
        default="10,10",
        help="H,W or none",
    )
    common.add_argument("--seed", type=int, default=10)
    common.add_argument("--device", type=str, default="cpu:1", help="cpu:N_THREADS or cuda:GPU_ID")

    # Discrete args
    discrete = models.add_parser("discrete", parents=[common])
    discrete.add_argument("--T", type=float, default=1000)

    # Guided args
    guided = models.add_parser("guided", parents=[common])
    guided.add_argument("--lr", type=float, default=1.0)
    guided.add_argument("--lambda", dest="lambda_", type=float, default=1.0)

    # Parse args
    args = parser.parse_args()

    # Args to dict
    params = vars(args)
    model_name, epochs = params.pop("model"), params.pop("epochs")

    # Model Initialization
    TM = DiscreteTM if model_name == "discrete" else GuidedTM
    tm = TM(
        **params,
        dim=(28, 28, 1),
        n_classes=10,
    )

    # Training
    train_model(tm, *load_mnist(), epochs=epochs, model_name=model_name)
