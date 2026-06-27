import argparse

import numpy as np
from datasets import load_dataset

from ytm.discrete.classifier import MultiClassTM
from ytm.utils import Timer, print_table


def train(tm: MultiClassTM, X_train, Y_train, X_test, Y_test, epochs=1):
    for epoch in range(epochs):
        train_fit_timer = Timer()
        with train_fit_timer:
            tm.fit(X_train, Y_train)

        test_timer = Timer()
        with test_timer:
            test_pred, _ = tm.predict(X_test)

        train_timer = Timer()
        with train_timer:
            train_pred, _ = tm.predict(X_train)

        test_acc = np.mean(Y_test == test_pred)
        train_acc = np.mean(Y_train == train_pred)
        print_table(
            f"Epoch {epoch + 1}/{epochs}",
            {
                "Train": {"Acc": f"{train_acc * 100:.4f}%", "Fit Time": f"{train_fit_timer.elapsed:.4f}s", "Infer Time": f"{train_timer.elapsed:.4f}s"},
                "Test":  {"Acc": f"{test_acc * 100:.4f}%", "Infer Time": f"{test_timer.elapsed:.4f}s"},
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clauses", type=int, default=1000)
    parser.add_argument("--T", type=int, default=5000)
    parser.add_argument("--s", type=float, default=10.0)
    parser.add_argument("--patch", type=int, nargs=2, default=[5, 5], metavar=("H", "W"))
    parser.add_argument("--seed", type=lambda x: None if x == "None" else int(x), default=42)
    parser.add_argument("--coalesced", type=int, choices=[0, 1], default=0)
    parser.add_argument("--n_threads", type=int, default=8)
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    ds = load_dataset("uoft-cs/cifar10")
    Y_train, Y_test = map(np.array, (ds["train"]["label"], ds["test"]["label"]))
    X_train, X_test = map(
        lambda x: np.asarray(20 * np.array(x).astype(np.float32) / 255.0, dtype=np.int32),
        (ds["train"]["img"], ds["test"]["img"]),
    )
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    print(f"X_train min: {X_train.min()}, X_train max: {X_train.max()}")

    tm = MultiClassTM(
        n_clauses=args.n_clauses,
        T=args.T,
        s=args.s,
        dim=(32, 32, 3),
        n_classes=10,
        patch_dim=tuple(args.patch),
        coalesced=True if args.coalesced == 1 else False,
        feat_mins=int(X_train.min()),
        feat_maxs=int(X_train.max()),
        n_threads=args.n_threads,
        device=args.device,
        seed=args.seed,
    )

    train(tm, X_train, Y_train, X_test, Y_test, epochs=args.epochs)
