import argparse

import numpy as np
from datasets import load_dataset
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize

from ytm.discrete.classifier import MultiClassTM
from ytm.discrete.interpret import wic
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
                "Train": {
                    "Acc": f"{train_acc * 100:.4f}%",
                    "Fit Time": f"{train_fit_timer.elapsed:.4f}s",
                    "Infer Time": f"{train_timer.elapsed:.4f}s",
                },
                "Test": {"Acc": f"{test_acc * 100:.4f}%", "Infer Time": f"{test_timer.elapsed:.4f}s"},
            },
        )


def plot_wic(wic_images, n_classes):
    fig, axes = plt.subplots(1, n_classes, figsize=(2 * n_classes, 2), layout="compressed")

    for c in range(n_classes):
        img = Normalize(-1, 1)(wic_images[c].copy())

        axes[c].imshow(img, cmap="coolwarm")
        axes[c].set_title(f"Class {c}")
        axes[c].axis("off")

    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clauses", type=int, default=500)
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--s", type=float, default=10.0)
    parser.add_argument("--patch", type=int, nargs=2, default=[10, 10], metavar=("H", "W"))
    parser.add_argument("--seed", type=lambda x: None if x == "None" else int(x), default=10)
    parser.add_argument("--coalesced", type=int, choices=[0, 1], default=1)
    parser.add_argument("--n_threads", type=int, default=8)
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    ds = load_dataset("ylecun/mnist")
    Y_train, Y_test = map(lambda x: np.array(x).astype(np.uint8), (ds["train"]["label"], ds["test"]["label"]))
    X_train, X_test = map(
        lambda x: np.where(np.array(x) > 75, 1, 0).astype(np.uint8),
        (ds["train"]["image"], ds["test"]["image"]),
    )

    tm = MultiClassTM(
        n_clauses=args.n_clauses,
        T=args.T,
        s=args.s,
        dim=(28, 28, 1),
        n_classes=10,
        patch_dim=tuple(args.patch),
        stride=(1, 1),
        feat_mins=0,
        feat_maxs=1,
        seed=args.seed,
        coalesced=True if args.coalesced == 1 else False,
        device=args.device,
        n_threads=args.n_threads,
    )

    train(tm, X_train, Y_train, X_test, Y_test, epochs=args.epochs)

    wic_output = wic(tm)  # (10, H, W, D)
    wic_images = wic_output.sum(axis=-1)  # (10, H, W)

    fig = plot_wic(wic_images, n_classes=10)
    plt.show()
