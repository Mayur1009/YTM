import argparse

import numpy as np
from datasets import load_dataset
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize

from ytm.discrete.classifier import MultiClassTM
from ytm.discrete.interpret import wac
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


def plot_wac(X, Y, wac_images):
    n = len(X)
    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4), layout="compressed")

    for i in range(n):
        axes[0, i].imshow(X[i].reshape(28, 28), cmap="gray")
        axes[0, i].set_title(f"Label: {Y[i]}")
        axes[0, i].axis("off")

        img = wac_images[i].copy()
        # Normalize positive and negative independently
        if img.min() < 0:
            img[img < 0] = img[img < 0] / (-1 * img[img < 0].min() + 1e-7)
        if img.max() > 0:
            img[img > 0] = img[img > 0] / (img[img > 0].max() + 1e-7)
        img = Normalize(-1, 1)(img)

        axes[1, i].imshow(img, cmap="coolwarm")
        axes[1, i].axis("off")

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

    # Pick one sample per class
    index_per_class = []
    for i in range(10):
        index_per_class.append(np.argwhere(Y_test == i).ravel()[0])

    Xs = X_test[index_per_class]
    Ys = Y_test[index_per_class]

    # Compute WAC for each sample using its true class
    wac_output = wac(tm, Xs, target_classes=Ys)  # (10, H, W, D)
    wac_images = wac_output.sum(axis=-1)  # (10, H, W)

    fig = plot_wac(Xs, Ys, wac_images)
    plt.show()
