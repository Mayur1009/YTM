import numpy as np
from datasets import load_dataset
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import seaborn as sns

from ytm.discrete.classifier import MultiClassTM
from ytm.discrete.interpret import wic
from ytm.utils import Timer

icefire = sns.color_palette("icefire", as_cmap=True)


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
        print(
            f"Epoch {epoch + 1} | Acc> Train: {train_acc * 100:.4f}% Test: {test_acc * 100:.4f}% | Time> Fit: {train_fit_timer.elapsed:.4f}s Infer Train: {train_timer.elapsed:.4f}s Infer Test: {test_timer.elapsed:.4f}s"
        )


def plot_wic(wic_images, n_classes):
    fig, axes = plt.subplots(1, n_classes, figsize=(2 * n_classes, 2))

    for c in range(n_classes):
        img = wic_images[c].copy()
        if img.min() < 0:
            img[img < 0] = img[img < 0] / (-1 * img[img < 0].min() + 1e-7)
        if img.max() > 0:
            img[img > 0] = img[img > 0] / (img[img > 0].max() + 1e-7)
        img = Normalize(-1, 1)(img)

        axes[c].imshow(img, cmap=icefire)
        axes[c].set_title(f"Class {c}")
        axes[c].axis("off")

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    ds = load_dataset("ylecun/mnist")
    Y_train, Y_test = map(lambda x: np.array(x).astype(np.uint8), (ds["train"]["label"], ds["test"]["label"]))
    X_train, X_test = map(
        lambda x: np.where(np.array(x) > 75, 1, 0).astype(np.uint8),
        (ds["train"]["image"], ds["test"]["image"]),
    )

    tm = MultiClassTM(
        n_clauses=500,
        T=1000,
        s=10,
        dim=(28, 28, 1),
        n_classes=10,
        patch_dim=(10, 10),
        stride=(1, 1),
        feat_mins=0,
        feat_maxs=1,
        seed=10,
        device="cuda",
        n_threads=8,
    )

    train(tm, X_train, Y_train, X_test, Y_test, epochs=1)

    wic_output = wic(tm)          # (10, H, W, D)
    wic_images = wic_output.sum(axis=-1)  # (10, H, W)

    fig = plot_wic(wic_images, n_classes=10)
    plt.show()
