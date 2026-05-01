import numpy as np
from datasets import load_dataset
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import seaborn as sns

from ytm.discrete.classifier import MultiClassTM
from ytm.discrete.interpret import wac
from ytm.utils import Timer

icefire = sns.color_palette("icefire", as_cmap=True)

CIFAR10_LABELS = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck",
]


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


def plot_wac(X_org, Y, wac_images):
    n = len(X_org)
    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4))

    for i in range(n):
        axes[0, i].imshow(X_org[i])
        axes[0, i].set_title(CIFAR10_LABELS[Y[i]], fontsize=8)
        axes[0, i].axis("off")

        # Sum over channels for a single heatmap
        img = wac_images[i].sum(axis=-1)
        img_copy = img.copy()
        if img_copy.min() < 0:
            img_copy[img_copy < 0] = img_copy[img_copy < 0] / (-1 * img_copy[img_copy < 0].min() + 1e-7)
        if img_copy.max() > 0:
            img_copy[img_copy > 0] = img_copy[img_copy > 0] / (img_copy[img_copy > 0].max() + 1e-7)
        img_copy = Normalize(-1, 1)(img_copy)

        axes[1, i].imshow(img_copy, cmap=icefire)
        axes[1, i].axis("off")

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    ds = load_dataset("uoft-cs/cifar10")
    Y_train, Y_test = map(np.array, (ds["train"]["label"], ds["test"]["label"]))
    X_train, X_test = map(
        lambda x: np.asarray(20 * np.array(x).astype(np.float32) / 255.0, dtype=np.int32),
        (ds["train"]["img"], ds["test"]["img"]),
    )
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    print(f"X_train min: {X_train.min()}, X_train max: {X_train.max()}")

    tm = MultiClassTM(
        n_clauses=1000,
        T=5000,
        s=10.0,
        dim=(32, 32, 3),
        n_classes=10,
        patch_dim=(5, 5),
        coalesced=False,
        feat_mins=int(X_train.min()),
        feat_maxs=int(X_train.max()),
        n_threads=8,
        device="cuda",
        seed=42,
    )

    train(tm, X_train, Y_train, X_test, Y_test, epochs=10)

    # Pick one sample per class
    index_per_class = []
    for i in range(10):
        index_per_class.append(np.argwhere(Y_test == i).ravel()[0])

    Xs = X_test[index_per_class]
    Xs_org = X_test_org[index_per_class]
    Ys = Y_test[index_per_class]

    # Compute WAC for each sample using its true class
    wac_output = wac(tm, Xs, target_classes=Ys)  # (10, H, W, D)

    fig = plot_wac(Xs_org, Ys, wac_output)
    plt.show()
