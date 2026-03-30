import numpy as np
from keras.datasets import fashion_mnist
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import seaborn as sns

from ytm.discrete.multiclass import MultiClassTM
from ytm.discrete.interpret import wac
from ytm.utils import Timer

icefire = sns.color_palette("icefire", as_cmap=True)

FMNIST_LABELS = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def train(tm: MultiClassTM, X_train, Y_train, X_test, Y_test, epochs=1):
    for epoch in range(epochs):
        train_fit_timer = Timer()
        with train_fit_timer:
            tm.fit(X_train, Y_train, clause_drop_p=0.5)

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


def plot_wac(X, Y, wac_images):
    n = len(X)
    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4))

    for i in range(n):
        axes[0, i].imshow(X[i].reshape(28, 28), cmap="gray")
        axes[0, i].set_title(FMNIST_LABELS[Y[i]], fontsize=8)
        axes[0, i].axis("off")

        img = wac_images[i].copy()
        # Normalize positive and negative independently
        if img.min() < 0:
            img[img < 0] = img[img < 0] / (-1 * img[img < 0].min() + 1e-7)
        if img.max() > 0:
            img[img > 0] = img[img > 0] / (img[img > 0].max() + 1e-7)
        img = Normalize(-1, 1)(img)

        axes[1, i].imshow(img, cmap=icefire)
        axes[1, i].axis("off")

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    (X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()

    # Discretize the pixel values in range [0, 255] to [0, 8] (i.e., 8 bins).
    X_train = np.asarray(8 * X_train.astype(np.float32) / 255.0, dtype=np.int32)
    X_test = np.asarray(8 * X_test.astype(np.float32) / 255.0, dtype=np.int32)

    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    print(f"X_train min: {X_train.min()}, X_train max: {X_train.max()}")

    tm = MultiClassTM(
        n_clauses=6000,
        T=10000,
        s=10,
        dim=(28, 28, 1),
        n_classes=10,
        patch_dim=(3, 3),
        feat_mins=X_train.min(),
        feat_maxs=X_train.max(),
        seed=10,
        device="cuda",
        n_threads=8,
    )

    train(tm, X_train, Y_train, X_test, Y_test, epochs=10)

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
