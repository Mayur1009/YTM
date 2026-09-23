import pickle
from lzma import LZMAFile

import numpy as np
import seaborn as sns
from keras.datasets import mnist
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize

from ytm._o.tm import MultiClassTM
from ytm.utils import Timer

icefire = sns.color_palette("icefire", as_cmap=True)


def binarize_images(imgs, ch=8):
    out = np.zeros((*imgs.shape, ch), dtype=np.uint32)
    for j in range(ch):
        t1 = (j + 1) * 255 / (ch + 1)
        out[:, :, :, j] = (imgs >= t1) & 1

    return out.reshape(imgs.shape[0], -1).astype(np.uint32)


def load_tm(path) -> MultiClassTM:
    with LZMAFile(path, "rb") as f:
        tm: MultiClassTM = pickle.load(f)
    return tm


def transform_Xs(tm: MultiClassTM, Xs: np.ndarray):
    num_samples = Xs.shape[0]

    img_shape = (28, 28)

    num_patch_x = tm.dim[0] - tm.args.patch_dim[0] + 1
    num_patch_y = tm.dim[1] - tm.args.patch_dim[1] + 1
    half_lits = tm.dev.n_literals // 2

    co_patchwise = tm.transform_patchwise(Xs.reshape(num_samples, -1)).reshape(
        num_samples,
        tm.n_clauses,
        num_patch_x,
        num_patch_y,
    )

    literals = (tm.get_ta_states() >= tm.args.include_state).reshape(tm.n_clauses, -1)
    weights = tm.get_weights()

    positive_literals = (
        literals[:, (num_patch_x - 1) + (num_patch_y - 1) : half_lits]
        .reshape((tm.n_clauses, *tm.args.patch_dim))
        .astype(np.float32)
    )
    negative_literals = (
        literals[:, half_lits + (num_patch_x - 1) + (num_patch_y - 1) :]
        .reshape((tm.n_clauses, *tm.args.patch_dim))
        .astype(np.float32)
    )

    transformed = np.zeros((num_samples, 2, *img_shape))

    for e in range(num_samples):
        for ci in range(tm.n_clauses):
            if weights[e, ci] > 0:
                clause_active_pos = np.argwhere(co_patchwise[e, ci] > 0)

                timg = np.zeros((2, *img_shape))

                for m, n in clause_active_pos:
                    timg[0, m : m + tm.args.patch_dim[0], n : n + tm.args.patch_dim[1]] += positive_literals[ci]
                    timg[1, m : m + tm.args.patch_dim[0], n : n + tm.args.patch_dim[1]] += negative_literals[ci]

                transformed[e] += timg * weights[e, ci]

    return transformed


def plot_transformed(Xs, Ys, transformed):
    fig, axd = plt.subplot_mosaic(
        [
            ["X0", "X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9"],
            ["T0", "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9"],
        ],
        layout="compressed",
        figsize=(7, 6),
        sharex=True,
        sharey=True,
    )

    for i in range(10):
        axd[f"X{i}"].imshow(Xs[i].reshape(28, 28), cmap="gray")
        axd[f"X{i}"].axis("off")

        img = transformed[i, 0] - transformed[i, 1]

        if img.min() < 0:
            img[img < 0] = img[img < 0] / (-1 * img[img < 0].min() + 1e-7)
        if img.max() > 0:
            img[img > 0] = img[img > 0] / (img[img > 0].max() + 1e-7)

        img = Normalize(-1, 1)(img)

        axd[f"T{i}"].imshow(img, cmap=icefire)
        axd[f"T{i}"].axis("off")

    # cbar = fig.colorbar(
    #     axd["T0"].images[0],
    #     ax=axd["EMPTY"],
    #     fraction=1,
    #     pad=0,
    # )

    return fig

def train(tm: MultiClassTM, X_train, Y_train, X_test, Y_test, epochs=1):
    encoded_X_train = tm.encode(X_train)
    encoded_X_test = tm.encode(X_test)
    for epoch in range(epochs):
        train_fit_timer = Timer()
        with train_fit_timer:
            tm.fit(encoded_X_train, Y_train, is_X_encoded=True)

        test_timer = Timer()
        with test_timer:
            test_pred, _ = tm.predict(encoded_X_test, is_X_encoded=True)

        train_timer = Timer()
        with train_timer:
            train_pred, _ = tm.predict(encoded_X_train, is_X_encoded=True)

        test_acc = np.mean(Y_test == test_pred)
        train_acc = np.mean(Y_train == train_pred)
        print(
            f"Epoch {epoch + 1} | Acc> Train: {train_acc * 100:.4f}% Test: {test_acc * 100:.4f}% | Time> Fit: {train_fit_timer.elapsed:.4f}s Infer Train: {train_timer.elapsed:.4f}s Infer Test: {test_timer.elapsed:.4f}s"
        )


if __name__ == "__main__":
    ch = 8
    (X_train, Y_train_org), (X_test, Y_test_org) = mnist.load_data()

    X_train = np.where(X_train.reshape((X_train.shape[0], 28 * 28)) > 75, 1, 0)
    X_test = np.where(X_test.reshape((X_test.shape[0], 28 * 28)) > 75, 1, 0)
    X_train = np.asarray(X_train, dtype=np.int8)
    X_test = np.asarray(X_test, dtype=np.int8)

    Y_train, Y_test = Y_train_org, Y_test_org

    # tm = load_tm("./mnist_conv.tm")
    tm = MultiClassTM(
        n_clauses=500,
        T=1000,
        s=10,
        dim=(28, 28, 1),
        n_classes=10,
        patch_dim=(10, 10),
        seed=10,
        device="cpu",
        n_threads=8,
    )
    train(tm, X_train, Y_train, X_test, Y_test, epochs=2)

    index_per_class = []
    for i in range(10):
        index_per_class.append(np.argwhere(Y_test == i).ravel()[0])

    Xs = X_test[index_per_class]
    ys = Y_test[index_per_class]

    transformed = transform_Xs(tm, Xs)

    fig = plot_transformed(Xs, ys, transformed)
    plt.show()
