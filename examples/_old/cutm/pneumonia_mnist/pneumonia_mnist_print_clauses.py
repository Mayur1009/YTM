from lzma import LZMAFile
import pickle
from matplotlib.colors import Normalize
from medmnist.dataset import PneumoniaMNIST
import numpy as np
from matplotlib import pyplot as plt
from cutm import MultiClassTM

import seaborn as sns

icefire = sns.color_palette("icefire", as_cmap=True)


def binarize_images(imgs, ch=8):
    out = np.zeros((*imgs.shape, ch), dtype=np.uint32)
    for j in range(ch):
        t1 = (j + 1) * 255 / (ch + 1)
        out[:, :, :, j] = (imgs >= t1) & 1

    return out.reshape(imgs.shape[0], -1).astype(np.uint32)


def load_dataset(ch=8):
    train = PneumoniaMNIST(split="train", download=True)
    val = PneumoniaMNIST(split="val", download=True)
    test = PneumoniaMNIST(split="test", download=True)
    xtrain = binarize_images(train.imgs, ch)
    xval = binarize_images(val.imgs, ch)
    xtest = binarize_images(test.imgs, ch)
    return (
        (xtrain, train.labels.squeeze()),
        (xval, val.labels.squeeze()),
        (xtest, test.labels.squeeze()),
    )


def load_tm(path):
    with LZMAFile(path, "rb") as f:
        tm: MultiClassTM = pickle.load(f)

    return tm


def unbinarize(a, color_channels):
    def f(b):
        ret = np.empty(color_channels)
        t = len(b) // color_channels
        for i in range(color_channels):
            nz = np.argwhere(b[i * t : (i + 1) * t]).ravel()
            z = nz[-1] + 1 if len(nz) > 0 else 0
            ret[i] = z
        return ret

    img = np.apply_along_axis(f, -1, a)
    return img


def transform_Xs(tm: MultiClassTM, Xs: np.ndarray):
    num_samples = Xs.shape[0]
    levels = Xs.shape[3]

    img_shape = (*Xs.shape[1:-1], Xs.shape[3] // levels)

    num_patch_x = tm.dim[0] - tm.patch_dim[0] + 1
    num_patch_y = tm.dim[1] - tm.patch_dim[1] + 1
    half_lits = tm.number_of_literals // 2

    co_patchwise = (
        tm.transform_patchwise(Xs.reshape(num_samples, -1))
        .reshape(
            num_samples,
            tm.number_of_clauses,
            num_patch_x,
            num_patch_y,
        )
    )

    literals = tm.get_literals()
    weights = tm.get_weights()

    positive_literals = (
        literals[:, (num_patch_x - 1) + (num_patch_y - 1) : half_lits]
        .reshape((tm.number_of_clauses, *tm.patch_dim, levels))
        .astype(np.float32)
    )
    negative_literals = (
        literals[:, half_lits + (num_patch_x - 1) + (num_patch_y - 1) :]
        .reshape((tm.number_of_clauses, *tm.patch_dim, levels))
        .astype(np.float32)
    )

    positive_literals = unbinarize(positive_literals, 1) / levels
    negative_literals = unbinarize(negative_literals, 1) / levels

    transformed = np.zeros((num_samples, 2, *img_shape))

    for e in range(num_samples):
        for ci in range(tm.number_of_clauses):
            if weights[ci, e] > 0:
                clause_active_pos = np.argwhere(co_patchwise[e, ci] > 0)

                timg = np.zeros((2, *img_shape))

                for m, n in clause_active_pos:
                    timg[0, m : m + tm.patch_dim[0], n : n + tm.patch_dim[1]] += positive_literals[ci]
                    timg[1, m : m + tm.patch_dim[0], n : n + tm.patch_dim[1]] += negative_literals[ci]

                transformed[e] += timg * weights[ci, e]

    return transformed


def plot_transformed(Xs, Ys, transformed):
    num_samples = Xs.shape[0]
    assert num_samples == 2

    fig, axd = plt.subplot_mosaic(
        [
            ["X0", "T0", "EMPTY"],
            ["X1", "T1", "EMPTY"],
        ],
        empty_sentinel="EMPTY",
        layout="compressed",
        figsize=(7, 6),
        sharex=True,
        sharey=True,
        width_ratios=[1, 1, 0.1],
    )

    for i in range(2):
        axd[f"X{i}"].imshow(unbinarize(Xs[i], 1), cmap="gray")
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


if __name__ == "__main__":
    ch=8
    (xtrain, ytrain), (xval, yval), (xtest, ytest) = load_dataset(ch)

    tm = load_tm("./pneumonia_mnist.tm")

    inds0 = np.where(ytest == 0)[0]
    inds1 = np.where(ytest == 1)[0]

    rng = np.random.default_rng(1)
    i1 = rng.choice(inds0, 1, replace=False)[0]
    i2 = rng.choice(inds1, 1, replace=False)[0]

    Xs = np.array([xtest[i1], xtest[i2]]).reshape((2, 28, 28, ch))
    ys = np.array([ytest[i1], ytest[i2]])

    transformed = transform_Xs(tm, Xs)

    fig = plot_transformed(Xs, ys, transformed)
    plt.show()


