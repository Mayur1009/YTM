from lzma import LZMAFile
import pickle
from matplotlib.colors import Normalize
from medmnist.dataset import PneumoniaMNIST
import numpy as np
from matplotlib import pyplot as plt
from cutm import MultiClassTM
from tqdm import tqdm

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

def clause_positions_pos(lits):
    pos_len = lits.shape[1] // 2

    xpos = lits[:, :pos_len]
    ypos = lits[:, pos_len:]

    ret = np.zeros((lits.shape[0], 2))

    for ci in range(lits.shape[0]):
        nnz = np.argwhere(xpos[ci]).ravel()
        ret[ci, 0] = nnz[-1] + 1 if len(nnz) > 0 else 0

        nnz = np.argwhere(ypos[ci]).ravel()
        ret[ci, 1] = nnz[-1] + 1 if len(nnz) > 0 else 0

    return ret.astype(int)


def clause_positions_neg(lits):
    pos_len = lits.shape[1] // 2

    xpos = lits[:, :pos_len]
    ypos = lits[:, pos_len:]

    ret = np.zeros((lits.shape[0], 2))

    for ci in range(lits.shape[0]):
        nnz = np.argwhere(xpos[ci]).ravel()
        ret[ci, 0] = nnz[0]

        nnz = np.argwhere(ypos[ci]).ravel()
        ret[ci, 1] = nnz[0]

    return ret.astype(int)


def pixel_values_pos(a, levels):
    color_channels = a.shape[-1] // levels

    def f(b):
        ret = np.empty(color_channels)
        for i in range(color_channels):
            nz = np.argwhere(b[i * levels : (i + 1) * levels]).ravel()
            z = nz[-1] + 1 if len(nz) > 0 else 0
            ret[i] = z
        return ret

    img = np.apply_along_axis(f, -1, a) / (levels + 1)
    return img


def pixel_values_neg(a, levels):
    color_channels = a.shape[-1] // levels

    def f(b):
        ret = np.empty(color_channels)
        for i in range(color_channels):
            nz = np.argwhere(b[i * levels : (i + 1) * levels]).ravel()
            z = nz[0] if len(nz) > 0 else 0
            ret[i] = z
        return ret

    img = np.apply_along_axis(f, -1, a) / (levels + 1)
    return img


def global_transform(tm: MultiClassTM):
    img_shape = (28, 28)
    levels = 8
    color_channels = 1

    num_patch_x = tm.dim[0] - tm.patch_dim[0] + 1
    num_patch_y = tm.dim[1] - tm.patch_dim[1] + 1
    half_lits = tm.number_of_literals // 2

    literals = tm.get_literals()
    weights = tm.get_weights()
    patch_weights = tm.get_patch_weights().reshape((tm.number_of_clauses, num_patch_x, num_patch_y))

    patch_weights = patch_weights / patch_weights.max(axis=(1, 2), keepdims=True)
    patch_weights[patch_weights < 0.0] = 0

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
    positive_literals = pixel_values_pos(positive_literals, levels)
    negative_literals = pixel_values_neg(negative_literals, levels)

    positive_positions = clause_positions_pos(literals[:, : (num_patch_x - 1) + (num_patch_y - 1)])
    negative_positions = clause_positions_pos(
        literals[:, half_lits : half_lits + (num_patch_x - 1) + (num_patch_y - 1)]
    )

    transformed = np.zeros((tm.number_of_outputs, 2, *img_shape, color_channels))

    for c in range(tm.number_of_outputs):
        n = 0
        for ci in tqdm(range(tm.number_of_clauses), leave=False, desc=f"Output {c}"):
            if weights[ci, c] > 0:
                timg = np.zeros((2, *img_shape, color_channels))

                sx, sy = positive_positions[ci]
                for i in range(sx, num_patch_x):
                    for j in range(sy, num_patch_y):
                        if patch_weights[ci, i, j] > 0:
                            timg[0, i : i + tm.patch_dim[0], j : j + tm.patch_dim[1]] += (
                                patch_weights[ci, i, j] * positive_literals[ci]
                            )

                sx, sy = negative_positions[ci]
                for i in range(0, sx + 1):
                    for j in range(0, sy + 1):
                        if patch_weights[ci, i, j] > 0:
                            timg[1, i : i + tm.patch_dim[0], j : j + tm.patch_dim[1]] += (
                                patch_weights[ci, i, j] * negative_literals[ci]
                            )

                # transformed[c] = (n * transformed[c] + timg * weights[c, ci]) / (n + 1)
                transformed[c] += timg * weights[ci, c]
                n += 1

    return transformed

def plot_transformed(transformed):
    fig, axs = plt.subplots(1, 2, layout="compressed", figsize=(8, 4))

    for c in range(2):
        img = transformed[c, 0] - transformed[c, 1]

        if img.min() < 0:
            img[img < 0] /= (-1 * img[img < 0].min() + 1e-7)
        if img.max() > 0:
            img[img > 0] /= (img[img > 0].max() + 1e-7)

        img = Normalize(-1, 1)(img)
        axs[c].imshow(img, cmap=icefire)
        axs[c].axis("off")
        axs[c].set_title(f"Output {c}")

    return fig


if __name__ == "__main__":
    ch=8
    (xtrain, ytrain), (xval, yval), (xtest, ytest) = load_dataset(ch)

    tm = load_tm("./pneumonia_mnist.tm")
    transformed = global_transform(tm)

    fig = plot_transformed(transformed)
    plt.show()
