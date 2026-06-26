import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full", auto_download=["html", "ipynb"])

with app.setup(hide_code=True):
    import lzma
    import pickle
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.colors import Normalize
    icefire = sns.color_palette("icefire", as_cmap=True)
    from datasets import load_dataset
    from pathlib import Path
    from medmnist import PneumoniaMNIST as PneumoniaMNISTDs
    from medmnist import OCTMNIST as OCTMNISTDs
    from ytm.discrete.classifier import MultiClassTM as DiscreteTM
    from ytm.guided.classifier import MultiClassTM as GuidedTM
    from ytm.discrete.classifier import BinaryTM as DiscreteBinaryTM
    from ytm.guided.classifier import BinaryTM as GuidedBinaryTM
    from ytm.discrete.interpret import wac as discrete_wac, wic as discrete_wic
    from ytm.guided.interpret import wac as guided_wac, wic as guided_wic

    MNIST_DIR = Path(__file__).parent.parent / "results" / "mnist"
    FMNIST_DIR = Path(__file__).parent.parent / "results" / "fmnist"
    PNEUMONIA_DIR = Path(__file__).parent.parent / "results" / "pneumoniamnist"
    OCTMNIST_DIR = Path(__file__).parent.parent / "results" / "octmnist"


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Introduction

    ### Normal TM
    - Weight update harcoded to be +1/-1, applied stocastically with update probability.
    - The hyperparam `T`, controls what would be the max value of the weights.
    - `T` kind of controls the rate of learning.
    - The default loss used: Absolute error, $update\_prob = (T - v) / (2T)$.
    - Loss functions define the learning objective of the model.
    - No way to use different loss functions.

    ### Proposed Guiding mechanism
    - Rather than using update probability to update the weights, directly use the error from the class sums.
    - Ability to use different loss functions.
    - For multiclass, use softmax to convert the class sum `v` into probability.
    - Use this to calculate the loss using Cross-entropy.
    - Then the, weight update can be calculated using gradient of the cross-entropy. Therefore, $w_j = w_j + \eta \cdot g_c \cdot + a_j$, where $a_j$ is the clause output
    - This probability takes into account the probability of all the classes.
    - Use it to update clauses as well.

    """)
    return

@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## MNIST
    """)
    return


@app.cell(hide_code=True)
def _():
    mnist_df = pd.concat(
        [
            pd.read_csv(_run_dir / "epochs.csv").assign(
                scheme=_run_dir.name.split("_seed")[0],
                seed=int(_run_dir.name.split("_seed")[1].split("_")[0]),
            )
            for _run_dir in sorted(MNIST_DIR.iterdir())
            if (_run_dir / "epochs.csv").exists()
        ],
        ignore_index=True,
    )

    mnist_long = mnist_df.melt(
        id_vars=["epoch", "scheme", "seed"],
        value_vars=["train_acc", "test_acc"],
        var_name="split",
        value_name="acc",
    )
    mnist_df

    return mnist_df, mnist_long

@app.cell(hide_code=True)
def _(mnist_long):
    with plt.style.context("seaborn-v0_8-whitegrid"):
        _fig, _ax = plt.subplots(1, 1, figsize=(8, 4), layout="compressed")
        sns.lineplot(data=mnist_long, x="epoch", y="acc", hue="scheme", style="split", ax=_ax)
        _ax.legend(loc="center left", frameon=True, bbox_to_anchor=(1, 0.5))
    _fig
    return




@app.cell(hide_code=True)
def _():
    ds = load_dataset("ylecun/mnist")
    X_train = np.asarray(np.array(ds["train"]["image"]) > 75, dtype=np.uint8)
    Y_train = np.array(ds["train"]["label"], dtype=np.uint8)
    X_test = np.asarray(np.array(ds["test"]["image"]) > 75, dtype=np.uint8)
    Y_test = np.array(ds["test"]["label"], dtype=np.uint8)
    return X_test, X_train, Y_test, Y_train


@app.cell(hide_code=True)
def _():
    def _load(scheme):
        _path = next(
            p for p in MNIST_DIR.iterdir()
            if p.name.startswith(scheme) and (p / "model.tm.lzma").exists()
        ) / "model.tm.lzma"
        with lzma.open(_path, "rb") as _f:
            return pickle.load(_f)

    mnist_discretetm = _load("discrete")
    mnist_guidedtm = _load("guided")
    return mnist_discretetm, mnist_guidedtm


@app.cell(hide_code=True)
def _(mnist_discretetm, mnist_guidedtm):
    _dw = mnist_discretetm.get_weights()
    _gw = mnist_guidedtm.get_weights()
    _n_classes = _dw.shape[0]
    _w_df = pd.concat(
        [pd.DataFrame({"weight": np.concatenate([_dw[c], _gw[c]]),
                       "scheme": ["Discrete"] * _dw.shape[1] + ["Guided"] * _gw.shape[1],
                       "class": c})
         for c in range(_n_classes)],
        ignore_index=True,
    )
    with plt.style.context("seaborn-v0_8-whitegrid"):
        _fig, _axes = plt.subplots(2, 5, figsize=(14, 6), layout="compressed")
        for _c, _ax in enumerate(_axes.flat):
            sns.histplot(data=_w_df[_w_df["class"] == _c], x="weight", hue="scheme", bins=50, kde=True, alpha=0.6, ax=_ax)
            _ax.set_title(f"Class {_c}")
            _ax.set_xlabel("")
            if _c != 0:
                _ax.get_legend().remove()
    _fig
    return


@app.cell(hide_code=True)
def _(X_test, Y_test, mnist_discretetm, mnist_guidedtm):
    def _norm(img):
        img = img.copy()
        if img.min() < 0:
            img[img < 0] = img[img < 0] / (-1 * img[img < 0].min() + 1e-7)
        if img.max() > 0:
            img[img > 0] = img[img > 0] / (img[img > 0].max() + 1e-7)
        return Normalize(-1, 1)(img)

    _idx = [np.where(Y_test == _c)[0][0] for _c in range(10)]
    _X = X_test[_idx]
    _disc_wac = discrete_wac(mnist_discretetm, _X).sum(axis=-1)
    _guided_wac = guided_wac(mnist_guidedtm, _X).sum(axis=-1)
    _fig, _axes = plt.subplots(3, 10, figsize=(2 * 10, 6))
    for _c in range(10):
        _axes[0, _c].imshow(_X[_c], cmap="gray")
        _axes[0, _c].set_title(f"Class {_c}")
        _axes[1, _c].imshow(_norm(_disc_wac[_c]), cmap=icefire)
        _axes[2, _c].imshow(_norm(_guided_wac[_c]), cmap=icefire)
    _axes[0, 0].set_ylabel("Original")
    _axes[1, 0].set_ylabel("Discrete")
    _axes[2, 0].set_ylabel("Guided")
    for _ax in _axes.flat:
        _ax.axis("off")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mnist_discretetm, mnist_guidedtm):
    def _norm(img):
        img = img.copy()
        if img.min() < 0:
            img[img < 0] = img[img < 0] / (-1 * img[img < 0].min() + 1e-7)
        if img.max() > 0:
            img[img > 0] = img[img > 0] / (img[img > 0].max() + 1e-7)
        return Normalize(-1, 1)(img)

    _disc_wic = discrete_wic(mnist_discretetm).sum(axis=-1)
    _guided_wic = guided_wic(mnist_guidedtm).sum(axis=-1)
    _n_classes = _disc_wic.shape[0]
    _fig, _axes = plt.subplots(2, _n_classes, figsize=(2 * _n_classes, 4))
    for _c in range(_n_classes):
        _axes[0, _c].imshow(_norm(_disc_wic[_c]), cmap=icefire)
        _axes[0, _c].set_title(f"Class {_c}")
        _axes[0, _c].axis("off")
        _axes[1, _c].imshow(_norm(_guided_wic[_c]), cmap=icefire)
        _axes[1, _c].axis("off")
    _axes[0, 0].set_ylabel("Discrete")
    _axes[1, 0].set_ylabel("Guided")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Fashion-MNIST
    """)
    return


@app.cell(hide_code=True)
def _():
    fmnist_df = pd.concat(
        [
            pd.read_csv(_run_dir / "epochs.csv").assign(
                scheme=_run_dir.name.split("_seed")[0],
                seed=int(_run_dir.name.split("_seed")[1].split("_")[0]),
            )
            for _run_dir in sorted(FMNIST_DIR.iterdir())
            if (_run_dir / "epochs.csv").exists()
        ],
        ignore_index=True,
    )

    fmnist_long = fmnist_df.melt(
        id_vars=["epoch", "scheme", "seed"],
        value_vars=["train_acc", "test_acc"],
        var_name="split",
        value_name="acc",
    )
    fmnist_df
    return fmnist_df, fmnist_long


@app.cell(hide_code=True)
def _(fmnist_long):
    with plt.style.context("seaborn-v0_8-whitegrid"):
        _fig, _ax = plt.subplots(1, 1, figsize=(8, 4), layout="compressed")
        sns.lineplot(data=fmnist_long, x="epoch", y="acc", hue="scheme", style="split", ax=_ax)
        _ax.legend(loc="center left", frameon=True, bbox_to_anchor=(1, 0.5))
    _fig
    return


@app.cell(hide_code=True)
def _():
    ds_fmnist = load_dataset("zalando-datasets/fashion_mnist")
    X_train_fmnist = np.asarray(8 * np.array(ds_fmnist["train"]["image"]).astype(np.float32) / 255.0, dtype=np.int32)
    Y_train_fmnist = np.array(ds_fmnist["train"]["label"], dtype=np.uint8)
    X_test_fmnist = np.asarray(8 * np.array(ds_fmnist["test"]["image"]).astype(np.float32) / 255.0, dtype=np.int32)
    Y_test_fmnist = np.array(ds_fmnist["test"]["label"], dtype=np.uint8)
    return X_test_fmnist, X_train_fmnist, Y_test_fmnist, Y_train_fmnist


@app.cell(hide_code=True)
def _():
    def _load(scheme):
        _path = next(
            p for p in FMNIST_DIR.iterdir()
            if p.name.startswith(scheme) and (p / "model.tm.lzma").exists()
        ) / "model.tm.lzma"
        with lzma.open(_path, "rb") as _f:
            return pickle.load(_f)

    fmnist_discretetm = _load("discrete")
    fmnist_guidedtm = _load("guided")
    return fmnist_discretetm, fmnist_guidedtm


@app.cell(hide_code=True)
def _(fmnist_discretetm, fmnist_guidedtm):
    _dw = fmnist_discretetm.get_weights()
    _gw = fmnist_guidedtm.get_weights()
    _n_classes = _dw.shape[0]
    _w_df = pd.concat(
        [pd.DataFrame({"weight": np.concatenate([_dw[c], _gw[c]]),
                       "scheme": ["Discrete"] * _dw.shape[1] + ["Guided"] * _gw.shape[1],
                       "class": c})
         for c in range(_n_classes)],
        ignore_index=True,
    )
    with plt.style.context("seaborn-v0_8-whitegrid"):
        _fig, _axes = plt.subplots(2, 5, figsize=(14, 6), layout="compressed")
        for _c, _ax in enumerate(_axes.flat):
            sns.histplot(data=_w_df[_w_df["class"] == _c], x="weight", hue="scheme", bins=50, kde=True, alpha=0.6, ax=_ax)
            _ax.set_title(f"Class {_c}")
            _ax.set_xlabel("")
            if _c != 0:
                _ax.get_legend().remove()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## OCTMNIST
    """)
    return


@app.cell(hide_code=True)
def _():
    octmnist_df = pd.concat(
        [
            pd.read_csv(_run_dir / "epochs.csv").assign(
                scheme=_run_dir.name.split("_seed")[0],
                seed=int(_run_dir.name.split("_seed")[1].split("_")[0]),
            )
            for _run_dir in sorted(OCTMNIST_DIR.iterdir())
            if (_run_dir / "epochs.csv").exists()
        ],
        ignore_index=True,
    )

    octmnist_long = octmnist_df.melt(
        id_vars=["epoch", "scheme", "seed"],
        value_vars=["train_acc", "test_acc"],
        var_name="split",
        value_name="acc",
    )
    octmnist_df
    return octmnist_df, octmnist_long


@app.cell(hide_code=True)
def _(octmnist_long):
    with plt.style.context("seaborn-v0_8-whitegrid"):
        _fig, _ax = plt.subplots(1, 1, figsize=(8, 4), layout="compressed")
        sns.lineplot(data=octmnist_long, x="epoch", y="acc", hue="scheme", style="split", ax=_ax)
        _ax.legend(loc="center left", frameon=True, bbox_to_anchor=(1, 0.5))
    _fig
    return


@app.cell(hide_code=True)
def _():
    ds_oct_train = OCTMNISTDs(split="train", download=True)
    ds_oct_test = OCTMNISTDs(split="test", download=True)
    X_train_octmnist = (8 * ds_oct_train.imgs.astype(np.float32) / 255.0).astype(np.int32)
    Y_train_octmnist = ds_oct_train.labels.squeeze()
    X_test_octmnist = (8 * ds_oct_test.imgs.astype(np.float32) / 255.0).astype(np.int32)
    Y_test_octmnist = ds_oct_test.labels.squeeze()
    return X_test_octmnist, X_train_octmnist, Y_test_octmnist, Y_train_octmnist


@app.cell(hide_code=True)
def _():
    def _load(scheme):
        _path = next(
            p for p in OCTMNIST_DIR.iterdir()
            if p.name.startswith(scheme) and (p / "model.tm.lzma").exists()
        ) / "model.tm.lzma"
        with lzma.open(_path, "rb") as _f:
            return pickle.load(_f)

    octmnist_discretetm = _load("discrete")
    octmnist_guidedtm = _load("guided")
    return octmnist_discretetm, octmnist_guidedtm


@app.cell(hide_code=True)
def _(octmnist_discretetm, octmnist_guidedtm):
    _dw = octmnist_discretetm.get_weights()
    _gw = octmnist_guidedtm.get_weights()
    _n_classes = _dw.shape[0]
    _w_df = pd.concat(
        [pd.DataFrame({"weight": np.concatenate([_dw[c], _gw[c]]),
                       "scheme": ["Discrete"] * _dw.shape[1] + ["Guided"] * _gw.shape[1],
                       "class": c})
         for c in range(_n_classes)],
        ignore_index=True,
    )
    with plt.style.context("seaborn-v0_8-whitegrid"):
        _fig, _axes = plt.subplots(1, 4, figsize=(14, 3), layout="compressed")
        for _c, _ax in enumerate(_axes.flat):
            sns.histplot(data=_w_df[_w_df["class"] == _c], x="weight", hue="scheme", bins=50, kde=True, alpha=0.6, ax=_ax)
            _ax.set_title(f"Class {_c}")
            _ax.set_xlabel("")
            if _c != 0:
                _ax.get_legend().remove()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## PneumoniaMNIST
    """)
    return


@app.cell(hide_code=True)
def _():
    pneumonia_df = pd.concat(
        [
            pd.read_csv(_run_dir / "epochs.csv").assign(
                scheme=_run_dir.name.split("_seed")[0],
                seed=int(_run_dir.name.split("_seed")[1].split("_")[0]),
            )
            for _run_dir in sorted(PNEUMONIA_DIR.iterdir())
            if (_run_dir / "epochs.csv").exists()
        ],
        ignore_index=True,
    )

    pneumonia_long = pneumonia_df.melt(
        id_vars=["epoch", "scheme", "seed"],
        value_vars=["train_acc", "test_acc"],
        var_name="split",
        value_name="acc",
    )
    pneumonia_df
    return pneumonia_df, pneumonia_long


@app.cell(hide_code=True)
def _(pneumonia_long):
    with plt.style.context("seaborn-v0_8-whitegrid"):
        _fig, _ax = plt.subplots(1, 1, figsize=(8, 4), layout="compressed")
        sns.lineplot(data=pneumonia_long, x="epoch", y="acc", hue="scheme", style="split", ax=_ax)
        _ax.legend(loc="center left", frameon=True, bbox_to_anchor=(1, 0.5))
    _fig
    return


@app.cell(hide_code=True)
def _():
    ds_pneumonia = PneumoniaMNISTDs(split="train", download=True)
    ds_pneumonia_test = PneumoniaMNISTDs(split="test", download=True)
    X_train_pneumonia = (8 * ds_pneumonia.imgs.astype(np.float32) / 255.0).astype(np.int32)
    Y_train_pneumonia = ds_pneumonia.labels.squeeze()
    X_test_pneumonia = (8 * ds_pneumonia_test.imgs.astype(np.float32) / 255.0).astype(np.int32)
    Y_test_pneumonia = ds_pneumonia_test.labels.squeeze()
    return X_test_pneumonia, X_train_pneumonia, Y_test_pneumonia, Y_train_pneumonia


@app.cell(hide_code=True)
def _():
    def _load(scheme):
        _path = next(
            p for p in PNEUMONIA_DIR.iterdir()
            if p.name.startswith(scheme) and (p / "model.tm.lzma").exists()
        ) / "model.tm.lzma"
        with lzma.open(_path, "rb") as _f:
            return pickle.load(_f)

    pneumonia_discretetm = _load("discrete")
    pneumonia_guidedtm = _load("guided")
    return pneumonia_discretetm, pneumonia_guidedtm


@app.cell(hide_code=True)
def _(pneumonia_discretetm, pneumonia_guidedtm):
    _w_df = pd.DataFrame({
        "weight": np.concatenate([pneumonia_discretetm.get_weights().flatten(), pneumonia_guidedtm.get_weights().flatten()]),
        "scheme": (["Discrete"] * pneumonia_discretetm.get_weights().size + ["Guided"] * pneumonia_guidedtm.get_weights().size),
    })
    with plt.style.context("seaborn-v0_8-whitegrid"):
        _fig, _ax = plt.subplots(figsize=(6, 6), layout="compressed")
        sns.histplot(data=_w_df, x="weight", hue="scheme", bins=100, kde=True, alpha=0.6, ax=_ax)
        _ax.set_title("Weight Distribution")
    _fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
