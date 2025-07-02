import os
import pickle
from datetime import datetime
from lzma import LZMAFile

import matplotlib.pyplot as plt
import numpy as np
from medmnist.dataset import PneumoniaMNIST
from tqdm import tqdm

import wandb
from cutm import MultiClassTM
from tm_utils import Timer
from tm_utils.metrics import multiclass_metrics

NAME = f"pneumonia_mnist_no_balancing_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
TAGS = ["pneumonia_mnist", "default"]


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


def train(tm: MultiClassTM, xtrain, ytrain, xval, yval, xtest, ytest, run, epochs=1):
    for epoch in tqdm(range(epochs), dynamic_ncols=True, desc="Training Epochs"):
        log: dict = {
            "epoch": epoch + 1,
        }
        iota = np.arange(len(ytrain))
        np.random.shuffle(iota)
        xtrain = xtrain[iota]
        ytrain = ytrain[iota]

        # Train
        train_timer = Timer()
        with train_timer:
            tm.fit(xtrain, ytrain)

        log["fit_time"] = train_timer.elapsed()

        preds_train, cs_train = tm.predict(xtrain)
        prob_train = (np.clip(cs_train, -tm.T, tm.T) + tm.T) / (2 * tm.T)
        prob_train = prob_train / (np.sum(prob_train, axis=1, keepdims=True) + 1e-7)
        met_train = multiclass_metrics(ytrain, preds_train, prob_train, ["0", "1"])
        met_train["confusion_matrix"] = wandb.Image(met_train["confusion_matrix"])

        # Validation
        preds_val, cs_val = tm.predict(xval)
        prob_val = (np.clip(cs_val, -tm.T, tm.T) + tm.T) / (2 * tm.T)
        prob_val = prob_val / (np.sum(prob_val, axis=1, keepdims=True) + 1e-7)
        met_val = multiclass_metrics(yval, preds_val, prob_val, ["0", "1"])
        met_val["confusion_matrix"] = wandb.Image(met_val["confusion_matrix"])

        # Test
        test_timer = Timer()
        with test_timer:
            preds, cs_test = tm.predict(xtest)
        log["test_time"] = test_timer.elapsed()

        prob_test = (np.clip(cs_test, -tm.T, tm.T) + tm.T) / (2 * tm.T)
        prob_test = prob_test / (np.sum(prob_test, axis=1, keepdims=True) + 1e-7)
        met_test = multiclass_metrics(ytest, preds, prob_test, ["0", "1"])
        met_test["confusion_matrix"] = wandb.Image(met_test["confusion_matrix"])

        run.log(
            {
                **log,
                **{f"train/{k}": v for k, v in met_train.items()},
                **{f"val/{k}": v for k, v in met_val.items()},
                **{f"test/{k}": v for k, v in met_test.items()},
                "epoch": epoch + 1,
            }
        )
        plt.close("all")


if __name__ == "__main__":
    dir = f"./runs/pneumonia_mnist/{NAME}"
    ch = 8
    (xtrain, ytrain), (xval, yval), (xtest, ytest) = load_dataset(ch)

    config = {
        "dataset": "PneumoniaMNIST",
        "binarization": "thermometer",
        "binarization_channels": ch,
        "training_samples": xtrain.shape[0],
        "validation_samples": xval.shape[0],
        "test_samples": xtest.shape[0],
        "n_classes": 2,
        "number_of_clauses": 80,
        "T": 8000,
        "s": 5,
        "q": 1,
        "dim": (28, 28, ch),
        "patch_dim": (10, 10),
        "seed": 10,
        "total_epochs": 100,
    }

    run = wandb.init(
        project="IMBALANCE",
        name=NAME,
        tags=TAGS,
        group="pneumonia_mnist",
        save_code=True,
        config=config,
    )
    run.log_code(
        include_fn=lambda path: ".pixi" not in path
        and (path.endswith(".py") or path.endswith(".cu") or path.endswith(".toml"))
    )

    tm = MultiClassTM(
        number_of_clauses=config["number_of_clauses"],
        T=config["T"],
        s=config["s"],
        q=config["q"],
        dim=config["dim"],
        n_classes=config["n_classes"],
        patch_dim=config["patch_dim"],
        seed=config["seed"],
        block_size=4,
    )
    train(tm, xtrain, ytrain, xval, yval, xtest, ytest, run, epochs=config["total_epochs"])

    os.makedirs(dir, exist_ok=True)
    with open(f"{dir}/config.tsv", "w") as f:
        for key, value in config.items():
            f.write(f"{key}\t{value}\n")

    with LZMAFile(f"{dir}/model.tm", "wb") as f:
        pickle.dump(tm, f)

    wandb.finish()
