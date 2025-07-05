import os
import pickle
from datetime import datetime
from lzma import LZMAFile

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from utils import load_celeba, encode_data, subset1 as label_names
from tqdm import tqdm

import wandb
from cutm import MultiOutputTM
from tm_utils import Timer
from tm_utils.metrics import multilabel_metrics

PROJECT = "IMBALANCE"
NAME = f"celeba_auto_bal_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
DIR = f"./runs/celeba/{NAME}"
TAGS = ["celeba", "auto_balance", "subset1"]
GROUP = "celeba"

plt.rcParams.update({"figure.max_open_warning": 0})


def train(tm: MultiOutputTM, file, ids_train, Ytrain, ids_val, Yval, ids_test, Ytest, ch, run, epochs=1):
    print("Encoding training, validation, and test data...")
    encoded_X_train = encode_data(tm, file, ids_train, ch)
    encoded_X_val = encode_data(tm, file, ids_val, ch)
    encoded_X_test = encode_data(tm, file, ids_test, ch)

    print(f"Training for {epochs} epochs")

    for epoch in (pbar := tqdm(range(epochs), desc="Epochs", dynamic_ncols=True)):
        iota = np.arange(len(ids_train))
        np.random.shuffle(iota)

        Xtrain_suf = encoded_X_train[iota]
        Ytrain_suf = Ytrain[iota]

        train_fit_timer = Timer()
        with train_fit_timer:
            tm.fit(Xtrain_suf, Ytrain_suf, is_X_encoded=True, balance=True)
        train_fit_time = train_fit_timer.elapsed()

        train_preds, train_cs = tm.predict(encoded_X_train, is_X_encoded=True, block_size=1024)
        train_prob = (np.clip(train_cs, -tm.T, tm.T) + tm.T) / (2 * tm.T)
        train_metrics = multilabel_metrics(Ytrain, train_preds, train_prob, label_names)

        val_preds, val_cs = tm.predict(encoded_X_val, is_X_encoded=True, block_size=1024)
        val_prob = (np.clip(val_cs, -tm.T, tm.T) + tm.T) / (2 * tm.T)
        val_metrics = multilabel_metrics(Yval, val_preds, val_prob, label_names)

        test_timer = Timer()
        with test_timer:
            test_preds, test_cs = tm.predict(encoded_X_test, is_X_encoded=True, block_size=1024)
        test_time = test_timer.elapsed()
        test_prob = (np.clip(test_cs, -tm.T, tm.T) + tm.T) / (2 * tm.T)
        test_metrics = multilabel_metrics(Ytest, test_preds, test_prob, label_names)

        log = {
            "epoch": epoch + 1,
            "fit_time": train_fit_time,
            "test_time": test_time,
            **{f"train/{k}": v for k, v in train_metrics.items()},
            **{f"val/{k}": v for k, v in val_metrics.items()},
            **{f"test/{k}": v for k, v in test_metrics.items()},
        }

        for k, v in log.items():
            if isinstance(v, Figure):
                log[k] = wandb.Image(v)

        run.log(log)

        plt.close("all")

        pbar.set_postfix_str(
            f"train_f1: {train_metrics['f1/macro']:.4f}, val_f1: {val_metrics['f1/macro']:.4f}, test_f1: {test_metrics['f1/macro']:.4f}"
        )


if __name__ == "__main__":
    file, ids_train, Y_train = load_celeba("./data/CelebA", 0, label_names)
    _, ids_val, Y_val = load_celeba("./data/CelebA", 1, label_names)
    _, ids_test, Y_test = load_celeba("./data/CelebA", 2, label_names)
    ch = 8

    config = {
        "dataset": "PneumoniaMNIST",
        "binarization": "thermometer",
        "binarization_channels": ch,
        "training_samples": Y_train.shape[0],
        "validation_samples": Y_val.shape[0],
        "test_samples": Y_test.shape[0],
        "n_classes": len(label_names),
        "label_names": label_names,
        "number_of_clauses": 30000,
        "T": 60000,
        "s": 25,
        "q": 25,
        "dim": (64, 64, ch * 3),
        "patch_dim": (3, 3),
        "seed": 10,
        "total_epochs": 50,
    }

    run = wandb.init(
        project=PROJECT,
        name=NAME,
        tags=TAGS,
        group=GROUP,
        save_code=True,
        config=config,
    )
    run.log_code(
        include_fn=lambda path: ".pixi" not in path
        and (path.endswith(".py") or path.endswith(".cu") or path.endswith(".toml"))
    )

    tm = MultiOutputTM(
        number_of_clauses=config["number_of_clauses"],
        T=config["T"],
        s=config["s"],
        q=config["q"],
        dim=config["dim"],
        n_classes=config["n_classes"],
        patch_dim=config["patch_dim"],
        seed=config["seed"],
        block_size=128,
    )

    train(tm, file, ids_train, Y_train, ids_val, Y_val, ids_test, Y_test, ch, run, epochs=config["total_epochs"])

    os.makedirs(DIR, exist_ok=True)
    with open(f"{DIR}/config.tsv", "w") as f:
        for key, value in config.items():
            f.write(f"{key}\t{value}\n")

    with LZMAFile(f"{DIR}/model.tm", "wb") as f:
        pickle.dump(tm, f)

    wandb.finish()
