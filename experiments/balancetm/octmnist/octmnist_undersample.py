import os
import pickle
from datetime import datetime
from lzma import LZMAFile

import wandb
from cutm import MultiClassTM
from utils import load_dataset

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from tqdm import tqdm
from tm_utils.metrics import multiclass_metrics
from tm_utils import Timer
from utils import label_names
np.random.seed(10)

DATASET = "octmnist"
STRAT = "undersample"
NAME = f"{DATASET}_{STRAT}_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
DIR = f"./runs/{DATASET}/{NAME}"
PROJECT = "IMBALANCE"
GROUP = DATASET
TAGS = [DATASET, STRAT]

def balance_Y(Y, n_class):
    counts = np.bincount(Y, minlength=n_class)
    min_count = np.min(counts)

    selected_indices = []
    for c in range(n_class):
        indices = np.where(Y == c)[0]
        si = np.random.choice(indices, size=min_count, replace=True)
        selected_indices.extend(si)

    return selected_indices

def train(tm: MultiClassTM, xtrain, ytrain, xval, yval, xtest, ytest, run, epochs: int):
    print("Encoding training, validation, and test data...")
    encoded_xtrain = tm.encode(xtrain.reshape((len(xtrain), -1)))
    encoded_xval = tm.encode(xval.reshape((len(xval), -1)))
    encoded_xtest = tm.encode(xtest.reshape((len(xtest), -1)))

    print(f"Training for {epochs} epochs")

    for epoch in (pbar := tqdm(range(epochs), desc="Epochs", dynamic_ncols=True)):
        sel_inds = balance_Y(ytrain, len(label_names))
        encoded_xtrain_bal = encoded_xtrain[sel_inds]
        ytrain_bal = ytrain[sel_inds]

        iota = np.arange(len(ytrain_bal))
        np.random.shuffle(iota)

        xtrain_suf = encoded_xtrain_bal[iota]
        ytrain_suf = ytrain_bal[iota]
        train_timer = Timer()
        with train_timer:
            tm.fit(xtrain_suf, ytrain_suf, is_X_encoded=True)

        # Train
        preds_train, cs_train = tm.predict(encoded_xtrain, is_X_encoded=True)
        prob_train = (np.clip(cs_train, -tm.T, tm.T) + tm.T) / (2 * tm.T)
        train_met = multiclass_metrics(ytrain, preds_train, prob_train, label_names)

        # Validation
        preds_val, cs_val = tm.predict(encoded_xval, is_X_encoded=True)
        prob_val = (np.clip(cs_val, -tm.T, tm.T) + tm.T) / (2 * tm.T)
        val_met = multiclass_metrics(yval, preds_val, prob_val, label_names)

        # Test
        test_timer = Timer()
        with test_timer:
            preds, cs_test = tm.predict(encoded_xtest, is_X_encoded=True)
        prob_test = (np.clip(cs_test, -tm.T, tm.T) + tm.T) / (2 * tm.T)
        test_met = multiclass_metrics(ytest, preds, prob_test, label_names)

        log = {
            "epoch": epoch + 1,
            "fit_time": train_timer.elapsed(),
            "test_time": test_timer.elapsed(),
            **{f"train/{k}": v for k, v in train_met.items()},
            **{f"val/{k}": v for k, v in val_met.items()},
            **{f"test/{k}": v for k, v in test_met.items()},
        }

        for k, v in log.items():
            if isinstance(v, Figure):
                log[k] = wandb.Image(v)

        run.log(log)

        plt.close("all")

        pbar.set_postfix_str(
            f"train_acc: {train_met['accuracy']:.4f}, val_acc: {val_met['accuracy']:.4f}, test_acc: {test_met['accuracy']:.4f}"
        )

if __name__ == "__main__":
    ch = 8
    (xtrain, ytrain), (xval, yval), (xtest, ytest) = load_dataset(ch)

    config = {
        "dataset": DATASET,
        "binarization": "thermometer",
        "binarization_channels": ch,
        "training_samples": xtrain.shape[0],
        "validation_samples": xval.shape[0],
        "test_samples": xtest.shape[0],
        "n_classes": 4,
        "number_of_clauses": 18000,
        "T": 50000,
        "s": 25,
        "q": 1,
        "dim": (28, 28, ch),
        "patch_dim": (7, 7),
        "seed": 10,
        "total_epochs": 100,
        "strategy": STRAT,
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

    tm = MultiClassTM(
        number_of_clauses=config["number_of_clauses"],
        T=config["T"],
        s=config["s"],
        q=config["q"],
        dim=config["dim"],
        n_classes=config["n_classes"],
        patch_dim=config["patch_dim"],
        seed=config["seed"],
        block_size=256,
    )

    train(tm, xtrain, ytrain, xval, yval, xtest, ytest, run, config["total_epochs"])

    os.makedirs(DIR, exist_ok=True)
    with open(f"{DIR}/config.tsv", "w") as f:
        for key, value in config.items():
            f.write(f"{key}\t{value}\n")

    with LZMAFile(f"{DIR}/model.tm", "wb") as f:
        pickle.dump(tm, f)

    wandb.finish()
