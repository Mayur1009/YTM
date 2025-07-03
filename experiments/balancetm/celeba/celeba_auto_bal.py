import os
import pickle
from datetime import datetime
from lzma import LZMAFile

import matplotlib.pyplot as plt
import numpy as np
from load import label_names, load_celeba, load_image_batch
from tqdm import tqdm

import wandb
from cutm import MultiOutputTM
from tm_utils import Timer
from tm_utils.metrics import multilabel_metrics

BATCH_SIZE = 90000
PROJECT = "IMBALANCE"
NAME = f"celeba_auto_bal_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
DIR  = f"./runs/celeba/{NAME}"
TAGS = ["celeba", "auto_balance"]
GROUP = "celeba"


def encode_data(tm, file, ids, ch):
    encoded_X = []
    for i in tqdm(range(0, len(ids), BATCH_SIZE), desc="Encoding", dynamic_ncols=True, leave=False):
        batch_ids = ids[i : i + BATCH_SIZE]
        batch_X = load_image_batch(file, batch_ids, ch)
        encoded_X.append(tm.encode(batch_X))
    return np.vstack(encoded_X)

def eval(tm: MultiOutputTM, encoded_X, Y):
    eval_time = 0

    preds = []
    css= []
    for i in tqdm(range(0, encoded_X.shape[0], BATCH_SIZE), desc="Evaluating", dynamic_ncols=True, leave=False):
        batch_X_encoded = encoded_X[i : i + BATCH_SIZE]
        eval_timer = Timer()
        with eval_timer:
            batch_preds, batch_css = tm.predict(batch_X_encoded, is_X_encoded=True, block_size=512)
        eval_time += eval_timer.elapsed()

        preds.append(batch_preds)
        css.append(batch_css)

    preds = np.vstack(preds)
    css = np.vstack(css)

    eval_timer = Timer()
    with eval_timer:
        preds, css = tm.predict(encoded_X, is_X_encoded=True, block_size=512)
    eval_time += eval_timer.elapsed()

    metrics = multilabel_metrics(Y, preds, css, label_names)
    for k, v in metrics.items():
        if "confusion_matrix" in k:
            metrics[k] = wandb.Image(v)

    return metrics, eval_time

def train(tm: MultiOutputTM, file, ids_train, Ytrain, ids_val, Yval, ids_test, Ytest, ch, run, epochs=1):
    print("Encoding training, validation, and test data...")
    encoded_X_train = encode_data(tm, file, ids_train, ch)
    encoded_X_val = encode_data(tm, file, ids_val, ch)
    encoded_X_test = encode_data(tm, file, ids_test, ch)

    print(f"Training for {epochs} epochs")

    for epoch in tqdm(range(epochs), desc="Epochs", dynamic_ncols=True):
        iota = np.arange(len(ids_train))
        np.random.shuffle(iota)
        train_fit_time = 0
        for i in tqdm(range(0, len(iota), BATCH_SIZE), desc="Training", dynamic_ncols=True, leave=False):
            indices = iota[i : i + BATCH_SIZE]
            batch_X_encoded = encoded_X_train[indices]
            batch_Y = Ytrain[indices]

            train_fit_timer = Timer()
            with train_fit_timer:
                tm.fit(batch_X_encoded, batch_Y, is_X_encoded=True)
            train_fit_time += train_fit_timer.elapsed()

        train_fit_timer = Timer()
        with train_fit_timer:
            tm.fit(encoded_X_train, Ytrain, is_X_encoded=True, balance=True)
        train_fit_time = train_fit_timer.elapsed()


        train_met, _ = eval(tm, encoded_X_train, Ytrain)
        val_met, _ = eval(tm, encoded_X_val, Yval)
        test_met, test_eval_time = eval(tm, encoded_X_test, Ytest)


        run.log(
            {
                "epoch": epoch + 1,
                "fit_time": train_fit_time,
                "test_time": test_eval_time,
                **{f"train/{k}": v for k, v in train_met.items()},
                **{f"val/{k}": v for k, v in val_met.items()},
                **{f"test/{k}": v for k, v in test_met.items()},
            }
        )

        plt.close("all")



if __name__ == "__main__":
    file, ids_train, Y_train = load_celeba("./data/CelebA", 0)
    _, ids_val, Y_val = load_celeba("./data/CelebA", 1)
    _, ids_test, Y_test = load_celeba("./data/CelebA", 2)
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
        "number_of_clauses": 100000,
        "T": 250000,
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
