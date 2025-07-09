import os
import pickle
from datetime import datetime
from lzma import LZMAFile

import wandb
from cutm import MultiClassTM
from utils import load_dataset, train, label_names

DATASET = "octmnist"
STRAT = "auto_bal"
NAME = f"{DATASET}_{STRAT}_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
DIR = f"./runs/{DATASET}/{NAME}"
PROJECT = "IMBALANCE"
GROUP = DATASET
TAGS = [DATASET, STRAT]

if __name__ == "__main__":
    ch = 8
    (xtrain, ytrain), (xval, yval), (xtest, ytest) = load_dataset(ch)

    config = {
        "dataset": DATASET,
        "label_names": label_names,
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

    train(tm, xtrain, ytrain, xval, yval, xtest, ytest, run, config["total_epochs"], True)

    os.makedirs(DIR, exist_ok=True)
    with open(f"{DIR}/config.tsv", "w") as f:
        for key, value in config.items():
            f.write(f"{key}\t{value}\n")

    with LZMAFile(f"{DIR}/model.tm", "wb") as f:
        pickle.dump(tm, f)

    wandb.finish()
