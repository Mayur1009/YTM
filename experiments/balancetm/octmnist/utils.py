import numpy as np
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb
from cutm import MultiClassTM
from tm_utils import Timer
from tm_utils.metrics import multiclass_metrics
from medmnist.dataset import OCTMNIST
from tm_utils.binarizer import ThermometerBinarizer

# "label": {
#     "0": "choroidal neovascularization",
#     "1": "diabetic macular edema",
#     "2": "drusen",
#     "3": "normal",
# },

label_names = [
    "CNV",
    "DME",
    "drusen",
    "normal",
]


def load_dataset(ch=8):
    train = OCTMNIST(split="train", download=True)
    val = OCTMNIST(split="val", download=True)
    test = OCTMNIST(split="test", download=True)
    binarizer = ThermometerBinarizer(ch=ch)
    xtrain = binarizer.binarize_gray(train.imgs)
    xval = binarizer.binarize_gray(val.imgs)
    xtest = binarizer.binarize_gray(test.imgs)
    return (
        (xtrain, train.labels.squeeze()),
        (xval, val.labels.squeeze()),
        (xtest, test.labels.squeeze()),
    )


def train(tm: MultiClassTM, xtrain, ytrain, xval, yval, xtest, ytest, run, epochs: int, balance: bool):
    print("Encoding training, validation, and test data...")
    encoded_xtrain = tm.encode(xtrain.reshape((len(xtrain), -1)))
    encoded_xval = tm.encode(xval.reshape((len(xval), -1)))
    encoded_xtest = tm.encode(xtest.reshape((len(xtest), -1)))

    print(f"Training for {epochs} epochs")

    for epoch in (pbar := tqdm(range(epochs), desc="Epochs", dynamic_ncols=True)):
        iota = np.arange(len(ytrain))
        np.random.shuffle(iota)

        xtrain_suf = encoded_xtrain[iota]
        ytrain_suf = ytrain[iota]
        train_timer = Timer()
        with train_timer:
            tm.fit(xtrain_suf, ytrain_suf, is_X_encoded=True, balance=balance)

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

