import pickle
from lzma import LZMAFile

import numpy as np
from medmnist.dataset import PneumoniaMNIST
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

from cutm import BinaryTM
from ytm.utils import Binarizer, Timer


def load_dataset(ch=8):
    train = PneumoniaMNIST(split="train", download=True)
    val = PneumoniaMNIST(split="val", download=True)
    test = PneumoniaMNIST(split="test", download=True)

    b = Binarizer(ch)
    b.fit(train.imgs)

    xtrain = b.transform(train.imgs).reshape(len(train.imgs), -1).astype(np.uint32)
    xval = b.transform(val.imgs).reshape(len(val.imgs), -1).astype(np.uint32)
    xtest = b.transform(test.imgs).reshape(len(test.imgs), -1).astype(np.uint32)
    return (
        (xtrain, train.labels.squeeze()),
        (xval, val.labels.squeeze()),
        (xtest, test.labels.squeeze()),
    )


def balance(y):
    c0_inds = np.where(y == 0)[0]
    c1_inds = np.where(y == 1)[0]

    new_len = min(len(c0_inds), len(c1_inds))

    c0_inds = np.random.choice(c0_inds, new_len, replace=False)
    c1_inds = np.random.choice(c1_inds, new_len, replace=False)

    return np.concatenate([c0_inds, c1_inds])


def train(tm: BinaryTM, xtrain, ytrain, xval, yval, xtest, ytest, epochs=1):
    for epoch in range(epochs):
        balanced_indices = balance(ytrain)
        xtrain = xtrain[balanced_indices]
        ytrain = ytrain[balanced_indices]
        iota = np.arange(len(ytrain))
        np.random.shuffle(iota)
        xtrain = xtrain[iota]
        ytrain = ytrain[iota]

        train_timer = Timer()
        with train_timer:
            tm.fit(xtrain, ytrain)

        # Train
        preds_train, cs_train = tm.predict(xtrain)
        prob_train = (np.clip(cs_train.squeeze(), -tm.T, tm.T) + tm.T) / (2 * tm.T)
        train_acc = accuracy_score(ytrain, preds_train)
        auc_train = roc_auc_score(ytrain, prob_train)

        # Validation
        preds_val, cs_val = tm.predict(xval)
        prob_val = (np.clip(cs_val.squeeze(), -tm.T, tm.T) + tm.T) / (2 * tm.T)
        acc_val = accuracy_score(yval, preds_val)
        auc_val = roc_auc_score(yval, prob_val)

        # Test
        preds, cs_test = tm.predict(xtest)
        prob_test = (np.clip(cs_test.squeeze(), -tm.T, tm.T) + tm.T) / (2 * tm.T)
        acc_test = np.mean(preds == ytest)
        cm_test = confusion_matrix(ytest, preds)
        auc_test = roc_auc_score(ytest, prob_test)

        print(
            f"Epoch {epoch + 1} | Time: {train_timer.elapsed():.4f}s | Train Acc: {train_acc}| Train AUC: {auc_train} | Val Acc: {acc_val} | Val AUC: {auc_val} | Test Acc: {acc_test} | AUC: {auc_test}"
        )
        print(f"Confusion Matrix:\n{cm_test}")


if __name__ == "__main__":
    ch = 8
    (xtrain, ytrain), (xval, yval), (xtest, ytest) = load_dataset(ch)

    tm = BinaryTM(
        number_of_clauses_per_class=80,
        T=500,
        s=5,
        q=1,
        dim=(28, 28, ch),
        patch_dim=(10, 10),
        seed=10,
        block_size=4,
    )

    train(tm, xtrain, ytrain, xval, yval, xtest, ytest, epochs=30)

    with LZMAFile("pneumonia_mnist.tm", "wb") as f:
        pickle.dump(tm, f)
