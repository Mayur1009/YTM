from medmnist.dataset import PneumoniaMNIST
import numpy as np
from tm_utils import Timer
from cutm import MultiClassTM
import pickle
from lzma import LZMAFile

from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score


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


def train(tm: MultiClassTM, xtrain, ytrain, xval, yval, xtest, ytest, epochs=1):
    for epoch in range(epochs):
        iota = np.arange(len(ytrain))
        np.random.shuffle(iota)
        xtrain = xtrain[iota]
        ytrain = ytrain[iota]

        train_timer = Timer()
        with train_timer:
            tm.fit(xtrain, ytrain)

        # Train
        preds_train, cs_train = tm.predict(xtrain)
        prob_train = (np.clip(cs_train, -tm.T, tm.T) + tm.T) / (2 * tm.T)
        prob_train = prob_train / (np.sum(prob_train, axis=1, keepdims=True) + 1e-7)
        train_acc = accuracy_score(ytrain, preds_train)
        ytrain_bin = np.zeros((len(ytrain), 2))
        ytrain_bin[np.arange(len(ytrain)), ytrain] = 1
        auc_train = roc_auc_score(ytrain_bin, prob_train)

        # Validation
        preds_val, cs_val = tm.predict(xval)
        prob_val = (np.clip(cs_val, -tm.T, tm.T) + tm.T) / (2 * tm.T)
        prob_val = prob_val / (np.sum(prob_val, axis=1, keepdims=True) + 1e-7)
        acc_val = accuracy_score(yval, preds_val)
        yval_bin = np.zeros((len(yval), 2))
        yval_bin[np.arange(len(yval)), yval] = 1
        auc_val = roc_auc_score(yval_bin, prob_val)

        # Test
        preds, cs_test = tm.predict(xtest)
        prob_test = (np.clip(cs_test, -tm.T, tm.T) + tm.T) / (2 * tm.T)
        prob_test = prob_test / (np.sum(prob_test, axis=1, keepdims=True) + 1e-7)
        acc_test = np.mean(preds == ytest)
        cm_test = confusion_matrix(ytest, preds)
        ytest_bin = np.zeros((len(ytest), 2))
        ytest_bin[np.arange(len(ytest)), ytest] = 1
        auc_test = roc_auc_score(ytest_bin, prob_test)

        print(
            f"Epoch {epoch + 1}|Time: {train_timer.elapsed():.4f}s|Train Acc: {train_acc:.4f}|Train AUC: {auc_train:.4f}|Val Acc: {acc_val:.4f}|Val AUC: {auc_val:.4f}|Test Acc: {acc_test:.4f}|AUC: {auc_test:.4f}"
        )
        # print(f"Confusion Matrix:\n{cm_test}")


if __name__ == "__main__":
    ch = 8
    (xtrain, ytrain), (xval, yval), (xtest, ytest) = load_dataset(ch)

    tm = MultiClassTM(
        number_of_clauses=80,
        T=500,
        s=5,
        q=1,
        dim=(28, 28, ch),
        n_classes=2,
        patch_dim=(10, 10),
        seed=10,
        block_size=4,
    )

    train(tm, xtrain, ytrain, xval, yval, xtest, ytest, epochs=30)

    with LZMAFile("pneumonia_mnist.tm", "wb") as f:
        pickle.dump(tm, f)

