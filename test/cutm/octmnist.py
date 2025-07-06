from medmnist.dataset import OCTMNIST
import numpy as np
from tm_utils import Timer
from cutm import MultiClassTM
from tm_utils.binarizer import ThermometerBinarizer

from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score


def load_dataset(ch=8):
    train = OCTMNIST(split="train", download=True)
    val = OCTMNIST(split="val", download=True)
    test = OCTMNIST(split="test", download=True)
    binarizer = ThermometerBinarizer(ch = ch)
    xtrain = binarizer.binarize_gray(train.imgs)
    xval = binarizer.binarize_gray(val.imgs)
    xtest = binarizer.binarize_gray(test.imgs)
    return (
        (xtrain, train.labels.squeeze()),
        (xval, val.labels.squeeze()),
        (xtest, test.labels.squeeze()),
    )


def train(tm: MultiClassTM, xtrain, ytrain, xval, yval, xtest, ytest, epochs=1):
    encoded_xtrain = tm.encode(xtrain.reshape((len(xtrain), -1)))
    encoded_xval = tm.encode(xval.reshape((len(xval), -1)))
    encoded_xtest = tm.encode(xtest.reshape((len(xtest), -1)))
    for epoch in range(epochs):
        iota = np.arange(len(ytrain))
        np.random.shuffle(iota)
        xtrain_suf = encoded_xtrain[iota]
        ytrain_suf = ytrain[iota]
        train_timer = Timer()
        with train_timer:
            tm.fit(xtrain_suf, ytrain_suf, is_X_encoded=True)

        # Train
        preds_train, cs_train = tm.predict(encoded_xtrain, is_X_encoded=True)
        prob_train = (np.clip(cs_train, -tm.T, tm.T) + tm.T) / (2 * tm.T)
        prob_train = prob_train / (np.sum(prob_train, axis=1, keepdims=True) + 1e-7)
        train_acc = accuracy_score(ytrain, preds_train)
        ytrain_bin = np.zeros((len(ytrain), tm.number_of_outputs))
        ytrain_bin[np.arange(len(ytrain)), ytrain] = 1
        auc_train = roc_auc_score(ytrain_bin, prob_train)

        # Validation
        preds_val, cs_val = tm.predict(encoded_xval, is_X_encoded=True)
        prob_val = (np.clip(cs_val, -tm.T, tm.T) + tm.T) / (2 * tm.T)
        prob_val = prob_val / (np.sum(prob_val, axis=1, keepdims=True) + 1e-7)
        acc_val = accuracy_score(yval, preds_val)
        yval_bin = np.zeros((len(yval), tm.number_of_outputs))
        yval_bin[np.arange(len(yval)), yval] = 1
        auc_val = roc_auc_score(yval_bin, prob_val)

        # Test
        preds, cs_test = tm.predict(encoded_xtest, is_X_encoded=True)
        prob_test = (np.clip(cs_test, -tm.T, tm.T) + tm.T) / (2 * tm.T)
        prob_test = prob_test / (np.sum(prob_test, axis=1, keepdims=True) + 1e-7)
        acc_test = np.mean(preds == ytest)
        cm_test = confusion_matrix(ytest, preds)
        ytest_bin = np.zeros((len(ytest), tm.number_of_outputs))
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
        number_of_clauses=10000,
        T=25000,
        s=15,
        q=1,
        dim=(28, 28, ch),
        n_classes=4,
        patch_dim=(9, 9),
        seed=10,
        block_size=256,
    )

    train(tm, xtrain, ytrain, xval, yval, xtest, ytest, epochs=30)
