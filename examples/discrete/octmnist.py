import numpy as np
from medmnist import OCTMNIST
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

from ytm.discrete.classifier import MultiClassTM
from ytm.utils import Timer

N_CLASSES = 4
CLASS_NAMES = ["CNV", "DME", "DRUSEN", "NORMAL"]


def load_data(n_levels=8):


def cs_to_prob(cs, t):
    cs_clipped = np.clip(cs, -t, t)
    prob = (cs_clipped + t) / (2 * t)
    return prob / (prob.sum(axis=1, keepdims=True) + 1e-7)


def evaluate(tm, X, Y):
    pred, cs = tm.predict(X)
    prob = cs_to_prob(cs, tm.args.T_max)
    Y_bin = np.zeros((len(Y), N_CLASSES))
    Y_bin[np.arange(len(Y)), Y] = 1
    acc = accuracy_score(Y, pred)
    auc = roc_auc_score(Y_bin, prob, multi_class="ovr")
    return acc, auc, pred


def train(tm: MultiClassTM, X_train, Y_train, X_val, Y_val, X_test, Y_test, epochs=1):
    for epoch in range(epochs):
        fit_timer = Timer()
        with fit_timer:
            tm.fit(X_train, Y_train)

        train_acc, train_auc, _ = evaluate(tm, X_train, Y_train)
        val_acc, val_auc, _ = evaluate(tm, X_val, Y_val)
        test_acc, test_auc, test_pred = evaluate(tm, X_test, Y_test)

        print(
            f"Epoch {epoch + 1} | Fit: {fit_timer.elapsed:.2f}s | "
            f"Train Acc: {train_acc:.4f} AUC: {train_auc:.4f} | "
            f"Val Acc: {val_acc:.4f} AUC: {val_auc:.4f} | "
            f"Test Acc: {test_acc:.4f} AUC: {test_auc:.4f}"
        )
        print(f"Confusion Matrix:\n{confusion_matrix(Y_test, test_pred)}")

def preprocess(imgs):
    return np.asarray(n_levels * imgs.astype(np.float32) / 255.0, dtype=np.int32)

if __name__ == "__main__":
    n_levels = 8
    ds_train = OCTMNIST(split="train", download=True)
    ds_val = OCTMNIST(split="val", download=True)
    ds_test = OCTMNIST(split="test", download=True)


    X_train, Y_train = preprocess(ds_train.imgs), ds_train.labels.squeeze()
    X_val, Y_val = preprocess(ds_val.imgs), ds_val.labels.squeeze()
    X_test, Y_test = preprocess(ds_test.imgs), ds_test.labels.squeeze()
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    tm = MultiClassTM(
        n_clauses=1000,
        T=5000,
        s=10,
        dim=(28, 28, 1),
        n_classes=4,
        patch_dim=(9, 9),
        feat_mins=X_train.min(),
        feat_maxs=X_train.max(),
        seed=10,
        device="cuda",
    )

    train(tm, X_train, Y_train, X_val, Y_val, X_test, Y_test, epochs=100)
