import argparse

import numpy as np
from medmnist import OCTMNIST
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

from ytm.discrete.classifier import MultiClassTM
from ytm.utils import Timer, print_table

N_CLASSES = 4
CLASS_NAMES = ["CNV", "DME", "DRUSEN", "NORMAL"]


def preprocess(imgs, n_levels=8):
    return np.asarray(n_levels * imgs.astype(np.float32) / 255.0, dtype=np.int32)


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

        print_table(
            f"Epoch {epoch + 1}/{epochs}",
            {
                "Train": {"Acc": f"{train_acc:.4f}", "AUC": f"{train_auc:.4f}", "Fit Time": f"{fit_timer.elapsed:.2f}s"},
                "Val":   {"Acc": f"{val_acc:.4f}", "AUC": f"{val_auc:.4f}"},
                "Test":  {"Acc": f"{test_acc:.4f}", "AUC": f"{test_auc:.4f}"},
            },
        )
        print(f"Confusion Matrix:\n{confusion_matrix(Y_test, test_pred)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clauses", type=int, default=1000)
    parser.add_argument("--T", type=int, default=5000)
    parser.add_argument("--s", type=float, default=10.0)
    parser.add_argument("--patch", type=int, nargs=2, default=[9, 9], metavar=("H", "W"))
    parser.add_argument("--n_levels", type=int, default=8)
    parser.add_argument("--seed", type=lambda x: None if x == "None" else int(x), default=10)
    parser.add_argument("--coalesced", type=int, choices=[0, 1], default=1)
    parser.add_argument("--n_threads", type=int, default=8)
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    ds_train = OCTMNIST(split="train", download=True)
    ds_val = OCTMNIST(split="val", download=True)
    ds_test = OCTMNIST(split="test", download=True)

    X_train, Y_train = preprocess(ds_train.imgs, args.n_levels), ds_train.labels.squeeze()
    X_val, Y_val = preprocess(ds_val.imgs, args.n_levels), ds_val.labels.squeeze()
    X_test, Y_test = preprocess(ds_test.imgs, args.n_levels), ds_test.labels.squeeze()
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    tm = MultiClassTM(
        n_clauses=args.n_clauses,
        T=args.T,
        s=args.s,
        dim=(28, 28, 1),
        n_classes=N_CLASSES,
        patch_dim=tuple(args.patch),
        feat_mins=X_train.min(),
        feat_maxs=X_train.max(),
        seed=args.seed,
        coalesced=True if args.coalesced == 1 else False,
        device=args.device,
        n_threads=args.n_threads,
    )

    train(tm, X_train, Y_train, X_val, Y_val, X_test, Y_test, epochs=args.epochs)
