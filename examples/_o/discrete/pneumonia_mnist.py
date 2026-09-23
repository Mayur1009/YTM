import argparse

import numpy as np
from medmnist import PneumoniaMNIST
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

from ytm._o.discrete.classifier import BinaryTM
from ytm.utils import Timer, print_table


def load_data(n_levels=8):
    train = PneumoniaMNIST(split="train", download=True)
    val = PneumoniaMNIST(split="val", download=True)
    test = PneumoniaMNIST(split="test", download=True)

    def preprocess(imgs):
        return np.asarray(n_levels * imgs.astype(np.float32) / 255.0, dtype=np.int32)

    return (
        (preprocess(train.imgs), train.labels.squeeze()),
        (preprocess(val.imgs), val.labels.squeeze()),
        (preprocess(test.imgs), test.labels.squeeze()),
    )


def cs_to_prob(cs, t):
    return (np.clip(cs.squeeze(), -t, t) + t) / (2 * t)


def evaluate(tm, X, Y):
    pred, cs = tm.predict(X)
    prob = cs_to_prob(cs, tm.args.T_max)
    acc = accuracy_score(Y, pred)
    auc = roc_auc_score(Y, prob)
    return acc, auc, pred


def train(tm: BinaryTM, X_train, Y_train, X_val, Y_val, X_test, Y_test, epochs=1):
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
    parser.add_argument("--T", type=int, default=2000)
    parser.add_argument("--s", type=float, default=2.0)
    parser.add_argument("--patch", type=int, nargs=2, default=[10, 10], metavar=("H", "W"))
    parser.add_argument("--n_levels", type=int, default=8)
    parser.add_argument("--seed", type=lambda x: None if x == "None" else int(x), default=10)
    parser.add_argument("--coalesced", type=int, choices=[0, 1], default=1)
    parser.add_argument("--n_threads", type=int, default=8)
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = load_data(args.n_levels)
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    tm = BinaryTM(
        n_clauses=args.n_clauses,
        T=args.T,
        s=args.s,
        dim=(28, 28, 1),
        patch_dim=tuple(args.patch),
        feat_mins=X_train.min(),
        feat_maxs=X_train.max(),
        seed=args.seed,
        coalesced=True if args.coalesced == 1 else False,
        device=args.device,
        n_threads=args.n_threads,
    )

    train(tm, X_train, Y_train, X_val, Y_val, X_test, Y_test, epochs=args.epochs)
