import numpy as np
from datasets import load_dataset

from ytm.discrete.classifier import MultiClassTM
from ytm.utils import Timer


def train(tm: MultiClassTM, X_train, Y_train, X_test, Y_test, epochs=1):
    for epoch in range(epochs):
        train_fit_timer = Timer()
        with train_fit_timer:
            tm.fit(X_train, Y_train)

        test_timer = Timer()
        with test_timer:
            test_pred, _ = tm.predict(X_test)

        train_timer = Timer()
        with train_timer:
            train_pred, _ = tm.predict(X_train)

        test_acc = np.mean(Y_test == test_pred)
        train_acc = np.mean(Y_train == train_pred)
        print(
            f"Epoch {epoch + 1} | Acc> Train: {train_acc * 100:.4f}% Test: {test_acc * 100:.4f}% | Time> Fit: {train_fit_timer.elapsed:.4f}s Infer Train: {train_timer.elapsed:.4f}s Infer Test: {test_timer.elapsed:.4f}s"
        )


if __name__ == "__main__":
    ds = load_dataset("ylecun/mnist")
    Y_train, Y_test = map(lambda x: np.array(x).astype(np.uint8), (ds["train"]["label"], ds["test"]["label"]))
    X_train, X_test = map(
        lambda x: np.where(np.array(x) > 75, 1, 0).astype(np.uint8),
        (ds["train"]["image"], ds["test"]["image"]),
    )

    tm = MultiClassTM(
        n_clauses=500,
        T=1000,
        s=10,
        dim=(28, 28, 1),
        n_classes=10,
        patch_dim=(10, 10),
        feat_mins=X_train.min(),  # Since all the features have a min of 0.
        feat_maxs=X_train.max(),  # Since all the features have a max of 1.
        seed=10,
        device="cpu",
        n_threads=8,
    )

    train(tm, X_train, Y_train, X_test, Y_test, epochs=10)
