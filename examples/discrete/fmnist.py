import numpy as np
from datasets import load_dataset
from ytm.discrete.classifier import MultiClassTM
from ytm.utils import Timer


def train(tm: MultiClassTM, X_train, Y_train, X_test, Y_test, epochs=1):
    for epoch in range(epochs):
        train_fit_timer = Timer()
        with train_fit_timer:
            tm.fit(X_train, Y_train, clause_drop_p=0.5)

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
    ds = load_dataset("zalando-datasets/fashion_mnist")
    Y_train, Y_test = map(np.array, (ds["train"]["label"], ds["test"]["label"]))
    X_train, X_test = map(
        lambda x: np.asarray(8 * np.array(x).astype(np.float32) / 255.0, dtype=np.int32),
        (ds["train"]["image"], ds["test"]["image"]),
    )
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    print(f"X_train min: {X_train.min()}, X_train max: {X_train.max()}")

    tm = MultiClassTM(
        n_clauses=6000,
        T=10000,
        s=10,
        dim=(28, 28, 1),
        n_classes=10,
        patch_dim=(3, 3),
        feat_mins=X_train.min(),
        feat_maxs=X_train.max(),
        seed=10,
        device="cuda",
        n_threads=8,
    )
    train(tm, X_train, Y_train, X_test, Y_test, epochs=10)
