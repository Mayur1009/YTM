import numpy as np
from keras.datasets import mnist

from ytm.discrete.multiclass import MultiClassTM
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
    (X_train, Y_train_org), (X_test, Y_test_org) = mnist.load_data()

    # Convert pixel values to binary (0 or 1) based on a threshold of 75
    X_train = np.where(X_train.reshape((X_train.shape[0], 28 * 28)) > 75, 1, 0)
    X_test = np.where(X_test.reshape((X_test.shape[0], 28 * 28)) > 75, 1, 0)
    X_train = np.asarray(X_train, dtype=np.int8)
    X_test = np.asarray(X_test, dtype=np.int8)

    Y_train, Y_test = Y_train_org, Y_test_org
    tm = MultiClassTM(
        n_clauses=500,
        T=1000,
        s=10,
        dim=(28, 28, 1),
        n_classes=10,
        patch_dim=(10, 10),
        stride=(1, 1),
        feat_mins=X_train.min(), # Since all the features have a min of 0.
        feat_maxs=X_train.max(), # Since all the features have a max of 1.
        seed=10,
        device="cpu",
        n_threads=8,
    )

    train(tm, X_train, Y_train, X_test, Y_test, epochs=10)
