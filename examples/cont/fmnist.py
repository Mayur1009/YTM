import numpy as np
from keras.datasets import fashion_mnist

from ytm.utils import Timer, Binarizer
from ytm.continuous.multiclass import MultiClassTM


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
    (X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()

    # Discretize the pixel values in range [0, 255] to [0, 8] (i.e., 8 bins).
    X_train = np.asarray(8 * X_train.astype(np.float32) / 255.0, dtype=np.int32)
    X_test = np.asarray(8 * X_test.astype(np.float32) / 255.0, dtype=np.int32)

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

