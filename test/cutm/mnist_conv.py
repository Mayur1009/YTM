import numpy as np
from keras.datasets import mnist

from tm_utils import Timer
from cutm import MultiClassTM


def train(tm: MultiClassTM, X_train, Y_train, X_test, Y_test, epochs=1):
    for epoch in range(epochs):
        train_timer = Timer()
        with train_timer:
            tm.fit(X_train, Y_train)

        test_timer = Timer()
        with test_timer:
            test_pred, _ = tm.predict(X_test)
        test_acc = np.mean(Y_test == test_pred)
        print(f"Epoch {epoch + 1}/{epochs}, Test Accuracy: {test_acc * 100:.4f}, Train Time: {train_timer.elapsed():.4f}s, Test Time: {test_timer.elapsed():.4f}s")


if __name__ == "__main__":
    (X_train, Y_train_org), (X_test, Y_test_org) = mnist.load_data()

    X_train = np.where(X_train.reshape((X_train.shape[0], 28 * 28)) > 75, 1, 0)
    X_test = np.where(X_test.reshape((X_test.shape[0], 28 * 28)) > 75, 1, 0)

    Y_train, Y_test = Y_train_org, Y_test_org

    clauses = 500
    T = 1000
    s = 10
    dim = (28, 28, 1)
    patch_dim = (10, 10)
    q = 1
    tm_params = {
        "number_of_clauses": clauses,
        "T": T,
        "s": s,
        "q": q,
        "dim": dim,
        "patch_dim": patch_dim,
        "encode_loc": True,
        "seed": 10,
        "block_size": 16,
    }
    tm = MultiClassTM(**tm_params)

    train(tm, X_train, Y_train, X_test, Y_test, epochs=10)

