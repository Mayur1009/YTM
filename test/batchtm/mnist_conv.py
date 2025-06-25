import numpy as np
from keras.datasets import mnist
from tqdm import tqdm

from tm_utils import Timer
from batchtm import MultiClassTM

np.random.seed(42)

def train(tm: MultiClassTM, X_train, Y_train, X_test, Y_test, epochs=1, batch_size=256):
    for epoch in range(epochs):
        train_fit_timer = Timer()
        iota = np.arange(len(X_train))
        np.random.shuffle(iota)
        X_train = X_train[iota]
        Y_train = Y_train[iota]
        with train_fit_timer:
            for i in tqdm(range(0, len(X_train), batch_size), leave=False, dynamic_ncols=True):
                # label_cnts = np.bincount(Y_train[i:i + batch_size], minlength=tm.number_of_outputs)
                # print(f'{label_cnts=}')
                end = i + batch_size
                if end > len(X_train):
                    end = len(X_train)
                tm.fit(X_train[i:end], Y_train[i:end])
                # tm.fit(X_train, Y_train)

        test_timer = Timer()
        with test_timer:
            test_pred, _ = tm.predict(X_test)

        train_timer = Timer()
        with train_timer:
            train_pred, _ = tm.predict(X_train)

        test_acc = np.mean(Y_test == test_pred)
        train_acc = np.mean(Y_train == train_pred)
        print(
            f"Epoch {epoch + 1} | Acc> Train: {train_acc * 100:.4f}% Test: {test_acc * 100:.4f}% | Time> Fit: {train_fit_timer.elapsed():.4f}s Infer Train: {train_timer.elapsed():.4f}s Infer Test: {test_timer.elapsed():.4f}s"
        )


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
        "n_classes": 10,
        "q": q,
        "dim": dim,
        "patch_dim": patch_dim,
        "encode_loc": True,
        "seed": 10,
        "block_size": 16,
    }
    tm = MultiClassTM(**tm_params)

    train(tm, X_train, Y_train, X_test, Y_test, epochs=10)
