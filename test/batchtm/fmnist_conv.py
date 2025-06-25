import numpy as np
from keras.datasets import fashion_mnist
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
                end = i + batch_size
                if end > len(X_train):
                    end = len(X_train)
                tm.fit(X_train[i:end], Y_train[i:end])

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
    (X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()
    X_train = np.copy(X_train)
    X_test = np.copy(X_test)

    ch = 8

    out = np.zeros((*X_train.shape, ch))
    for j in range(ch):
        t1 = (j + 1) * 255 / (ch + 1)
        out[:, :, :, j] = (X_train >= t1) & 1
    X_train = np.array(out)
    X_train = X_train.reshape((X_train.shape[0], -1)).astype(np.uint32)

    out = np.zeros((*X_test.shape, ch))
    for j in range(ch):
        t1 = (j + 1) * 255 / (ch + 1)
        out[:, :, :, j] = (X_test >= t1) & 1
    X_test = np.array(out)
    X_test = X_test.reshape((X_test.shape[0], -1)).astype(np.uint32)

    tm = MultiClassTM(
        number_of_clauses=40000,
        T=15000,
        s=10,
        dim=(28, 28, 8),
        n_classes=10,
        patch_dim=(3, 3),
        seed=10,
        block_size=128,
    )
    train(tm, X_train, Y_train, X_test, Y_test, epochs=30)
