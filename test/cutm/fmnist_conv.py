import numpy as np
from keras.datasets import fashion_mnist

from tm_utils import Timer
from tm_utils.binarizer import ThermometerBinarizer
from cutm import MultiClassTM


def train(tm: MultiClassTM, X_train, Y_train, X_test, Y_test, epochs=1):
    encoded_X_train = tm.encode(X_train)
    encoded_X_test = tm.encode(X_test)
    for epoch in range(epochs):
        train_fit_timer = Timer()
        iota = np.arange(encoded_X_train.shape[0])
        np.random.shuffle(iota)
        with train_fit_timer:
            # tm.fit(X_train[iota], Y_train[iota])
            tm.fit(encoded_X_train[iota, ...], Y_train[iota], is_X_encoded=True)

        test_timer = Timer()
        with test_timer:
            # test_pred, _ = tm.predict(X_test)
            test_pred, _ = tm.predict(encoded_X_test, is_X_encoded=True, block_size=256)

        train_timer = Timer()
        with train_timer:
            # train_pred, _ = tm.predict(X_train)
            train_pred, _ = tm.predict(encoded_X_train, is_X_encoded=True, block_size=256)

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

    ther_bin = ThermometerBinarizer(ch=ch)
    X_train = ther_bin.binarize_gray(X_train).reshape((X_train.shape[0], -1)).astype(np.uint32)
    X_test = ther_bin.binarize_gray(X_test).reshape((X_test.shape[0], -1)).astype(np.uint32)

    tm = MultiClassTM(
        number_of_clauses=10000,
        T=15000,
        s=10,
        dim=(28, 28, 8),
        n_classes=10,
        patch_dim=(3, 3),
        seed=10,
        block_size=64,
    )
    train(tm, X_train, Y_train, X_test, Y_test, epochs=1)

