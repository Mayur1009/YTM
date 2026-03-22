import numpy as np
from keras.datasets import fashion_mnist

from ytm.utils import Timer, Binarizer
from ytm.tm import MultiClassTM


def train(tm: MultiClassTM, X_train, Y_train, X_test, Y_test, epochs=1):
    encoded_X_train = tm.encode(X_train)
    encoded_X_test = tm.encode(X_test)
    for epoch in range(epochs):
        train_fit_timer = Timer()
        # iota = np.arange(encoded_X_train.shape[0])
        with train_fit_timer:
            tm.fit(encoded_X_train, Y_train, is_X_encoded=True, clause_drop_p=0.5)
            # tm.fit2(X_train, Y_train, clause_drop_p=0.5)


        test_timer = Timer()
        with test_timer:
            test_pred, _ = tm.predict(encoded_X_test, is_X_encoded=True)
            # test_pred, _ = tm.predict2(X_test)


        train_timer = Timer()
        with train_timer:
            train_pred, _ = tm.predict(encoded_X_train, is_X_encoded=True)
            # train_pred, _ = tm.predict2(X_train)

        test_acc = np.mean(Y_test == test_pred)
        train_acc = np.mean(Y_train == train_pred)
        print(
            f"Epoch {epoch + 1} | Acc> Train: {train_acc * 100:.4f}% Test: {test_acc * 100:.4f}% | Time> Fit: {train_fit_timer.elapsed:.4f}s Infer Train: {train_timer.elapsed:.4f}s Infer Test: {test_timer.elapsed:.4f}s"
        )


if __name__ == "__main__":
    (X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()
    X_train = np.copy(X_train)
    X_test = np.copy(X_test)

    ch = 8

    b = Binarizer(ch)
    b.fit(X_train)
    X_train = b.transform(X_train).reshape((X_train.shape[0], -1)).astype(np.int8)
    X_test = b.transform(X_test).reshape((X_test.shape[0], -1)).astype(np.int8)


    tm = MultiClassTM(
        n_clauses=6000,
        T=10000,
        s=10,
        dim=(28, 28, 8),
        n_classes=10,
        patch_dim=(3, 3),
        seed=10,
        device="cuda",
        n_threads=8,
    )
    train(tm, X_train, Y_train, X_test, Y_test, epochs=10)

