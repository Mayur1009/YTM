from lzma import LZMAFile
import pickle
import numpy as np
from keras.datasets import mnist

from tm_utils import Timer
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
            tm.fit(encoded_X_train[iota, ...], Y_train[iota], is_X_encoded=True, block_size=16)

        test_timer = Timer()
        with test_timer:
            # test_pred, _ = tm.predict(X_test)
            test_pred, _ = tm.predict(encoded_X_test, is_X_encoded=True, block_size=512)

        train_timer = Timer()
        with train_timer:
            # train_pred, _ = tm.predict(X_train)
            train_pred, _ = tm.predict(encoded_X_train, is_X_encoded=True, block_size=512)

        test_acc = np.mean(Y_test == test_pred)
        train_acc = np.mean(Y_train == train_pred)
        print(
            f"Epoch {epoch + 1} | Acc> Train: {train_acc * 100:.4f}% Test: {test_acc * 100:.4f}% | Time> Fit: {train_fit_timer.elapsed():.4f}s Infer Train: {train_timer.elapsed():.4f}s Infer Test: {test_timer.elapsed():.4f}s"
        )


if __name__ == "__main__":
    (X_train, Y_train_org), (X_test, Y_test_org) = mnist.load_data()

    X_train = np.where(X_train.reshape((X_train.shape[0], 28 * 28)) > 75, 1, 0)
    X_test = np.where(X_test.reshape((X_test.shape[0], 28 * 28)) > 75, 1, 0)
    X_train = np.asarray(X_train, dtype=np.uint32)
    X_test = np.asarray(X_test, dtype=np.uint32)

    Y_train, Y_test = Y_train_org, Y_test_org

    tm = MultiClassTM(
        number_of_clauses=500,
        T=1000,
        s=10,
        dim=(28, 28, 1),
        n_classes=10,
        patch_dim=(10, 10),
        encode_loc=True,
        seed=10,
        block_size=16,
    )

    train(tm, X_train, Y_train, X_test, Y_test, epochs=10)

    # with LZMAFile("mnist_conv.tm", "wb") as f:
    #     pickle.dump(tm, f)
    #
    # print("Model saved to mnist_conv.tm")
    #
    # with LZMAFile("mnist_conv.tm", "rb") as f:
    #     tm2 = pickle.load(f)
    #
    # print("Model loaded from mnist_conv.tm")
    # train(tm2, X_train, Y_train, X_test, Y_test, epochs=1)


