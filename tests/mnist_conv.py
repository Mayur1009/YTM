from lzma import LZMAFile
import pickle
import numpy as np
from sklearn.datasets import fetch_openml

from ytm.utils import Timer
from ytm.cpu import MultiClassTM


def train(tm: MultiClassTM, X_train, Y_train, X_test, Y_test, epochs=1):
    encoded_X_train = tm.encode(X_train)
    encoded_X_test = tm.encode(X_test)
    for epoch in range(epochs):
        train_fit_timer = Timer()
        iota = np.arange(encoded_X_train.shape[0])
        np.random.shuffle(iota)
        with train_fit_timer:
            tm.fit(encoded_X_train[iota, ...], Y_train[iota], is_X_encoded=True)

        test_timer = Timer()
        with test_timer:
            test_pred, _ = tm.predict(encoded_X_test, is_X_encoded=True)

        train_timer = Timer()
        with train_timer:
            train_pred, _ = tm.predict(encoded_X_train, is_X_encoded=True)

        test_acc = np.mean(Y_test == test_pred)
        train_acc = np.mean(Y_train == train_pred)
        print(
            f"Epoch {epoch + 1} | Acc> Train: {train_acc * 100:.4f}% Test: {test_acc * 100:.4f}% | Time> Fit: {train_fit_timer.elapsed():.4f}s Infer Train: {train_timer.elapsed():.4f}s Infer Test: {test_timer.elapsed():.4f}s"
        )


if __name__ == "__main__":
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X = np.array(mnist.data).astype(np.uint8)
    Y = np.array(mnist.target).astype(np.uint32)

    X_train, X_test = X[:60000], X[60000:]
    Y_train, Y_test = Y[:60000], Y[60000:]

    X_train = np.where(X_train > 75, 1, 0).astype(np.uint32)
    X_test = np.where(X_test > 75, 1, 0).astype(np.uint32)

    tm = MultiClassTM(
        number_of_clauses_per_class=500,
        T=1000,
        s=10,
        dim=(28, 28, 1),
        n_classes=10,
        patch_dim=(10, 10),
        seed=10,
        num_threads=16,
    )

    train(tm, X_train, Y_train, X_test, Y_test, epochs=10)

    # with LZMAFile("mnist_conv.tm", "wb") as f:
    #     pickle.dump(tm, f)
    #
    # print("Model saved to mnist_conv.tm")
    #
    # # Load the model back
    # with LZMAFile("mnist_conv.tm", "rb") as f:
    #     tm2 = pickle.load(f)
    #
    # print("Model loaded from mnist_conv.tm")
    # train(tm2, X_train, Y_train, X_test, Y_test, epochs=5)
