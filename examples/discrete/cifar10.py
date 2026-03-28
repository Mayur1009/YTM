import numpy as np
from keras.datasets import cifar10

from ytm.discrete.multiclass import MultiClassTM
from ytm.utils import Timer

if __name__ == "__main__":
    (X_train_org, Y_train), (X_test_org, Y_test) = cifar10.load_data()

    # Discretize the images into 8 bins.
    X_train = np.asarray(8 * X_train_org.astype(np.float32) / 255.0, dtype=np.int32)
    X_test = np.asarray(8 * X_test_org.astype(np.float32) / 255.0, dtype=np.int32)
    Y_train = Y_train.reshape(Y_train.shape[0]).astype(np.int8)
    Y_test = Y_test.reshape(Y_test.shape[0]).astype(np.int8)

    tm = MultiClassTM(
        n_clauses=1000,
        T=5000,
        s=10.0,
        dim=(32, 32, 3),  # Images are 32x32x3
        n_classes=10,
        patch_dim=(5, 5),
        coalesced=False,
        feat_mins=int(X_train.min()),
        feat_maxs=int(X_train.max()),
        n_threads=8,
        device="cuda",
        seed=42,
    )

    for epoch in range(10):
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
