import numpy as np
from ytm._o.discrete import MultiClassTM


def generate_NoisyXOR(num_samples: int, noise: float, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 2, size=(num_samples, 2)).astype(np.int8)
    Y = np.logical_xor(X[:, 0], X[:, 1]).astype(np.uint8)

    if noise > 0:
        num_noisy = int(noise * num_samples)
        noisy_indices = rng.choice(num_samples, size=num_noisy, replace=False)
        Y[noisy_indices] = np.logical_not(Y[noisy_indices]).astype(np.uint8)

    return X, Y


def train(tm: MultiClassTM, X_train, Y_train, X_test, Y_test, epochs=1):
    for epoch in range(epochs):
        iota = np.arange(X_train.shape[0])
        np.random.shuffle(iota)
        tm.fit(X_train[iota, ...], Y_train[iota])

        test_pred, _ = tm.predict(X_test)
        train_pred, _ = tm.predict(X_train)

        test_acc = np.mean(Y_test == test_pred)
        train_acc = np.mean(Y_train == train_pred)
        print(f"Epoch {epoch + 1} | Acc> Train: {train_acc * 100:.4f}% Test: {test_acc * 100:.4f}%")


if __name__ == "__main__":
    X_train, Y_train = generate_NoisyXOR(num_samples=500, noise=0.1, seed=10)
    X_test, Y_test = generate_NoisyXOR(num_samples=100, noise=0, seed=11)

    tm = MultiClassTM(
        n_clauses=4,
        T=15,
        s=2,
        dim=(2, 1, 1),
        n_classes=10,
        coalesced=False,
        allow_polarity_change=False,
        n_states=32,
        seed=10,
        device="cpu",
        n_threads=4,
    )

    train(tm, X_train, Y_train, X_test, Y_test, epochs=100)
