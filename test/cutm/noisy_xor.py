import numpy as np
from cutm import MultiClassTM


def generate_NoisyXOR(num_samples: int, noise: float, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 2, size=(num_samples, 2)).astype(np.uint32)
    Y = np.logical_xor(X[:, 0], X[:, 1]).astype(np.uint32)

    if noise > 0:
        num_noisy = int(noise * num_samples)
        noisy_indices = rng.choice(num_samples, size=num_noisy, replace=False)
        Y[noisy_indices] = np.logical_not(Y[noisy_indices]).astype(np.uint32)

    return X, Y


def train(tm: MultiClassTM, X_train, Y_train, X_test, Y_test, epochs=1):
    encoded_X_train = tm.encode(X_train)
    encoded_X_test = tm.encode(X_test)
    for epoch in range(epochs):
        iota = np.arange(encoded_X_train.shape[0])
        np.random.shuffle(iota)
        tm.fit(encoded_X_train[iota, ...], Y_train[iota], is_X_encoded=True)

        test_pred, _ = tm.predict(encoded_X_test, is_X_encoded=True)
        train_pred, _ = tm.predict(encoded_X_train, is_X_encoded=True)

        test_acc = np.mean(Y_test == test_pred)
        train_acc = np.mean(Y_train == train_pred)
        print(f"Epoch {epoch + 1} | Acc> Train: {train_acc * 100:.4f}% Test: {test_acc * 100:.4f}%")


if __name__ == "__main__":
    X_train, Y_train = generate_NoisyXOR(num_samples=500, noise=0.1, seed=10)
    X_test, Y_test = generate_NoisyXOR(num_samples=200, noise=0, seed=11)

    tm = MultiClassTM(
        number_of_clauses_per_class=4,
        T=2,
        s=2,
        dim=(2, 1, 1),
        n_classes=2,
        weighted=False,
        coalesced=False,
        number_of_ta_states=16,
        block_size=2
    )

    train(tm, X_train, Y_train, X_test, Y_test, epochs=100)

    weights = tm.get_weights()
    print("Weights:", weights)

    clauses = tm.get_literals()
    print("Clauses:", clauses)

    states = tm.get_ta_state()
    print("TA States:", states)
