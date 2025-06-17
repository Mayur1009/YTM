import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from pytm.tm import MultiClassTsetlinMachine
from mltm.utils import Timer
from sklearn.metrics import confusion_matrix

def make_digits_imbalanced(seed=10):
    X, y =  load_digits(n_class=2, return_X_y=True)

    X_train_orig, X_test_orig, y_train_orig, y_test_orig = map(np.array, train_test_split(X, y, test_size=0.2, random_state=seed))

    # Create an imbalanced dataset by removing some samples from the minority class
    class_0_indices = np.where(y_train_orig == 0)[0]
    class_1_indices = np.where(y_train_orig == 1)[0]

    rng = np.random.default_rng(seed)

    selected_class_0_indices = rng.choice(class_0_indices, size=int(len(class_0_indices) * 0.1), replace=False)

    X_train = X_train_orig[np.concatenate([selected_class_0_indices, class_1_indices])]
    y_train = y_train_orig[np.concatenate([selected_class_0_indices, class_1_indices])]

    return X_train, y_train, X_test_orig, y_test_orig

if __name__ == "__main__":

    X_train, y_train, X_test, y_test = make_digits_imbalanced()

    print(f'{X_train.shape=}')
    print(f'{y_train.shape=}')
    print(f'{X_test.shape=}')
    print(f'{y_test.shape=}')

    print(f" Number of samples per class: {np.bincount(y_train)}")

    params = {
        "number_of_clauses": 16,
        "T": 100,
        "s": 3,
        "append_negated": True,
    }

    tm = MultiClassTsetlinMachine(**params)

    epochs = 10
    for epoch in range(epochs):
        train_timer = Timer()
        with train_timer:
            tm.fit(X_train, y_train, epochs=1, incremental=True)


        pred = tm.predict(X_test)
        acc = np.mean(pred == y_test)

        cm = confusion_matrix(y_test, pred)

        print(f"Epoch {epoch+ 1}, Acc: {acc}, train time: {train_timer.elapsed():.4f}s")
        print(f"Confusion Matrix:\n{cm}")

