import numpy as np
import albumentations as A
from datasets import load_dataset
from ytm.discrete.classifier import MultiOutputTM
from ytm.utils import Timer
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

label_names = [
    "Attractive",
    "Heavy_Makeup",
    "High_Cheekbones",
    "Male",
    "Mouth_Slightly_Open",
    "Smiling",
    "Wearing_Lipstick",
]


def preprocessing(images, transforms):
    res = np.stack([transforms(image=np.array(img))["image"] for img in images], axis=0)
    return np.asarray(8 * res.astype(np.float32) / 255.0, dtype=np.int32)


def process_dataset():
    ds = load_dataset("tpremoli/CelebA-attrs")
    Y_train, Y_test = map(
        lambda x: np.asarray((np.stack([x[label] for label in label_names], axis=1) + 1) // 2, dtype=np.int32),
        (ds["train"], ds["test"]),
    )

    train_transforms = A.Compose([A.Resize(64, 64)])
    test_transforms = A.Compose([A.Resize(64, 64)])
    X_train = preprocessing(ds["train"]["image"], train_transforms)
    X_test = preprocessing(ds["test"]["image"], test_transforms)

    # Remove samples with no classes.
    mask = Y_train.any(axis=1)

    return X_train[mask], Y_train[mask], X_test, Y_test


def print_metrics(epoch, train_met: dict, test_met: dict):
    """Prints the training and testing metrics in a formatted table."""
    col_width = 9
    metrics = train_met.keys()
    header = f"| {'Epoch = ' + str(epoch):^{col_width}} |"
    for metric in metrics:
        header += f" {metric:>{col_width}} |"
    print(header)
    separator = "+" + "+".join(["-" * (col_width + 2)] * (len(metrics) + 1)) + "+"
    print(separator)
    for name, data in [("Train", train_met), ("Test", test_met)]:
        row = f"| {name:>{col_width}} |"
        for metric in metrics:
            row += f" {data[metric]:>{col_width}.4f} |"
        print(row)
    print(separator)


def multilabel_metrics(y_true, y_pred, y_prob):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    auc = roc_auc_score(y_true, y_prob, average="weighted")

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auc": auc,
    }


def cs_to_prob(cs, t):
    return (cs + t) / (2 * t)


def train(tm: MultiOutputTM, X_train, Y_train, X_test, Y_test, epochs=1):
    for epoch in range(epochs):
        train_fit_timer = Timer()
        with train_fit_timer:
            tm.fit(X_train, Y_train, clause_drop_p=0.5)

        test_timer = Timer()
        with test_timer:
            test_pred, test_cs = tm.predict(X_test)

        train_timer = Timer()
        with train_timer:
            train_pred, train_cs = tm.predict(X_train)

        train_mets = multilabel_metrics(Y_train, train_pred, cs_to_prob(train_cs, tm.args.T_max))
        test_mets = multilabel_metrics(Y_test, test_pred, cs_to_prob(test_cs, tm.args.T_max))

        print_metrics(epoch + 1, train_mets, test_mets)
        print(
            f"Training time: {train_fit_timer.elapsed:.2f}s, Testing time: {test_timer.elapsed:.2f}s, Train prediction time: {train_timer.elapsed:.2f}s\n"
        )


if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = process_dataset()
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    print(f"X_train min: {X_train.min()}, X_train max: {X_train.max()}")
    print(f"Y_train shape: {Y_train.shape}, Y_test shape: {Y_test.shape}")

    tm = MultiOutputTM(
        n_clauses=25000,
        T=40000,
        s=27,
        q=4,
        dim=X_train.shape[1:],
        n_classes=len(label_names),
        patch_dim=(3, 3),
        feat_mins=X_train.min(),
        feat_maxs=X_train.max(),
        seed=10,
        device="cuda",
    )
    train(tm, X_train, Y_train, X_test, Y_test, epochs=100)
