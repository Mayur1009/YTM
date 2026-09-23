import argparse

import albumentations as A
import numpy as np
from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from ytm._o.discrete.classifier import MultiOutputTM
from ytm.utils import Timer, print_table

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
    Y_train, Y_test = (
        np.asarray((np.stack([x[label] for label in label_names], axis=1) + 1) // 2, dtype=np.int32) for x in (ds["train"], ds["test"])
    )

    train_transforms = A.Compose([A.Resize(64, 64)])
    test_transforms = A.Compose([A.Resize(64, 64)])
    X_train = preprocessing(ds["train"]["image"], train_transforms)
    X_test = preprocessing(ds["test"]["image"], test_transforms)

    mask = Y_train.any(axis=1)
    return X_train[mask], Y_train[mask], X_test, Y_test


def multilabel_metrics(y_true, y_pred, y_prob):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    auc = roc_auc_score(y_true, y_prob, average="weighted")
    return {"precision": precision, "recall": recall, "f1_score": f1, "auc": auc}


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

        print_table(
            f"Epoch {epoch + 1}/{epochs}",
            {
                "Train": {
                    **{k: f"{v:.4f}" for k, v in train_mets.items()},
                    "Fit Time": f"{train_fit_timer.elapsed:.2f}s",
                    "Infer Time": f"{train_timer.elapsed:.2f}s",
                },
                "Test": {**{k: f"{v:.4f}" for k, v in test_mets.items()}, "Infer Time": f"{test_timer.elapsed:.2f}s"},
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clauses", type=int, default=25000)
    parser.add_argument("--T", type=int, default=40000)
    parser.add_argument("--s", type=float, default=27.0)
    parser.add_argument("--q", type=int, default=4)
    parser.add_argument("--patch", type=int, nargs=2, default=[3, 3], metavar=("H", "W"))
    parser.add_argument("--seed", type=lambda x: None if x == "None" else int(x), default=10)
    parser.add_argument("--coalesced", type=int, choices=[0, 1], default=1)
    parser.add_argument("--n_threads", type=int, default=8)
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    X_train, Y_train, X_test, Y_test = process_dataset()
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    print(f"X_train min: {X_train.min()}, X_train max: {X_train.max()}")
    print(f"Y_train shape: {Y_train.shape}, Y_test shape: {Y_test.shape}")

    tm = MultiOutputTM(
        n_clauses=args.n_clauses,
        T=args.T,
        s=args.s,
        q=args.q,
        dim=X_train.shape[1:],
        n_classes=len(label_names),
        patch_dim=tuple(args.patch),
        feat_mins=X_train.min(),
        feat_maxs=X_train.max(),
        seed=args.seed,
        coalesced=True if args.coalesced == 1 else False,
        device=args.device,
        n_threads=args.n_threads,
    )
    train(tm, X_train, Y_train, X_test, Y_test, epochs=args.epochs)
