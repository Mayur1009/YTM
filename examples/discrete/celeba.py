import numpy as np
import albumentations as A
from datasets import load_dataset
from ytm.discrete.classifier import MultiOutputTM
from ytm.utils import Timer

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
        (ds["train"], ds["test"])
    )

    train_transforms = A.Compose([
        A.Resize(128, 128),
        A.HorizontalFlip(p=0.5),
    ])
    test_transforms = A.Compose([
        A.Resize(128, 128),
    ])
    X_train = preprocessing(ds["train"]["image"], train_transforms)
    X_test = preprocessing(ds["test"]["image"], test_transforms)

    # Remove samples with no classes.
    mask = Y_train.any(axis=1)

    return X_train[mask], Y_train[mask], X_test, Y_test

def train(tm: MultiOutputTM, X_train, Y_train, X_test, Y_test, epochs=1):
    for epoch in range(epochs):
        train_fit_timer = Timer()
        with train_fit_timer:
            tm.fit(X_train, Y_train, clause_drop_p=0.5)

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

if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = process_dataset()
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    print(f"X_train min: {X_train.min()}, X_train max: {X_train.max()}")
    print(f"Y_train shape: {Y_train.shape}, Y_test shape: {Y_test.shape}")

    tm = MultiOutputTM(
        n_clauses=28000,
        T=40000,
        s=25,
        dim=(128, 128, 3),
        n_classes=len(label_names),
        patch_dim=(3, 3),
        feat_mins=X_train.min(),
        feat_maxs=X_train.max(),
        seed=10,
        device="cuda",
    )
    train(tm, X_train, Y_train, X_test, Y_test, epochs=10)
