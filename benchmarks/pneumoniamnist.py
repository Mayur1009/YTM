import argparse
import csv
import lzma
import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
from medmnist import PneumoniaMNIST
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from ytm.discrete.classifier import BinaryTM as DiscreteTM
from ytm.guided.classifier import BinaryTM as GuidedTM
from ytm.utils.timer import Timer

EPOCHS = 100
LVLS = 8

common_args = dict(
    n_clauses=1000,
    T=2000,
    s=2,
    dim=(28, 28, 1),
    patch_dim=(10, 10),
    feat_mins=0,
    feat_maxs=LVLS,
    seed=10,
    device="cuda",
)

discrete_args = dict(**common_args)
guided_args = dict(**common_args, lr=0.5)

SCHEMES = [
    ("discrete", DiscreteTM, discrete_args),
    ("guided", GuidedTM, guided_args),
]

CSV_HEADER = ["epoch", "train_acc", "test_acc", "train_f1", "test_f1", "train_auc", "test_auc", "mean_loss", "fit_time_s", "infer_train_s", "infer_test_s"]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x.squeeze().astype(np.float64)))

BASE_DIR = Path(__file__).parent / "results" / "pneumoniamnist"


def run(name, TM, tm_args, X_train, Y_train, X_test, Y_test, epochs):
    run_path = BASE_DIR / f"{name}_seed{tm_args['seed']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_path.mkdir(parents=True, exist_ok=True)

    with open(run_path / "args.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(tm_args.keys()))
        writer.writeheader()
        writer.writerow(tm_args)

    model = TM(**tm_args)
    fit_timer = Timer()
    infer_train_timer = Timer()
    infer_test_timer = Timer()

    csv_path = run_path / "epochs.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()

        for epoch in range(1, epochs + 1):
            with fit_timer:
                result = model.fit(X_train, Y_train)

            mean_loss = float(np.mean(result)) if result is not None else float("nan")

            with infer_train_timer:
                train_preds, train_cs = model.predict(X_train)

            with infer_test_timer:
                test_preds, test_cs = model.predict(X_test)

            train_probs = _sigmoid(train_cs)
            test_probs = _sigmoid(test_cs)

            train_acc = accuracy_score(Y_train, train_preds)
            test_acc = accuracy_score(Y_test, test_preds)
            train_f1 = f1_score(Y_train, train_preds, average="macro", zero_division=0)
            test_f1 = f1_score(Y_test, test_preds, average="macro", zero_division=0)
            train_auc = roc_auc_score(Y_train, train_probs)
            test_auc = roc_auc_score(Y_test, test_probs)

            writer.writerow(
                {
                    "epoch": epoch,
                    "train_acc": f"{train_acc:.6f}",
                    "test_acc": f"{test_acc:.6f}",
                    "train_f1": f"{train_f1:.6f}",
                    "test_f1": f"{test_f1:.6f}",
                    "train_auc": f"{train_auc:.6f}",
                    "test_auc": f"{test_auc:.6f}",
                    "mean_loss": f"{mean_loss:.6f}",
                    "fit_time_s": f"{fit_timer.elapsed:.3f}",
                    "infer_train_s": f"{infer_train_timer.elapsed:.3f}",
                    "infer_test_s": f"{infer_test_timer.elapsed:.3f}",
                }
            )
            f.flush()

            print(
                f"[{name}] {epoch:3d}/{epochs} | "
                f"train {train_acc * 100:.2f}% test {test_acc * 100:.2f}% | "
                f"f1 {test_f1:.4f} auc {test_auc:.4f} | "
                f"loss {mean_loss:.4f} | fit {fit_timer.elapsed:.1f}s"
            )

    model_path = run_path / "model.tm.lzma"
    with lzma.open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"[{name}] saved {csv_path.name} and {model_path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--scheme", nargs="+", choices=["discrete", "guided"], default=None)
    args = parser.parse_args()

    schemes = [s for s in SCHEMES if args.scheme is None or s[0] in args.scheme]

    print("Loading PneumoniaMNIST...")

    def preprocess(imgs):
        return np.asarray(LVLS * imgs.astype(np.float32) / 255.0, dtype=np.int32)

    ds_train = PneumoniaMNIST(split="train", download=True)
    ds_test = PneumoniaMNIST(split="test", download=True)

    X_train = preprocess(ds_train.imgs)
    Y_train = ds_train.labels.squeeze()
    X_test = preprocess(ds_test.imgs)
    Y_test = ds_test.labels.squeeze()

    print(f"X_train {X_train.shape} range [{X_train.min()}, {X_train.max()}]")

    for name, TM, tm_args in schemes:
        run(name, TM, tm_args, X_train, Y_train, X_test, Y_test, args.epochs)
