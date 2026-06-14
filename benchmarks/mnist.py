import csv
import lzma
import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
from datasets import load_dataset

from ytm.discrete.classifier import MultiClassTM as DiscreteTM
from ytm.guided.classifier import MultiClassTM as GuidedTM
from ytm.utils.timer import Timer

EPOCHS = 100

common_args = dict(
    n_clauses=500,
    T=1000,
    s=10,
    dim=(28, 28, 1),
    n_classes=10,
    patch_dim=(10, 10),
    feat_mins=0,
    feat_maxs=1,
    seed=10,
    device="cuda",
)

discrete_args = dict(**common_args)
guided_args = dict(**common_args, lr=1.0)

SCHEMES = [
    ("discrete", DiscreteTM, discrete_args),
    ("guided", GuidedTM, guided_args),
]

CSV_HEADER = ["epoch", "train_acc", "test_acc", "mean_loss", "fit_time_s", "infer_train_s", "infer_test_s"]

BASE_DIR = Path(__file__).parent / "results" / "mnist"


def run(name, TM, tm_args, X_train, Y_train, X_test, Y_test):
    run_path = BASE_DIR / f"{name}_seed{tm_args['seed']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_path.mkdir(parents=True)

    with open(run_path / "args.csv") as f:
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

        for epoch in range(1, EPOCHS + 1):
            with fit_timer:
                result = model.fit(X_train, Y_train)

            mean_loss = float(np.mean(result)) if result is not None else float("nan")

            with infer_train_timer:
                train_preds, _ = model.predict(X_train)

            with infer_test_timer:
                test_preds, _ = model.predict(X_test)

            train_acc = float(np.mean(Y_train == train_preds))
            test_acc = float(np.mean(Y_test == test_preds))

            writer.writerow(
                {
                    "epoch": epoch,
                    "train_acc": f"{train_acc:.6f}",
                    "test_acc": f"{test_acc:.6f}",
                    "mean_loss": f"{mean_loss:.6f}",
                    "fit_time_s": f"{fit_timer.elapsed:.3f}",
                    "infer_train_s": f"{infer_train_timer.elapsed:.3f}",
                    "infer_test_s": f"{infer_test_timer.elapsed:.3f}",
                }
            )
            f.flush()

            print(
                f"[{name}] {epoch:3d}/{EPOCHS} | "
                f"train {train_acc * 100:.2f}% test {test_acc * 100:.2f}% | "
                f"loss {mean_loss:.4f} | fit {fit_timer.elapsed:.1f}s"
            )

    model_path = run_path / "model.tm.lzma"
    with lzma.open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"[{name}] saved {csv_path.name} and {model_path.name}")


if __name__ == "__main__":
    print("Loading MNIST...")

    ds = load_dataset("ylecun/mnist")
    Y_train = np.array(ds["train"]["label"], dtype=np.uint8)
    Y_test = np.array(ds["test"]["label"], dtype=np.uint8)

    X_train = np.asarray(np.array(ds["train"]["image"]) > 75, dtype=np.uint8)
    X_test = np.asarray(np.array(ds["test"]["image"]) > 75, dtype=np.uint8)

    print(f"X_train {X_train.shape} range [{X_train.min()}, {X_train.max()}]")

    for name, TM, tm_args in SCHEMES:
        run(name, TM, tm_args, X_train, Y_train, X_test, Y_test)
