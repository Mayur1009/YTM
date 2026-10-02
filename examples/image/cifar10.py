"""Train discrete or guided TM on CIFAR-10 dataset.

Images are RGB(or gray when --gray).
Binarization per channel:
    adaptive     - adaptive Gaussian thresholding (window 11, C 2), 1 bit per channel (default).
    thermometer  - pixels 0-255 -> levels 0-L (--levels, default 16), thermometer encoded by the TM.

Usage:
    python cifar10.py {discrete,guided} [options]

Examples:
    python cifar10.py discrete --device cuda:0                 # non-coalesced, 6000 clauses per class
    python cifar10.py discrete --binarization thermometer --device cuda:0
    python cifar10.py discrete --coalesced 1 --n_clauses 20000 --device cuda:0
    python cifar10.py guided --lr 0.5 --device cuda:0 --epochs 20

Options:
    - python cifar10.py <discrete/guided> --help shows all options.
    - Common:    --epochs --clause_drop_p --gray --binarization adaptive|thermometer --levels --n_clauses --s --coalesced 0|1 --patch H,W|none --seed --device cpu:N|cuda:N
    - discrete:  --T
    - guided:    --lr --lambda
"""

import argparse

import numpy as np
from datasets import load_dataset
from scipy.ndimage import gaussian_filter

from ytm.discrete import MultiClassTM as DiscreteTM
from ytm.guided import MultiClassTM as GuidedTM
from ytm.utils import Timer, print_table


def adaptive_threshold(X):
    # Same as cv2.adaptiveThreshold(ADAPTIVE_THRESH_GAUSSIAN_C, blockSize=11, C=2) per channel:
    # pixel is 1 if above the Gaussian-weighted mean of its 11x11 neighbourhood minus 2.
    X = X.astype(np.float32)
    local_mean = gaussian_filter(X, sigma=(0, 2.0, 2.0, 0), truncate=2.5, mode="nearest")
    return (X > local_mean - 2).astype(np.uint8)


def thermometer(X, levels: int):
    # Pixels 0-255 -> levels 0-levels, thermometer encoded by the TM.
    return (levels * X.astype(np.float32) / 255.0).astype(np.uint8)


def to_gray(X):
    # ITU-R BT.601 luma, (N, H, W, 3) -> (N, H, W, 1).
    return np.rint(X @ np.array([0.299, 0.587, 0.114], dtype=np.float32))[..., None].astype(np.uint8)


def load_cifar10(binarization: str, levels: int, gray: bool):
    ds = load_dataset("uoft-cs/cifar10")
    encode = adaptive_threshold if binarization == "adaptive" else lambda X: thermometer(X, levels)
    X_train, X_test = (np.array(ds[split]["img"]) for split in ("train", "test"))
    if gray:
        X_train, X_test = to_gray(X_train), to_gray(X_test)
    X_train, X_test = encode(X_train), encode(X_test)
    Y_train, Y_test = (np.array(ds[split]["label"], dtype=np.uint8) for split in ("train", "test"))
    return X_train, Y_train, X_test, Y_test


def train_model(tm, xtrain, ytrain, xtest, ytest, epochs: int, clause_drop_p: float, model_name: str):
    for epoch in range(epochs):
        with (fit_timer := Timer()):
            loss = tm.fit(xtrain, ytrain, clause_drop_p=clause_drop_p)

        with (test_timer := Timer()):
            test_pred, _ = tm.predict(xtest)

        with (train_timer := Timer()):
            train_pred, _ = tm.predict(xtrain)

        train_log = {
            "Acc": f"{(train_pred == ytrain).mean() * 100:.4f}%",
            "Eval Time": f"{train_timer.elapsed:.2f}s",
            "Fit Time": f"{fit_timer.elapsed:.2f}s",
        }
        test_log = {"Acc": f"{(test_pred == ytest).mean() * 100:.4f}%", "Eval Time": f"{test_timer.elapsed:.2f}s"}

        if loss is not None:  # Only loss-guided TM returns loss
            train_log["Loss"] = f"{loss:.4f}"

        print_table(
            f"{model_name} epoch {epoch + 1}/{epochs}",
            {
                "Train": train_log,
                "Test": test_log,
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TM CIFAR-10 dataset")
    # Subcommand to select discrete or guided
    models = parser.add_subparsers(dest="model", required=True)

    # Common args
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--epochs", type=int, default=10)
    common.add_argument("--binarization", choices=["adaptive", "thermometer"], default="adaptive")
    common.add_argument("--levels", type=int, default=16, help="thermometer levels")
    common.add_argument("--gray", action="store_true", help="use grayscale images instead of RGB")
    common.add_argument("--n_clauses", type=int, default=6000)
    common.add_argument("--s", type=float, default=10.0)
    common.add_argument("--coalesced", type=lambda v: bool(int(v)), default=False, metavar="0|1")
    common.add_argument(
        "--patch",
        dest="patch_dim",
        type=lambda v: None if v.lower() == "none" else tuple(map(int, v.split(","))),
        default="8,8",
        help="H,W or none",
    )
    common.add_argument("--clause_drop_p", type=float, default=0.5)
    common.add_argument("--seed", type=int, default=10)
    common.add_argument("--device", type=str, default="cpu:1", help="cpu:N_THREADS or cuda:GPU_ID")

    # Discrete args
    discrete = models.add_parser("discrete", parents=[common])
    discrete.add_argument("--T", type=float, default=48000)

    # Guided args
    guided = models.add_parser("guided", parents=[common])
    guided.add_argument("--lr", type=float, default=1.0)
    guided.add_argument("--lambda", dest="lambda_", type=float, default=1.0)

    # Parse args
    args = parser.parse_args()

    # Args to dict
    params = vars(args)
    model_name, epochs, clause_drop_p = params.pop("model"), params.pop("epochs"), params.pop("clause_drop_p")
    binarization, levels, gray = params.pop("binarization"), params.pop("levels"), params.pop("gray")

    # Model Initialization
    TM = DiscreteTM if model_name == "discrete" else GuidedTM
    tm = TM(
        **params,
        dim=(32, 32, 1 if gray else 3),
        n_classes=10,
        feat_maxs=1 if binarization == "adaptive" else levels,
    )

    # Training
    train_model(tm, *load_cifar10(binarization, levels, gray), epochs=epochs, clause_drop_p=clause_drop_p, model_name=model_name)
