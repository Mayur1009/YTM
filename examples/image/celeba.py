"""Train discrete or guided TM on CelebA dataset (40 binary face attributes, multilabel).

Images resized to 64x64 RGB (or gray with --gray), pixels encoded as levels 0-L (--levels, default 8, thermometer), random horizontal flip during training.

Usage:
    python celeba.py {discrete,guided} [options]

Examples:
    python celeba.py discrete --device cuda:0                  # 3x3 conv, 80000 clauses
    python celeba.py guided --act_loss bce --device cuda:0
    python celeba.py guided --n_clauses 20000 --lambda 10 --device cuda:0 --epochs 20

Options:
    - python celeba.py <discrete/guided> --help shows all options.
    - Common:    --epochs --clause_drop_p --gray --levels --n_clauses --s --patch H,W|none --seed --device cpu:N|cuda:N
    - discrete:  --T --q
    - guided:    --lr --lambda --act_loss asl|bce
"""

import argparse
from io import BytesIO

import numpy as np
import PIL.Image
from datasets import Image, load_dataset
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from ytm.discrete import MultiOutputTM as DiscreteTM
from ytm.guided import MultiOutputTM as GuidedTM
from ytm.guided.backends.act_loss import ASL, SigmoidBCE
from ytm.utils import Timer, print_table

ACT_LOSSES = {
    "asl": lambda: ASL(gamma_pos=2.0, gamma_neg=4.0, clip=0.01),
    "bce": lambda: SigmoidBCE(),
}


def to_gray(X):
    # ITU-R BT.601 luma, (N, H, W, 3) -> (N, H, W, 1).
    return np.rint(X @ np.array([0.299, 0.587, 0.114], dtype=np.float32))[..., None].astype(np.uint8)


def load_split(split, levels: int, gray: bool):
    # Raw JPEG bytes, so the decoder can downscale while decoding (draft).
    split = split.select_columns("image").cast_column("image", Image(decode=False))
    X = np.empty((len(split), 64, 64, 3), dtype=np.uint8)
    for i, row in enumerate(split):
        img = PIL.Image.open(BytesIO(row["image"]["bytes"]))
        img.draft("RGB", (64, 64))
        X[i] = np.asarray(img.convert("RGB").resize((64, 64)))
    if gray:
        X = to_gray(X)
    # Pixels 0-255 -> levels 0-levels, thermometer encoded by the TM.
    return (levels * X.astype(np.float32) / 255.0).astype(np.uint8)


def load_celeba(levels: int, gray: bool):
    ds = load_dataset("tpremoli/CelebA-attrs")
    attrs = [name for name in ds["train"].column_names if name not in ("image", "prompt_string")]
    # Attributes are -1/1 -> 0/1.
    Y_train, Y_test = ((np.stack([ds[split][a] for a in attrs], axis=1) + 1) // 2 for split in ("train", "test"))
    return (
        load_split(ds["train"], levels, gray),
        Y_train.astype(np.uint8),
        load_split(ds["test"], levels, gray),
        Y_test.astype(np.uint8),
    )


def multilabel_metrics(y, pred, scores):
    # Scores rank the samples per label, so raw votes (discrete) and probabilities (guided) both work for the AUCs.
    return {
        "Macro F1": f"{f1_score(y, pred, average='macro', zero_division=0) * 100:.2f}%",
        "Weighted F1": f"{f1_score(y, pred, average='weighted', zero_division=0) * 100:.2f}%",
        "ROC AUC": f"{roc_auc_score(y, scores, average='macro') * 100:.2f}%",
        "PR AUC": f"{average_precision_score(y, scores, average='macro') * 100:.2f}%",
    }


def train_model(tm, xtrain, ytrain, xtest, ytest, epochs: int, clause_drop_p: float, seed: int, model_name: str):
    rng = np.random.default_rng(seed)
    for epoch in range(epochs):
        # Random horizontal flip of half the training images.
        flip = rng.random(len(xtrain)) < 0.5
        xtrain_epoch = xtrain.copy()
        xtrain_epoch[flip] = xtrain_epoch[flip, :, ::-1]

        with (fit_timer := Timer()):
            loss = tm.fit(xtrain_epoch, ytrain, clause_drop_p=clause_drop_p)

        with (test_timer := Timer()):
            test_pred, test_scores = tm.predict(xtest)

        with (train_timer := Timer()):
            train_pred, train_scores = tm.predict(xtrain)

        train_log = {
            **multilabel_metrics(ytrain, train_pred, tm.to_prob(train_scores)),
            "Eval Time": f"{train_timer.elapsed:.2f}s",
            "Fit Time": f"{fit_timer.elapsed:.2f}s",
        }
        test_log = {**multilabel_metrics(ytest, test_pred, tm.to_prob(test_scores)), "Eval Time": f"{test_timer.elapsed:.2f}s"}

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
    parser = argparse.ArgumentParser(description="TM CelebA dataset")
    # Subcommand to select discrete or guided
    models = parser.add_subparsers(dest="model", required=True)

    # Common args
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--epochs", type=int, default=10)
    common.add_argument("--gray", action="store_true", help="use grayscale images instead of RGB")
    common.add_argument("--levels", type=int, default=8, help="thermometer levels")
    common.add_argument("--clause_drop_p", type=float, default=0.0)
    common.add_argument("--n_clauses", type=int, default=80000)
    common.add_argument("--s", type=float, default=30.0)
    common.add_argument(
        "--patch",
        dest="patch_dim",
        type=lambda v: None if v.lower() == "none" else tuple(map(int, v.split(","))),
        default="3,3",
        help="H,W or none",
    )
    common.add_argument("--seed", type=int, default=10)
    common.add_argument("--device", type=str, default="cpu:1", help="cpu:N_THREADS or cuda:GPU_ID")

    # Discrete args
    discrete = models.add_parser("discrete", parents=[common])
    discrete.add_argument("--T", type=float, default=80000)
    discrete.add_argument("--q", type=float, default=10.0)

    # Guided args
    guided = models.add_parser("guided", parents=[common])
    guided.add_argument("--lr", type=float, default=1.0)
    guided.add_argument("--lambda", dest="lambda_", type=float, default=25.0)
    guided.add_argument("--act_loss", choices=list(ACT_LOSSES), default="asl")

    # Parse args
    args = parser.parse_args()

    # Args to dict
    params = vars(args)
    model_name, epochs, clause_drop_p = params.pop("model"), params.pop("epochs"), params.pop("clause_drop_p")
    levels, gray = params.pop("levels"), params.pop("gray")
    if "act_loss" in params:
        params["act_loss"] = ACT_LOSSES[params["act_loss"]]()

    # Model Initialization
    TM = DiscreteTM if model_name == "discrete" else GuidedTM
    tm = TM(
        **params,
        dim=(64, 64, 1 if gray else 3),
        n_classes=40,
        feat_maxs=levels,
    )

    # Training
    train_model(tm, *load_celeba(levels, gray), epochs=epochs, clause_drop_p=clause_drop_p, seed=params["seed"], model_name=model_name)
