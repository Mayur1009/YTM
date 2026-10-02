"""Train discrete or guided regression TM on California Housing (8 numeric features, median house value in $100k).

Data from the Hugging Face mirror gvlassis/california_housing (same 20640 rows as sklearn), its train/test splits.
Features discretized per feature into --bins levels (sklearn KBinsDiscretizer), thermometer encoded by the TM.

Usage:
    python california_housing.py {discrete,guided} [options]

Examples:
    python california_housing.py discrete --device cuda:0
    python california_housing.py discrete --bins 32 --n_clauses 5000 --T 8000 --device cuda:0
    python california_housing.py guided --lr 0.05 --device cuda:0

Options:
    - python california_housing.py <discrete/guided> --help shows all options.
    - Common:    --epochs --bins --strategy uniform|quantile --n_clauses --s --max_includes --seed --device cpu:N|cuda:N
    - discrete:  --T
    - guided:    --lr --lambda --act_loss mse|mae|huber
"""

import argparse

import numpy as np
from datasets import load_dataset
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.preprocessing import KBinsDiscretizer

from ytm.discrete import RegressionTM as DiscreteTM
from ytm.guided import RegressionTM as GuidedTM
from ytm.guided.backends.act_loss import MAE, MSE, Huber

ACT_LOSSES = {"mse": MSE, "mae": MAE, "huber": Huber}


def load(bins: int, strategy: str):
    ds = load_dataset("gvlassis/california_housing")
    features = [c for c in ds["train"].column_names if c != "MedHouseVal"]
    X_train, X_test = (np.stack([np.array(ds[split][c]) for c in features], axis=1) for split in ("train", "test"))
    Y_train, Y_test = (np.array(ds[split]["MedHouseVal"], dtype=np.float32) for split in ("train", "test"))
    d = KBinsDiscretizer(n_bins=bins, encode="ordinal", strategy=strategy)
    X_train, X_test = d.fit_transform(X_train).astype(np.int32), d.transform(X_test).astype(np.int32)
    return X_train, Y_train, X_test, Y_test, d.n_bins_ - 1


def regression_metrics(y, pred):
    return f"RMSE {root_mean_squared_error(y, pred):.3f}  MAE {mean_absolute_error(y, pred):.3f}  R2 {r2_score(y, pred):.3f}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TM California Housing regression")
    # Subcommand to select discrete or guided
    models = parser.add_subparsers(dest="model", required=True)

    # Common args
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--epochs", type=int, default=30)
    common.add_argument("--bins", type=int, default=16, help="discretization levels per feature")
    common.add_argument("--strategy", choices=["uniform", "quantile"], default="quantile")
    common.add_argument("--n_clauses", type=int, default=2000)
    common.add_argument("--s", type=float, default=2.0)
    common.add_argument("--max_includes", type=int, default=None, help="max literals per clause (default: no limit)")
    common.add_argument("--seed", type=int, default=10)
    common.add_argument("--device", type=str, default="cpu:1", help="cpu:N_THREADS or cuda:GPU_ID")

    # Discrete args
    discrete = models.add_parser("discrete", parents=[common])
    discrete.add_argument("--T", type=float, default=3000)

    # Guided args
    guided = models.add_parser("guided", parents=[common])
    guided.add_argument("--lr", type=float, default=0.1)
    guided.add_argument("--lambda", dest="lambda_", type=float, default=1.0)
    guided.add_argument("--act_loss", choices=list(ACT_LOSSES), default="mse")

    # Parse args
    args = parser.parse_args()

    # Args to dict
    params = vars(args)
    model_name, epochs = params.pop("model"), params.pop("epochs")
    bins, strategy = params.pop("bins"), params.pop("strategy")
    if "act_loss" in params:
        params["act_loss"] = ACT_LOSSES[params["act_loss"]]()

    X_train, Y_train, X_test, Y_test, feat_maxs = load(bins, strategy)

    # Standardize targets with the training mean/std, metrics are reported in original units.
    mu, sd = Y_train.mean(), Y_train.std()
    Y_train_std = (Y_train - mu) / sd

    # Model Initialization. Discrete maps its votes [0, T] onto y_range.
    n_features = X_train.shape[1]
    if model_name == "discrete":
        tm = DiscreteTM(**params, dim=(n_features, 1, 1), y_range=(float(Y_train_std.min()), float(Y_train_std.max())), feat_maxs=feat_maxs)
    else:
        tm = GuidedTM(**params, dim=(n_features, 1, 1), feat_maxs=feat_maxs)

    # Training
    for epoch in range(epochs):
        loss = tm.fit(X_train, Y_train_std)
        train_pred, test_pred = tm.predict(X_train)[0] * sd + mu, tm.predict(X_test)[0] * sd + mu
        loss_str = f"  loss {loss:.4f}" if loss is not None else ""  # Only loss-guided TM returns loss
        print(f"{model_name} epoch {epoch + 1}/{epochs}  train {regression_metrics(Y_train, train_pred)}  test {regression_metrics(Y_test, test_pred)}{loss_str}")
