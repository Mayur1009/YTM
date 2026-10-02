"""Train discrete or guided regression TM on Diabetes (442 patients, disease progression after one year, 25-346).

Features: age, sex, bmi, bp, s1-s6 (blood serum). Raw values, not sklearn's scaled version.
`sex` is binary (1/2 -> 0/1) and kept as is; the other 9 features are discretized per feature into --bins levels
(sklearn KBinsDiscretizer). Each feature therefore has its own range, passed to the TM as a per-feature feat_maxs array.
Random 70/30 train/test split.

Usage:
    python diabetes.py {discrete,guided} [options]

Examples:
    python diabetes.py discrete
    python diabetes.py discrete --bins 4 --strategy uniform
    python diabetes.py guided --act_loss huber

Options:
    - python diabetes.py <discrete/guided> --help shows all options.
    - Common:    --epochs --bins --strategy uniform|quantile --n_clauses --s --max_includes --seed --device cpu:N|cuda:N
    - discrete:  --T
    - guided:    --lr --lambda --act_loss mse|mae|huber
"""

import argparse

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer

from ytm.discrete import RegressionTM as DiscreteTM
from ytm.guided import RegressionTM as GuidedTM
from ytm.guided.backends.act_loss import MAE, MSE, Huber

ACT_LOSSES = {"mse": MSE, "mae": MAE, "huber": Huber}
SEX = 1  # column index of the binary feature


def load(bins: int, strategy: str, seed: int):
    X, Y = load_diabetes(return_X_y=True, scaled=False)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y.astype(np.float32), test_size=0.3, random_state=seed)

    # Continuous features: per-feature bins. Sex: 1/2 -> 0/1.
    cont = [i for i in range(X.shape[1]) if i != SEX]
    d = KBinsDiscretizer(n_bins=bins, encode="ordinal", strategy=strategy)
    d.fit(X_train[:, cont])

    def encode(X):
        out = np.empty(X.shape, dtype=np.int32)
        out[:, cont] = d.transform(X[:, cont])
        out[:, SEX] = X[:, SEX] - 1
        return out

    # Highest level per feature: bins-1 for the discretized ones, 1 for sex.
    feat_maxs = np.empty(X.shape[1], dtype=np.int32)
    feat_maxs[cont] = d.n_bins_ - 1
    feat_maxs[SEX] = 1
    return encode(X_train), Y_train, encode(X_test), Y_test, feat_maxs


def regression_metrics(y, pred):
    return f"RMSE {root_mean_squared_error(y, pred):.3f}  MAE {mean_absolute_error(y, pred):.3f}  R2 {r2_score(y, pred):.3f}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TM Diabetes regression")
    # Subcommand to select discrete or guided
    models = parser.add_subparsers(dest="model", required=True)

    # Common args
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--epochs", type=int, default=50)
    common.add_argument("--bins", type=int, default=8, help="discretization levels per continuous feature")
    common.add_argument("--strategy", choices=["uniform", "quantile"], default="quantile")
    common.add_argument("--n_clauses", type=int, default=500)
    common.add_argument("--s", type=float, default=2.0)
    common.add_argument("--max_includes", type=int, default=None, help="max literals per clause (default: no limit)")
    common.add_argument("--seed", type=int, default=10)
    common.add_argument("--device", type=str, default="cpu:1", help="cpu:N_THREADS or cuda:GPU_ID")

    # Discrete args
    discrete = models.add_parser("discrete", parents=[common])
    discrete.add_argument("--T", type=float, default=1000)

    # Guided args
    guided = models.add_parser("guided", parents=[common])
    guided.add_argument("--lr", type=float, default=0.05)
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

    X_train, Y_train, X_test, Y_test, feat_maxs = load(bins, strategy, params["seed"])

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
