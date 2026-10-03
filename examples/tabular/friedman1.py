"""Train discrete or guided regression TM on Friedman #1 (synthetic, sklearn make_friedman1).

Usage:
    python friedman1.py {discrete,guided} [options]

Examples:
    python friedman1.py discrete
    python friedman1.py discrete --n_features 20 --N 10000 --device cuda:0
    python friedman1.py guided --n_clauses 1000

Options:
    - python friedman1.py <discrete/guided> --help shows all options.
    - Common:    --N --n_features --noise --epochs --bins --strategy uniform|quantile --n_clauses --s --max_includes --boost_tp_inc 0|1 --boost_tp_dec 0|1 --seed --device cpu:N|cuda:N
    - discrete:  --T
    - guided:    --lr --lambda --act_loss mse|mae|huber
"""

import argparse

import numpy as np
from sklearn.datasets import make_friedman1
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer

from ytm.discrete import RegressionTM as DiscreteTM
from ytm.guided import RegressionTM as GuidedTM
from ytm.guided.backends.act_loss import MAE, MSE, Huber
from ytm.utils import Timer, print_table

ACT_LOSSES = {"mse": MSE, "mae": MAE, "huber": Huber}


def load(N: int, n_features: int, noise: float, bins: int, strategy: str, seed: int):
    X, Y = make_friedman1(n_samples=N, n_features=n_features, noise=noise, random_state=seed)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y.astype(np.float32), test_size=0.3, random_state=seed)
    d = KBinsDiscretizer(n_bins=bins, encode="ordinal", strategy=strategy)
    X_train, X_test = d.fit_transform(X_train).astype(np.int32), d.transform(X_test).astype(np.int32)
    return X_train, Y_train, X_test, Y_test, d.n_bins_ - 1


def regression_metrics(y, pred):
    return {
        "RMSE": f"{root_mean_squared_error(y, pred):.3f}",
        "MAE": f"{mean_absolute_error(y, pred):.3f}",
        "R2": f"{r2_score(y, pred):.3f}",
    }


def train_model(tm, xtrain, ytrain, xtest, ytest, epochs: int, model_name: str):
    # Standardize targets with the training mean/std, metrics are reported in original units.
    mu, sd = ytrain.mean(), ytrain.std()
    ytrain_std = (ytrain - mu) / sd

    for epoch in range(epochs):
        with (fit_timer := Timer()):
            loss = tm.fit(xtrain, ytrain_std)

        with (test_timer := Timer()):
            test_pred = tm.predict(xtest)[0] * sd + mu

        with (train_timer := Timer()):
            train_pred = tm.predict(xtrain)[0] * sd + mu

        train_log = {
            **regression_metrics(ytrain, train_pred),
            "Eval Time": f"{train_timer.elapsed:.2f}s",
            "Fit Time": f"{fit_timer.elapsed:.2f}s",
        }
        test_log = {**regression_metrics(ytest, test_pred), "Eval Time": f"{test_timer.elapsed:.2f}s"}

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
    parser = argparse.ArgumentParser(description="TM Friedman #1 regression")
    # Subcommand to select discrete or guided
    models = parser.add_subparsers(dest="model", required=True)

    # Common args
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--N", type=int, default=5000, help="dataset size (70/30 train/test)")
    common.add_argument("--n_features", type=int, default=10, help="total features, first 5 are relevant")
    common.add_argument("--noise", type=float, default=1.0, help="std of gaussian noise on y")
    common.add_argument("--epochs", type=int, default=100)
    common.add_argument("--bins", type=int, default=32, help="discretization levels per feature")
    common.add_argument("--strategy", choices=["uniform", "quantile"], default="quantile")
    common.add_argument("--n_clauses", type=int, default=100)
    common.add_argument("--s", type=float, default=5.0)
    common.add_argument("--max_includes", type=int, default=8, help="max literals per clause")
    common.add_argument("--boost_tp_inc", type=lambda v: bool(int(v)), default=False, metavar="0|1")
    common.add_argument("--boost_tp_dec", type=lambda v: bool(int(v)), default=False, metavar="0|1")
    common.add_argument("--seed", type=int, default=10)
    common.add_argument("--device", type=str, default="cpu:1", help="cpu:N_THREADS or cuda:GPU_ID")

    # Discrete args
    discrete = models.add_parser("discrete", parents=[common])
    discrete.add_argument("--T", type=float, default=5000)

    # Guided args
    guided = models.add_parser("guided", parents=[common])
    guided.add_argument("--lr", type=float, default=0.01)
    guided.add_argument("--lambda", dest="lambda_", type=float, default=30.0)
    guided.add_argument("--act_loss", choices=list(ACT_LOSSES), default="mse")
    # Parse args
    args = parser.parse_args()

    # Args to dict
    params = vars(args)
    model_name, epochs = params.pop("model"), params.pop("epochs")
    N, n_features, noise = params.pop("N"), params.pop("n_features"), params.pop("noise")
    bins, strategy = params.pop("bins"), params.pop("strategy")
    if "act_loss" in params:
        params["act_loss"] = ACT_LOSSES[params["act_loss"]]()

    X_train, Y_train, X_test, Y_test, feat_maxs = load(N, n_features, noise, bins, strategy, params["seed"])

    # Model Initialization.
    if model_name == "discrete":
        y_std = (Y_train - Y_train.mean()) / Y_train.std()
        tm = DiscreteTM(**params, dim=(n_features, 1, 1), y_range=(float(y_std.min()), float(y_std.max())), feat_maxs=feat_maxs)
    else:
        tm = GuidedTM(**params, dim=(n_features, 1, 1), feat_maxs=feat_maxs)

    # Training
    train_model(tm, X_train, Y_train, X_test, Y_test, epochs=epochs, model_name=model_name)
