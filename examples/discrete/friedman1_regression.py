import numpy as np
from sklearn.datasets import make_friedman1
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from ytm.discrete.regressor import RegressionTM

rng = np.random.default_rng(43)

if __name__ == "__main__":
    N = 1000
    X, Y = make_friedman1(n_samples=N, n_features=10, noise=0.1, random_state=42)
    X = ((X / X.max()) * 300.0).astype(np.int32)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    yrange = (float(Y.min()), float(Y.max()))
    tm = RegressionTM(
        n_clauses=2000,
        T=3000,
        s=2.0,
        dim=(10, 1, 1),
        y_range=yrange,
        feat_mins=X.min(),
        feat_maxs=X.max(),
        device="cpu",
        n_threads=8,
        seed=42,
    )

    for epoch in range(200):
        tm.fit(X_train, Y_train)
        pred_train, cs_tran = tm.predict(X_train)
        pred_test, cs_test = tm.predict(X_test)
        mae_train = np.mean(np.abs(pred_train - Y_train))
        mae_test = np.mean(np.abs(pred_test - Y_test))
        r2_train = r2_score(Y_train, pred_train)
        r2_test = r2_score(Y_test, pred_test)
        mse_train = mean_squared_error(Y_train, pred_train)
        mse_test = mean_squared_error(Y_test, pred_test)
        print(
            f"Epoch {epoch + 1}, Train MAE: {mae_train:.4f}, Test MAE: {mae_test:.4f}, "
            f"Train R²: {r2_train:.4f}, Test R²: {r2_test:.4f}, "
            f"Train MSE: {mse_train:.4f}, Test MSE: {mse_test:.4f}"
        )
