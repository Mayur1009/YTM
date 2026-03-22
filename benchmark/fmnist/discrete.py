import numpy as np
import wandb

from ytm.continuous.multiclass import MultiClassTM
from ytm.utils import Profiler


def run(X_train_raw, Y_train, X_test_raw, Y_test, bins, seed, device, epochs, n_clauses, T, s, patch_dim, n_threads):
    # Discretize pixel values to [0, bins]
    X_train = np.asarray(bins * X_train_raw.astype(np.float32) / 255.0, dtype=np.int32)
    X_test = np.asarray(bins * X_test_raw.astype(np.float32) / 255.0, dtype=np.int32)

    tm = MultiClassTM(
        n_clauses=n_clauses,
        T=T,
        s=s,
        dim=(28, 28, 1),
        n_classes=10,
        patch_dim=patch_dim,
        stride=(1, 1),
        feat_mins=X_train.min(),
        feat_maxs=X_train.max(),
        seed=seed,
        device=device,
        n_threads=n_threads,
    )

    import dataclasses
    wandb.config.update(dataclasses.asdict(tm.args))

    for epoch in range(epochs):
        with Profiler() as p_fit:
            tm.fit(X_train, Y_train)

        with Profiler() as p_predict_test:
            test_pred, _ = tm.predict(X_test)

        with Profiler() as p_predict_train:
            train_pred, _ = tm.predict(X_train)

        fit_profile = p_fit.get_profile()
        predict_test_profile = p_predict_test.get_profile()
        predict_train_profile = p_predict_train.get_profile()

        test_acc = np.mean(Y_test == test_pred)
        train_acc = np.mean(Y_train == train_pred)

        log = {
            "epoch": epoch + 1,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "fit/time": fit_profile.elapsed,
            "fit/time_with_encode": fit_profile.elapsed,
            "fit/ram_peak": fit_profile.ram_peak,
            "fit/ram_delta": fit_profile.ram_delta,
            "predict_test/time": predict_test_profile.elapsed,
            "predict_train/time": predict_train_profile.elapsed,
        }

        if fit_profile.vram_peak is not None:
            log["fit/vram_peak"] = fit_profile.vram_peak
            log["fit/vram_delta"] = fit_profile.vram_delta
            log["predict_test/vram_peak"] = predict_test_profile.vram_peak

        wandb.log(log)

        print(
            f"Epoch {epoch + 1:>2} | "
            f"Train: {train_acc * 100:.2f}% Test: {test_acc * 100:.2f}% | "
            f"Fit: {fit_profile.elapsed:.3f}s Predict: {predict_test_profile.elapsed:.3f}s"
        )
