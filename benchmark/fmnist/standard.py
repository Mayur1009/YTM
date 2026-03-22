import numpy as np
import wandb

from ytm.tm import MultiClassTM
from ytm.utils import Binarizer, Profiler


def run(X_train_raw, Y_train, X_test_raw, Y_test, bins, seed, device, epochs, n_clauses, T, s, patch_dim, n_threads):
    b = Binarizer(bins)
    b.fit(X_train_raw)
    X_train = b.transform(X_train_raw).reshape((X_train_raw.shape[0], -1)).astype(np.int8)
    X_test = b.transform(X_test_raw).reshape((X_test_raw.shape[0], -1)).astype(np.int8)

    tm = MultiClassTM(
        n_clauses=n_clauses,
        T=T,
        s=s,
        dim=(28, 28, bins),
        n_classes=10,
        patch_dim=patch_dim,
        seed=seed,
        device=device,
        n_threads=n_threads,
    )

    import dataclasses
    wandb.config.update(dataclasses.asdict(tm.args))

    # Measure encode separately
    with Profiler() as p_encode_train:
        encoded_X_train = tm.encode(X_train)
    with Profiler() as p_encode_test:
        encoded_X_test = tm.encode(X_test)

    encode_train_profile = p_encode_train.get_profile()
    encode_test_profile = p_encode_test.get_profile()

    for epoch in range(epochs):
        with Profiler() as p_fit:
            tm.fit(encoded_X_train, Y_train, is_X_encoded=True)

        with Profiler() as p_predict_test:
            test_pred, _ = tm.predict(encoded_X_test, is_X_encoded=True)

        with Profiler() as p_predict_train:
            train_pred, _ = tm.predict(encoded_X_train, is_X_encoded=True)

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
            "fit/time_with_encode": fit_profile.elapsed + encode_train_profile.elapsed,
            "fit/ram_peak": fit_profile.ram_peak,
            "fit/ram_delta": fit_profile.ram_delta,
            "predict_test/time": predict_test_profile.elapsed,
            "predict_train/time": predict_train_profile.elapsed,
            "encode/train_time": encode_train_profile.elapsed,
            "encode/train_ram_peak": encode_train_profile.ram_peak,
            "encode/test_time": encode_test_profile.elapsed,
            "encode/test_ram_peak": encode_test_profile.ram_peak,
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
