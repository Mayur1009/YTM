import dataclasses

import numpy as np
import wandb
from keras.datasets import cifar10

from ytm.tm import MultiClassTM as StandardMultiClassTM
from ytm.continuous.multiclass import MultiClassTM as DiscreteMultiClassTM
from ytm.utils import Binarizer, Profiler

EPOCHS = 10
N_CLAUSES = 6000
T = 15000
S = 15
PATCH_DIM = (5, 5)
N_CLASSES = 10
N_THREADS = 32


def load_data():
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
    X_train = np.copy(X_train)
    X_test = np.copy(X_test)
    return X_train, Y_train.squeeze(), X_test, Y_test.squeeze()


def standard_run(X_train_raw, Y_train, X_test_raw, Y_test, bins, seed, device, epochs, n_clauses, T, s, patch_dim, n_threads):
    b = Binarizer(bins)
    b.fit(X_train_raw)
    X_train = b.transform(X_train_raw).reshape((X_train_raw.shape[0], -1)).astype(np.int8)
    X_test = b.transform(X_test_raw).reshape((X_test_raw.shape[0], -1)).astype(np.int8)

    tm = StandardMultiClassTM(
        n_clauses=n_clauses,
        T=T,
        s=s,
        dim=(32, 32, 3 * bins),
        n_classes=N_CLASSES,
        patch_dim=patch_dim,
        coalesced=False,
        seed=seed,
        device=device,
        n_threads=n_threads,
    )
    wandb.config.update(dataclasses.asdict(tm.args))

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


def discrete_run(X_train_raw, Y_train, X_test_raw, Y_test, bins, seed, device, epochs, n_clauses, T, s, patch_dim, n_threads):
    X_train = np.asarray(bins * X_train_raw.astype(np.float32) / 255.0, dtype=np.int32)
    X_test = np.asarray(bins * X_test_raw.astype(np.float32) / 255.0, dtype=np.int32)

    tm = DiscreteMultiClassTM(
        n_clauses=n_clauses,
        T=T,
        s=s,
        dim=(32, 32, 3),
        n_classes=N_CLASSES,
        patch_dim=patch_dim,
        stride=(1, 1),
        coalesced=False,
        feat_mins=X_train.min(),
        feat_maxs=X_train.max(),
        seed=seed,
        device=device,
        n_threads=n_threads,
    )
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


def main():
    run = wandb.init(group="cifar10", settings=wandb.Settings(quiet=True))
    config = run.config

    approach = config.approach
    seed = config.seed
    device = config.device
    bins = config.bins

    X_train, Y_train, X_test, Y_test = load_data()

    kwargs = dict(
        X_train_raw=X_train,
        Y_train=Y_train,
        X_test_raw=X_test,
        Y_test=Y_test,
        bins=bins,
        seed=seed,
        device=device,
        epochs=EPOCHS,
        n_clauses=N_CLAUSES,
        T=T,
        s=S,
        patch_dim=PATCH_DIM,
        n_threads=N_THREADS,
    )

    if approach == "standard":
        standard_run(**kwargs)
    elif approach == "discrete":
        discrete_run(**kwargs)

    wandb.finish()


if __name__ == "__main__":
    main()
