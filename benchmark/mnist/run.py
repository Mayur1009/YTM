import dataclasses

import numpy as np
import wandb
from keras.datasets import mnist

from ytm.tm import MultiClassTM as StandardMultiClassTM
from ytm.continuous.multiclass import MultiClassTM as DiscreteMultiClassTM
from ytm.utils import Profiler

EPOCHS = 10
N_CLAUSES = 500
T = 1000
S = 10
PATCH_DIM = (10, 10)
N_CLASSES = 10
DIM = (28, 28, 1)
N_THREADS = 8


def load_data():
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    X_train = np.where(X_train.reshape((X_train.shape[0], 28 * 28)) > 75, 1, 0)
    X_test = np.where(X_test.reshape((X_test.shape[0], 28 * 28)) > 75, 1, 0)
    X_train = np.asarray(X_train, dtype=np.int8)
    X_test = np.asarray(X_test, dtype=np.int8)
    return X_train, Y_train, X_test, Y_test


def run_standard(X_train, Y_train, X_test, Y_test, seed, device):
    tm = StandardMultiClassTM(
        n_clauses=N_CLAUSES,
        T=T,
        s=S,
        dim=DIM,
        n_classes=N_CLASSES,
        patch_dim=PATCH_DIM,
        seed=seed,
        device=device,
        n_threads=N_THREADS,
    )
    wandb.config.update(dataclasses.asdict(tm.args))

    # Measure encode separately
    with Profiler() as p_encode_train:
        encoded_X_train = tm.encode(X_train)
    with Profiler() as p_encode_test:
        encoded_X_test = tm.encode(X_test)

    encode_train_profile = p_encode_train.get_profile()
    encode_test_profile = p_encode_test.get_profile()

    for epoch in range(EPOCHS):
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


def run_discrete(X_train, Y_train, X_test, Y_test, seed, device):
    tm = DiscreteMultiClassTM(
        n_clauses=N_CLAUSES,
        T=T,
        s=S,
        dim=DIM,
        n_classes=N_CLASSES,
        patch_dim=PATCH_DIM,
        stride=(1, 1),
        feat_mins=X_train.min(),
        feat_maxs=X_train.max(),
        seed=seed,
        device=device,
        n_threads=N_THREADS,
    )
    wandb.config.update(dataclasses.asdict(tm.args))

    for epoch in range(EPOCHS):
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
    run = wandb.init(group="mnist", settings=wandb.Settings(quiet=True))
    config = run.config

    approach = config.approach
    seed = config.seed
    device = config.device

    X_train, Y_train, X_test, Y_test = load_data()

    if approach == "standard":
        run_standard(X_train, Y_train, X_test, Y_test, seed, device)
    elif approach == "discrete":
        run_discrete(X_train, Y_train, X_test, Y_test, seed, device)

    wandb.finish()


if __name__ == "__main__":
    main()
