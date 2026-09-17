import itertools

import numpy as np
import pytest
from scipy.stats import binomtest, bootstrap

from ytm._discrete import BinaryTM as DiscreteBinaryTM
from ytm._guided import BinaryTM as GuidedBinaryTM

N_BITS = [2, 3, 4]
MAX_EPOCHS = 10000
NOISE_RATE = 0.1
N_NOISY_SAMPLES = 100
N_STAT_SEEDS = 100
ACCEPTABLE_ERROR_RATE = 0.01
ACCEPTABLE_ERROR_RATE_OVERRIDES = {
    ("guided", 4): 0.06,
}

COMMON_CONFIGS = {
    2: {"n_clauses": 4, "s": 4, "dim": (2, 1, 1), "n_states": 10, "boost_tp_inc": False},
    3: {"n_clauses": 8, "s": 8, "dim": (3, 1, 1), "n_states": 10, "boost_tp_inc": False},
    4: {"n_clauses": 16, "s": 16, "dim": (4, 1, 1), "n_states": 20, "boost_tp_inc": False},
}

DISCRETE_CONFIGS = {
    2: {"T": 2},
    3: {"T": 4},
    4: {"T": 8},
}

GUIDED_CONFIGS = {
    2: {},
    3: {},
    4: {},
}

BACKENDS = ["discrete", "guided"]


def make_model(backend: str, n_bits: int, seed: int):
    cfg: dict = dict(COMMON_CONFIGS[n_bits])
    cfg["seed"] = seed
    if backend == "discrete":
        cfg.update(DISCRETE_CONFIGS[n_bits])
        return DiscreteBinaryTM(**cfg)
    cfg.update(GUIDED_CONFIGS[n_bits])
    return GuidedBinaryTM(**cfg)


def parity_truth_table(n_bits: int) -> tuple[np.ndarray, np.ndarray]:
    X = np.array(list(itertools.product((0, 1), repeat=n_bits)), dtype=np.int32)
    Y = (X.sum(axis=1) % 2).astype(np.uint32)
    return X, Y


@pytest.mark.benchmark
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("n_bits", N_BITS)
class TestXORClean:
    def test_error_rate_is_acceptable(self, backend, n_bits, request):
        seed_rng = np.random.default_rng(request.config.getoption("--seed"))
        seeds = seed_rng.integers(1, 2**31 - 1, size=N_STAT_SEEDS).tolist()

        X, Y = parity_truth_table(n_bits)
        n_rows = X.shape[0]
        max_accs = np.empty(N_STAT_SEEDS)
        for i, seed in enumerate(seeds):
            tm = make_model(backend, n_bits, seed)
            acc = max_acc = 0.0
            for _ in range(MAX_EPOCHS):
                tm.fit(X, Y)
                preds, _ = tm.predict(X)
                acc = float(np.mean(preds == Y))
                max_acc = max(max_acc, acc)
                if acc == 1.0:
                    break
            max_accs[i] = max_acc

        wrong_counts = np.round((1 - max_accs) * n_rows).astype(int)
        W, N = int(wrong_counts.sum()), N_STAT_SEEDS * n_rows

        acceptable_error_rate = ACCEPTABLE_ERROR_RATE_OVERRIDES.get((backend, n_bits), ACCEPTABLE_ERROR_RATE)
        result = binomtest(k=W, n=N, p=acceptable_error_rate, alternative="greater")
        error_ci = result.proportion_ci(confidence_level=0.95, method="exact")
        acc_ci = bootstrap((max_accs,), np.mean, confidence_level=0.95, method="BCa").confidence_interval

        assert result.pvalue >= 0.05, (
            f"backend={backend} n_bits={n_bits}: aggregate wrong-row rate {W}/{N}={W / N:.4f} significantly "
            f"exceeds the acceptable {acceptable_error_rate:.2%} (p={result.pvalue:.4g}, "
            f"95% CI on error rate=[{error_ci.low:.4f}, {error_ci.high:.4f}]); "
            f"mean accuracy 95% CI=[{acc_ci.low:.4f}, {acc_ci.high:.4f}]"
        )


@pytest.mark.benchmark
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("n_bits", N_BITS)
class TestXORNoisy:
    def test_error_rate_is_acceptable_on_clean_eval(self, backend, n_bits, request):
        seed_rng = np.random.default_rng(request.config.getoption("--seed"))
        seeds = seed_rng.integers(1, 2**31 - 1, size=N_STAT_SEEDS).tolist()

        X_full, Y_full = parity_truth_table(n_bits)
        n_rows = X_full.shape[0]
        max_accs = np.empty(N_STAT_SEEDS)
        for i, seed in enumerate(seeds):
            rng = np.random.default_rng(seed)
            idx = rng.integers(0, X_full.shape[0], size=N_NOISY_SAMPLES)
            X_train, Y_clean = X_full[idx], Y_full[idx]

            tm = make_model(backend, n_bits, seed)
            acc = max_acc = 0.0
            for _ in range(MAX_EPOCHS):
                flip = rng.random(N_NOISY_SAMPLES) < NOISE_RATE
                Y_epoch = np.where(flip, 1 - Y_clean, Y_clean).astype(np.uint32)
                tm.fit(X_train, Y_epoch)
                preds, _ = tm.predict(X_full)
                acc = float(np.mean(preds == Y_full))
                max_acc = max(max_acc, acc)
                if acc == 1.0:
                    break
            max_accs[i] = max_acc

        wrong_counts = np.round((1 - max_accs) * n_rows).astype(int)
        W, N = int(wrong_counts.sum()), N_STAT_SEEDS * n_rows

        acceptable_error_rate = ACCEPTABLE_ERROR_RATE_OVERRIDES.get((backend, n_bits), ACCEPTABLE_ERROR_RATE)
        result = binomtest(k=W, n=N, p=acceptable_error_rate, alternative="greater")
        error_ci = result.proportion_ci(confidence_level=0.95, method="exact")
        acc_ci = bootstrap((max_accs,), np.mean, confidence_level=0.95, method="BCa").confidence_interval

        assert result.pvalue >= 0.05, (
            f"backend={backend} n_bits={n_bits}: aggregate wrong-row rate on clean eval {W}/{N}={W / N:.4f} "
            f"significantly exceeds the acceptable {acceptable_error_rate:.2%} (p={result.pvalue:.4g}, "
            f"95% CI on error rate=[{error_ci.low:.4f}, {error_ci.high:.4f}]); "
            f"mean accuracy 95% CI=[{acc_ci.low:.4f}, {acc_ci.high:.4f}]"
        )
