"""Main benchmark runner logic"""

import time
import numpy as np
from typing import List
from tqdm import tqdm

from ..core.profiler import BenchmarkProfiler
from ..core.config import BenchmarkConfig, BenchmarkResult
from ..datasets import get_dataset


def run_binary_tm_benchmark(
    dataset,
    config: BenchmarkConfig,
    bins: int,
    run_number: int,
    n_epochs: int = None,
) -> BenchmarkResult:
    """Run standard TM benchmark for given configuration"""

    if n_epochs is None:
        n_epochs = config.n_epochs

    print(f"\n  {'=' * 56}")
    print(f"  [StandardImpl] bins={bins}, run={run_number}")
    print(f"  {'=' * 56}")

    with BenchmarkProfiler() as prof:
        prof.checkpoint("start")

        # Load data
        X_train_raw, Y_train, X_test_raw, Y_test = dataset.load_raw_data()
        prof.checkpoint("data_loaded")

        # Prepare data for binary TM
        X_train_bin, X_test_bin = dataset.prepare_binary(X_train_raw, X_test_raw, bins)
        prof.checkpoint("data_prepared")

        # Create TM
        from ytm.tm import MultiClassTM

        # Adjust dim for thermometer encoding (bins channels)
        if config.dataset.type == "multibin":
            shape = config.dataset.shape
            # For thermometer encoding, we have bins * channels
            adjusted_dim = (shape[0], shape[1], shape[2] * bins)
        else:
            adjusted_dim = config.dataset.shape

        tm = MultiClassTM(
            n_clauses=config.hyperparams.n_clauses,
            T=config.hyperparams.T,
            s=config.hyperparams.s,
            dim=adjusted_dim,
            n_classes=config.dataset.n_classes,
            patch_dim=config.hyperparams.patch_dim,
            device=config.device,
            n_threads=config.n_threads if config.device == "cpu" else 1,
            seed=config.hyperparams.seed + run_number,  # Different seed per run
        )
        prof.checkpoint("tm_created")

        # Encode data (once before all epochs)
        print(f"    Encoding data...")
        encoded_train = tm.encode(X_train_bin)
        encoded_test = tm.encode(X_test_bin)
        prof.checkpoint("data_encoded")

        # Get encoded data size
        encoded_size_mb = (encoded_train.nbytes + encoded_test.nbytes) / 1024 / 1024

        # Training loop
        epoch_times = {"train": [], "infer_train": [], "infer_test": []}
        epoch_acc = {"train": [], "test": []}

        print(f"    Training {config.n_epochs} epochs...")
        for epoch in tqdm(range(config.n_epochs), desc="    Epochs", leave=False):
            # Training
            t_start = time.perf_counter()
            tm.fit(encoded_train, Y_train, is_X_encoded=True, shuffle=True)
            epoch_times["train"].append(time.perf_counter() - t_start)
            prof.checkpoint(f"epoch_{epoch}_trained")

            # Inference - test
            t_start = time.perf_counter()
            pred_test, _ = tm.predict(encoded_test, is_X_encoded=True)
            epoch_times["infer_test"].append(time.perf_counter() - t_start)
            epoch_acc["test"].append(float(np.mean(Y_test == pred_test)))

            # Inference - train
            t_start = time.perf_counter()
            pred_train, _ = tm.predict(encoded_train, is_X_encoded=True)
            epoch_times["infer_train"].append(time.perf_counter() - t_start)
            epoch_acc["train"].append(float(np.mean(Y_train == pred_train)))

            prof.checkpoint(f"epoch_{epoch}_complete")

        prof.checkpoint("end")

    # Package results
    profiler_results = prof.get_results()

    # Compute time metrics
    data_prep_time = prof.get_time_between("data_loaded", "data_prepared")
    encoding_time = prof.get_time_between("data_prepared", "data_encoded")
    total_train_time = sum(epoch_times["train"])
    total_infer_test_time = sum(epoch_times["infer_test"])
    total_infer_train_time = sum(epoch_times["infer_train"])

    # Compute memory metrics
    peak_memory_mb = max(cp["memory_mb"] for cp in profiler_results["checkpoints"].values())
    memory_after_encoding = prof.get_memory_at("data_encoded")

    metrics = {
        "time": {
            "data_preparation": data_prep_time,
            "encoding_total": encoding_time,
            "training_total": total_train_time,
            "inference_test_total": total_infer_test_time,
            "inference_train_total": total_infer_train_time,
            "total_time": profiler_results["total_time"],
        },
        "memory": {
            "baseline_mb": profiler_results["baseline_memory_mb"],
            "after_encoding_mb": memory_after_encoding,
            "peak_mb": peak_memory_mb,
            "encoded_data_size_mb": encoded_size_mb,
        },
        "accuracy": {
            "train_per_epoch": epoch_acc["train"],
            "test_per_epoch": epoch_acc["test"],
        },
        "epoch_times": epoch_times,
        "epoch_acc": epoch_acc,
    }

    print(f"    Final test acc: {epoch_acc['test'][-1] * 100:.2f}%")

    result = BenchmarkResult(
        dataset=config.dataset.name,
        device=config.device,
        implementation="standard",
        bins=bins,
        run=run_number,
        epochs=n_epochs,
        metrics=metrics,
    )

    print(f"  {'=' * 56}")
    print(f"  ✓ Completed: {config.dataset.name} StandardImpl bins={bins} run={run_number}")
    print(f"  {'=' * 56}\n")

    return result


def run_continuous_tm_benchmark(
    dataset,
    config: BenchmarkConfig,
    bins: int,
    run_number: int,
    n_epochs: int = None,
) -> BenchmarkResult:
    """Run continuous TM benchmark for given configuration"""

    if n_epochs is None:
        n_epochs = config.n_epochs

    print(f"\n  {'=' * 56}")
    print(f"  [ContinuousImpl] bins={bins}, run={run_number}")
    print(f"  {'=' * 56}")

    with BenchmarkProfiler() as prof:
        prof.checkpoint("start")

        # Load data
        X_train_raw, Y_train, X_test_raw, Y_test = dataset.load_raw_data()
        prof.checkpoint("data_loaded")

        # Prepare data for continuous TM
        X_train_cont, X_test_cont, feat_mins, feat_maxs = dataset.prepare_continuous(X_train_raw, X_test_raw, bins)
        prof.checkpoint("data_prepared")

        # Create TM
        from ytm.continuous import MultiClassTM

        tm = MultiClassTM(
            n_clauses=config.hyperparams.n_clauses,
            T=config.hyperparams.T,
            s=config.hyperparams.s,
            dim=config.dataset.shape,
            n_classes=config.dataset.n_classes,
            patch_dim=config.hyperparams.patch_dim,
            feat_mins=feat_mins,
            feat_maxs=feat_maxs,
            device=config.device,
            n_threads=config.n_threads if config.device == "cpu" else 1,
            seed=config.hyperparams.seed + run_number,
        )
        prof.checkpoint("tm_created")

        # No encoding step for continuous TM!
        prof.checkpoint("encoding_skipped")

        # Training loop
        epoch_times = {"train": [], "infer_train": [], "infer_test": []}
        epoch_acc = {"train": [], "test": []}

        print(f"    Training {n_epochs} epochs...")
        for epoch in tqdm(range(n_epochs), desc="    Epochs", leave=False):
            # Training
            t_start = time.perf_counter()
            tm.fit(X_train_cont, Y_train, shuffle=True)
            epoch_times["train"].append(time.perf_counter() - t_start)
            prof.checkpoint(f"epoch_{epoch}_trained")

            # Inference - test
            t_start = time.perf_counter()
            pred_test, _ = tm.predict(X_test_cont)
            epoch_times["infer_test"].append(time.perf_counter() - t_start)
            epoch_acc["test"].append(float(np.mean(Y_test == pred_test)))

            # Inference - train
            t_start = time.perf_counter()
            pred_train, _ = tm.predict(X_train_cont)
            epoch_times["infer_train"].append(time.perf_counter() - t_start)
            epoch_acc["train"].append(float(np.mean(Y_train == pred_train)))

            prof.checkpoint(f"epoch_{epoch}_complete")

        prof.checkpoint("end")

    # Package results
    profiler_results = prof.get_results()

    # Compute time metrics
    data_prep_time = prof.get_time_between("data_loaded", "data_prepared")
    total_train_time = sum(epoch_times["train"])
    total_infer_test_time = sum(epoch_times["infer_test"])
    total_infer_train_time = sum(epoch_times["infer_train"])

    # Compute memory metrics
    peak_memory_mb = max(cp["memory_mb"] for cp in profiler_results["checkpoints"].values())

    metrics = {
        "time": {
            "data_preparation": data_prep_time,
            "encoding_total": 0.0,  # No encoding!
            "training_total": total_train_time,
            "inference_test_total": total_infer_test_time,
            "inference_train_total": total_infer_train_time,
            "total_time": profiler_results["total_time"],
        },
        "memory": {
            "baseline_mb": profiler_results["baseline_memory_mb"],
            "after_encoding_mb": 0.0,  # No encoding!
            "peak_mb": peak_memory_mb,
            "encoded_data_size_mb": 0.0,  # No encoded data!
        },
        "accuracy": {
            "train_per_epoch": epoch_acc["train"],
            "test_per_epoch": epoch_acc["test"],
        },
        "epoch_times": epoch_times,
        "epoch_acc": epoch_acc,
    }

    print(f"    Final test acc: {epoch_acc['test'][-1] * 100:.2f}%")

    result = BenchmarkResult(
        dataset=config.dataset.name,
        device=config.device,
        implementation="continuous",
        bins=bins,
        run=run_number,
        epochs=n_epochs,
        metrics=metrics,
    )

    print(f"  {'=' * 56}")
    print(f"  ✓ Completed: {config.dataset.name} ContinuousImpl bins={bins} run={run_number}")
    print(f"  {'=' * 56}\n")

    return result


def run_full_benchmark(config: BenchmarkConfig, test_mode: bool = False) -> List[BenchmarkResult]:
    """Run complete benchmark: both implementations, all bins, all runs

    Args:
        config: Benchmark configuration
        test_mode: If True, run with small data subset (dry-run mode)
    """

    results = []
    dataset = get_dataset(config.dataset.name, config.dataset, test_mode=test_mode, test_samples=100)

    # Determine bins to test
    bins_to_test = config.dataset.bins
    n_runs = config.n_runs
    n_epochs = config.n_epochs

    # In dry-run mode, reduce workload significantly
    if test_mode:
        bins_to_test = bins_to_test[:2] if len(bins_to_test) > 2 else bins_to_test
        n_runs = 1  # Only 1 run in dry-run mode
        n_epochs = 1  # Only 1 epoch in dry-run mode

    print(f"\n{'#' * 60}")
    print(f"# BENCHMARK: {config.dataset.name.upper()} on {config.device.upper()}")
    if test_mode:
        print(f"# MODE: DRY-RUN (100 train / 20 test, 1 run, 1 epoch)")
    else:
        print(f"# MODE: FULL")
    print(f"# Bins: {bins_to_test}")
    print(f"# Runs: {n_runs}, Epochs: {n_epochs}")
    print(f"{'#' * 60}\n")

    # Run standard TM
    print(f"\n{'*' * 60}")
    print(f"* Running StandardImpl benchmarks...")
    print(f"{'*' * 60}")
    for bins in bins_to_test:
        for run in range(1, n_runs + 1):
            try:
                result = run_binary_tm_benchmark(dataset, config, bins, run, n_epochs)
                results.append(result)
            except Exception as e:
                print(f"\n  ✗ ERROR in StandardImpl (bins={bins}, run={run}): {e}")
                import traceback

                traceback.print_exc()

    # Run continuous TM
    print(f"\n{'*' * 60}")
    print(f"* Running ContinuousImpl benchmarks...")
    print(f"{'*' * 60}")
    for bins in bins_to_test:
        for run in range(1, n_runs + 1):
            try:
                result = run_continuous_tm_benchmark(dataset, config, bins, run, n_epochs)
                results.append(result)
            except Exception as e:
                print(f"\n  ✗ ERROR in ContinuousImpl (bins={bins}, run={run}): {e}")
                import traceback

                traceback.print_exc()

    print(f"\n{'#' * 60}")
    print(f"# COMPLETED: {len(results)} benchmarks for {config.dataset.name.upper()}")
    print(f"{'#' * 60}\n")

    return results
