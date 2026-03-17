"""Report generation and visualization"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend

from .config import BenchmarkResult


def load_all_results(results_dir: str) -> List[BenchmarkResult]:
    """Load all JSON results from directory recursively"""
    results = []
    results_path = Path(results_dir)

    for json_file in results_path.rglob("*.json"):
        try:
            result = BenchmarkResult.load(str(json_file))
            results.append(result)
        except Exception as e:
            print(f"Warning: Could not load {json_file}: {e}")

    return results


def compute_statistics(results: List[BenchmarkResult]) -> Dict[str, Any]:
    """Compute mean, std, min, max across runs"""
    # Group by (dataset, device, implementation, bins)
    groups = defaultdict(list)

    for result in results:
        key = (result.dataset, result.device, result.implementation, result.bins)
        groups[key].append(result)

    stats = {}
    for key, group_results in groups.items():
        dataset, device, implementation, bins = key

        # Extract metrics from all runs
        all_metrics = [r.metrics for r in group_results]

        # Compute statistics for key metrics
        stats[key] = {
            "n_runs": len(group_results),
            "dataset": dataset,
            "device": device,
            "implementation": implementation,
            "bins": bins,
        }

        # Time statistics
        if "epoch_times" in all_metrics[0]:
            train_times = [np.sum(m["epoch_times"]["train"]) for m in all_metrics]
            infer_test_times = [np.sum(m["epoch_times"]["infer_test"]) for m in all_metrics]

            stats[key]["train_time_mean"] = np.mean(train_times)
            stats[key]["train_time_std"] = np.std(train_times)
            stats[key]["infer_test_time_mean"] = np.mean(infer_test_times)
            stats[key]["infer_test_time_std"] = np.std(infer_test_times)

        # Accuracy statistics
        if "epoch_acc" in all_metrics[0]:
            final_test_accs = [m["epoch_acc"]["test"][-1] for m in all_metrics]

            stats[key]["test_acc_mean"] = np.mean(final_test_accs)
            stats[key]["test_acc_std"] = np.std(final_test_accs)

        # Memory statistics
        if "memory" in all_metrics[0]:
            peak_mems = [m["memory"]["peak_mb"] for m in all_metrics]
            stats[key]["peak_memory_mean"] = np.mean(peak_mems)
            stats[key]["peak_memory_std"] = np.std(peak_mems)

            # Encoded data size (only for standard/binary implementation)
            if "encoded_data_size_mb" in all_metrics[0]["memory"]:
                encoded_sizes = [m["memory"]["encoded_data_size_mb"] for m in all_metrics]
                stats[key]["encoded_data_mb_mean"] = np.mean(encoded_sizes)
                stats[key]["encoded_data_mb_std"] = np.std(encoded_sizes)

    return stats


def generate_comparison_table(stats: Dict[str, Any], dataset: str, device: str) -> str:
    """Generate markdown table comparing implementations, separated by bins"""
    # Filter stats for this dataset and device
    relevant_stats = {k: v for k, v in stats.items() if v["dataset"] == dataset and v["device"] == device}

    if not relevant_stats:
        return f"No results found for {dataset} on {device}\n"

    # Map implementation names for display
    impl_display = {
        "standard": "StandardImpl",
        "binary": "StandardImpl",  # Legacy support
        "continuous": "ContinuousImpl",
    }

    # Group by bins
    bins_groups = {}
    for key, stat in relevant_stats.items():
        bins = stat["bins"]
        if bins not in bins_groups:
            bins_groups[bins] = []
        bins_groups[bins].append((key, stat))

    # Generate tables for each bin level
    table = f"### {dataset.upper()} on {device.upper()}\n\n"

    for bins in sorted(bins_groups.keys()):
        table += f"#### Bins: {bins}\n\n"
        table += "| Implementation | Train Time (s) | Infer Time (s) | Peak Memory (MB) | Test Acc (%) |\n"
        table += "|----------------|----------------|----------------|------------------|---------------|\n"

        # Sort by implementation name
        bin_stats = sorted(bins_groups[bins], key=lambda x: x[1]["implementation"])

        for key, stat in bin_stats:
            impl = stat["implementation"]
            impl_name = impl_display.get(impl, impl)
            train_time = stat.get("train_time_mean", 0)
            train_std = stat.get("train_time_std", 0)
            infer_time = stat.get("infer_test_time_mean", 0)
            infer_std = stat.get("infer_test_time_std", 0)
            peak_mem = stat.get("peak_memory_mean", 0)
            peak_mem_std = stat.get("peak_memory_std", 0)
            test_acc = stat.get("test_acc_mean", 0)
            acc_std = stat.get("test_acc_std", 0)

            table += f"| {impl_name:14s} | {train_time:6.2f} +/- {train_std:4.2f} | {infer_time:6.2f} +/- {infer_std:4.2f} | {peak_mem:7.1f} +/- {peak_mem_std:5.1f} | {test_acc * 100:5.2f} +/- {acc_std * 100:4.2f} |\n"

        table += "\n"

        # Add encoded data size for standard implementation if available
        for key, stat in bin_stats:
            if stat["implementation"] in ["standard", "binary"]:
                if "encoded_data_mb_mean" in stat:
                    enc_size = stat["encoded_data_mb_mean"]
                    enc_std = stat["encoded_data_mb_std"]
                    table += f"**Note**: StandardImpl encoded data size: {enc_size:.1f} +/- {enc_std:.1f} MB\n\n"
                    break

    return table


def generate_scaling_plot(stats: Dict[str, Any], dataset: str, device: str, output_dir: str):
    """Generate scaling plots for multibin datasets (time and memory)"""
    # Filter for this dataset/device and multibin results
    relevant_stats = {
        k: v for k, v in stats.items() if v["dataset"] == dataset and v["device"] == device and v["bins"] > 1
    }

    if not relevant_stats:
        return

    # Separate standard and continuous (support both old and new naming)
    standard_time = sorted(
        [
            (v["bins"], v["train_time_mean"], v["train_time_std"])
            for v in relevant_stats.values()
            if v["implementation"] in ["standard", "binary"]
        ]
    )
    continuous_time = sorted(
        [
            (v["bins"], v["train_time_mean"], v["train_time_std"])
            for v in relevant_stats.values()
            if v["implementation"] == "continuous"
        ]
    )

    standard_mem = sorted(
        [
            (v["bins"], v["peak_memory_mean"], v["peak_memory_std"])
            for v in relevant_stats.values()
            if v["implementation"] in ["standard", "binary"] and "peak_memory_mean" in v
        ]
    )
    continuous_mem = sorted(
        [
            (v["bins"], v["peak_memory_mean"], v["peak_memory_std"])
            for v in relevant_stats.values()
            if v["implementation"] == "continuous" and "peak_memory_mean" in v
        ]
    )

    if not standard_time and not continuous_time:
        return

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Training Time
    if standard_time:
        bins, times, stds = zip(*standard_time)
        ax1.errorbar(bins, times, yerr=stds, marker="o", label="StandardImpl", capsize=5, linewidth=2, markersize=8)

    if continuous_time:
        bins, times, stds = zip(*continuous_time)
        ax1.errorbar(bins, times, yerr=stds, marker="s", label="ContinuousImpl", capsize=5, linewidth=2, markersize=8)

    ax1.set_xlabel("Number of Bins", fontsize=12)
    ax1.set_ylabel("Training Time (seconds)", fontsize=12)
    ax1.set_title(f"{dataset.upper()} - Training Time vs Bins", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale("log", base=2)

    # Plot 2: Peak Memory
    if standard_mem:
        bins, mems, stds = zip(*standard_mem)
        ax2.errorbar(bins, mems, yerr=stds, marker="o", label="StandardImpl", capsize=5, linewidth=2, markersize=8)

    if continuous_mem:
        bins, mems, stds = zip(*continuous_mem)
        ax2.errorbar(bins, mems, yerr=stds, marker="s", label="ContinuousImpl", capsize=5, linewidth=2, markersize=8)

    ax2.set_xlabel("Number of Bins", fontsize=12)
    ax2.set_ylabel("Peak Memory (MB)", fontsize=12)
    ax2.set_title(f"{dataset.upper()} - Memory Usage vs Bins", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale("log", base=2)

    plt.tight_layout()

    # Save plot
    output_path = Path(output_dir) / f"{dataset}_scaling_{device}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Generated plot: {output_path}")


def generate_final_report(results_dir: str, output_file: str):
    """Generate complete benchmark report"""
    print("Loading results...")
    results = load_all_results(results_dir)

    if not results:
        print("No results found!")
        return

    print(f"Loaded {len(results)} results")

    print("Computing statistics...")
    stats = compute_statistics(results)

    # Get unique datasets and devices, with specific order for datasets
    dataset_order = ["mnist", "fmnist", "octmnist", "cifar10"]
    all_datasets = set(r.dataset for r in results)
    datasets = [d for d in dataset_order if d in all_datasets]
    # Add any other datasets not in the predefined order
    datasets.extend(sorted(all_datasets - set(datasets)))

    devices = sorted(set(r.device for r in results))

    print("Generating report...")

    # Start building markdown report
    report = "# YTM Benchmark Results: StandardImpl vs ContinuousImpl\n\n"
    report += "## Executive Summary\n\n"
    report += f"- **Datasets**: {', '.join([d.upper() for d in datasets])}\n"
    report += f"- **Devices**: {', '.join([d.upper() for d in devices])}\n"
    report += f"- **Total benchmarks**: {len(results)}\n"

    # Get max runs and epochs from results
    max_runs = max((r.run for r in results), default=0)
    max_epochs = max((r.epochs for r in results), default=0)
    report += f"- **Runs per configuration**: {max_runs}\n"
    report += f"- **Epochs per run**: {max_epochs}\n\n"

    # Compute overall performance comparison
    report += "### Performance Summary\n\n"

    # Calculate average speedup and memory savings across all configurations
    speedups = []
    memory_savings = []

    for key, stat in stats.items():
        dataset, device, impl, bins = key
        if impl == "continuous":
            # Find corresponding standard impl
            standard_key = (dataset, device, "standard", bins)
            if standard_key not in stats:
                standard_key = (dataset, device, "binary", bins)  # Legacy support

            if standard_key in stats:
                std_time = stats[standard_key].get("train_time_mean", 0)
                cont_time = stat.get("train_time_mean", 0)
                if std_time > 0 and cont_time > 0:
                    speedups.append(std_time / cont_time)

                std_mem = stats[standard_key].get("peak_memory_mean", 0)
                cont_mem = stat.get("peak_memory_mean", 0)
                if std_mem > 0 and cont_mem > 0:
                    memory_savings.append((std_mem - cont_mem) / std_mem * 100)

    if speedups:
        avg_speedup = np.mean(speedups)
        report += f"- **Average speedup**: {avg_speedup:.2f}x (ContinuousImpl vs StandardImpl)\n"

    if memory_savings:
        avg_mem_saving = np.mean(memory_savings)
        report += f"- **Average memory reduction**: {avg_mem_saving:.1f}%\n"

    report += "\n---\n\n"

    # Add configuration details
    report += "## Configurations\n\n"

    # Load config for each dataset to show parameters
    # results_dir is usually "results/raw" so we need to go up to benchmark root
    benchmark_root = Path(results_dir).parent.parent
    # But if results_dir is "results/raw/test" we need to go up 3 levels
    # Let's find the configs directory
    config_dir = benchmark_root / "configs"
    if not config_dir.exists():
        # Try one more level up
        config_dir = benchmark_root.parent / "benchmark" / "configs"
    if not config_dir.exists():
        # Last try - relative to this file
        config_dir = Path(__file__).parent.parent / "configs"
    for dataset in datasets:
        config_file = config_dir / f"{dataset}.json"
        if config_file.exists():
            import json

            with open(config_file, "r") as f:
                config_data = json.load(f)

            report += f"### {dataset.upper()}\n\n"

            # Create configuration table
            ds_config = config_data["dataset"]
            hp = config_data["hyperparams"]

            report += "| Parameter | Value |\n"
            report += "|-----------|-------|\n"
            report += f"| **Dataset Shape** | {ds_config['shape']} |\n"
            report += f"| **Number of Classes** | {ds_config['n_classes']} |\n"
            report += f"| **Dataset Type** | {ds_config['type']} |\n"
            if "threshold" in ds_config and ds_config["threshold"]:
                report += f"| **Binarization Threshold** | {ds_config['threshold']} |\n"
            report += f"| **Number of Clauses** | {hp['n_clauses']} |\n"
            report += f"| **T (Threshold)** | {hp['T']} |\n"
            report += f"| **s (Specificity)** | {hp['s']} |\n"
            report += f"| **Patch Dimension** | {hp['patch_dim']} |\n"
            report += f"| **Random Seed** | {hp['seed']} |\n"
            report += f"| **CPU Threads** | 32 |\n"
            report += f"| **CUDA GPUs** | 1 |\n"
            report += "\n"

    report += "---\n\n"
    report += "## Detailed Results\n\n"

    # Generate tables for each dataset/device combination
    plots_dir = Path(results_dir).parent / "plots"
    plots_dir.mkdir(exist_ok=True)

    for dataset in datasets:
        report += f"## Dataset: {dataset.upper()}\n\n"

        # Add scaling plot references for multibin datasets
        dataset_bins = [k[3] for k in stats.keys() if k[0] == dataset]
        is_multibin = any(b > 1 for b in dataset_bins)

        if is_multibin:
            report += "**Scaling Analysis**:\n\n"
            for device in devices:
                plot_file = f"{dataset}_scaling_{device}.png"
                plot_path = plots_dir / plot_file
                if Path(plot_path).exists() or True:  # Will be created
                    report += f"![{dataset.upper()} Scaling on {device.upper()}](results/plots/{plot_file})\n\n"

        for device in devices:
            table = generate_comparison_table(stats, dataset, device)
            report += table

            # Generate scaling plots for multibin datasets
            generate_scaling_plot(stats, dataset, device, str(plots_dir))

        report += "---\n\n"

    # Add methodology section
    report += "## Methodology\n\n"
    report += "### Benchmark Setup\n"
    report += "- **Runs per configuration**: 5\n"
    report += "- **Epochs per run**: 10\n"
    report += "- **CPU configuration**: 32 threads\n"
    report += "- **CUDA configuration**: Single GPU\n"
    report += "- **Memory profiling**: Python tracemalloc (tracks RAM usage)\n"
    report += "- **Random seed**: 42 (+ run number for variation)\n\n"

    report += "### Metrics Collected\n"
    report += "- **Time**: Data preparation, encoding (StandardImpl only), training, and inference\n"
    report += "- **Memory**: Peak RAM usage during execution, encoded data size (StandardImpl only)\n"
    report += "- **Accuracy**: Training and test accuracy after each epoch\n"
    report += "- **Statistics**: Mean +/- standard deviation across all runs\n\n"

    report += "### Implementation Details\n"
    report += "- **StandardImpl**: Pre-encodes data using thermometer encoding (for multibin)\n"
    report += "- **ContinuousImpl**: Works directly with discretized raw data (no encoding step)\n\n"

    report += "---\n\n"
    report += "## Reproducibility\n\n"
    report += "To reproduce these benchmarks:\n\n"
    report += "```bash\n"
    report += "# Run all benchmarks\n"
    report += "python benchmark/run_all.py\n\n"
    report += "# Run single benchmark\n"
    report += "python benchmark/run_single.py mnist --device cpu\n"
    report += "```\n"

    # Write report
    output_path = Path(results_dir).parent.parent / output_file
    with open(output_path, "w") as f:
        f.write(report)

    print(f"Report generated: {output_path}")
