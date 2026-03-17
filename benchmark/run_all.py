#!/usr/bin/env python
"""Run all benchmarks sequentially"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime


def run_benchmark(dataset: str, device: str, dry_run: bool = False) -> bool:
    """Run single benchmark and return success status"""
    print(f"\n{'=' * 60}")
    print(f"Starting: {dataset.upper()} on {device.upper()}")
    if dry_run:
        print("DRY-RUN MODE")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 60}\n")

    try:
        cmd = [sys.executable, "run_single.py", dataset, "--device", device]
        if dry_run:
            cmd.append("--dry-run")

        subprocess.run(cmd, check=True, cwd=Path(__file__).parent)
        print(f"\n✓ Completed: {dataset} on {device}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Failed: {dataset} on {device}")
        print(f"Error: {e}")
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run all benchmarks")
    parser.add_argument("--dry-run", action="store_true", help="Run in dry-run mode with small data subsets")
    args = parser.parse_args()

    datasets = ["mnist", "fmnist", "octmnist", "cifar10"]
    devices = ["cpu", "cuda"]

    start_time = datetime.now()
    print(f"\n{'#' * 60}")
    print(f"# YTM BENCHMARK SUITE")
    if args.dry_run:
        print(f"# DRY-RUN MODE ENABLED")
    print(f"# Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# Datasets: {', '.join(datasets)}")
    print(f"# Devices: {', '.join(devices)}")
    print(f"# Total benchmarks: {len(datasets) * len(devices)}")
    print(f"{'#' * 60}\n")

    results = []
    for device in devices:
        for dataset in datasets:
            success = run_benchmark(dataset, device, dry_run=args.dry_run)
            results.append((dataset, device, success))

    # Summary
    end_time = datetime.now()
    elapsed = end_time - start_time

    print(f"\n{'#' * 60}")
    print(f"# BENCHMARK SUITE COMPLETE")
    print(f"# Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# Elapsed: {elapsed}")
    print(f"{'#' * 60}\n")

    print("Results Summary:")
    for dataset, device, success in results:
        status = "✓" if success else "✗"
        print(f"  {status} {dataset:12s} on {device:4s}")

    # Generate report
    print("\nGenerating report...")
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from benchmark.core.reporter import generate_final_report

    try:
        generate_final_report("results/raw", "README.md")
        print("✓ Report generated: README.md")
    except Exception as e:
        print(f"✗ Report generation failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
