#!/usr/bin/env python
"""Run benchmark for a single dataset"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.core.config import BenchmarkConfig
from benchmark.benchmarks.runner import run_full_benchmark


def main():
    parser = argparse.ArgumentParser(description="Run single dataset benchmark")
    parser.add_argument("dataset", choices=["mnist", "fmnist", "octmnist", "cifar10"], help="Dataset to benchmark")
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True, help="Device to use")
    parser.add_argument("--config", type=str, help="Override default config file")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in dry-run mode with small data subset (100 train / 20 test samples, 2 bins)",
    )
    args = parser.parse_args()

    # Determine config file
    config_file = args.config or f"configs/{args.dataset}.json"
    config_path = Path(__file__).parent / config_file

    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    # Load config
    print(f"Loading config from: {config_path}")
    config = BenchmarkConfig.load(str(config_path))

    # Override device and threads
    config.device = args.device
    config.n_threads = 32 if args.device == "cpu" else 1

    # Run benchmark
    if args.dry_run:
        print("\n" + "=" * 60)
        print("  DRY-RUN MODE ENABLED")
        print("  - Using 100 train / 20 test samples")
        print("  - Testing only first 2 bins")
        print("  - Only 1 run × 1 epoch (very fast!)")
        print("  - This should complete in 10-60 seconds")
        print("=" * 60 + "\n")

    results = run_full_benchmark(config, test_mode=args.dry_run)

    # Save results
    output_dir = Path(__file__).parent / "results" / "raw" / args.device
    output_dir.mkdir(parents=True, exist_ok=True)

    for result in results:
        filename = f"{result.dataset}_{result.implementation}_bins{result.bins}_run{result.run}.json"
        filepath = output_dir / filename
        result.save(str(filepath))

    print(f"\n{'=' * 60}")
    print(f"✓ Benchmark complete!")
    print(f"✓ Saved {len(results)} results to: {output_dir}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
