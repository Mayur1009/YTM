"""Benchmark execution logic"""

from .runner import run_binary_tm_benchmark, run_continuous_tm_benchmark, run_full_benchmark

__all__ = [
    "run_binary_tm_benchmark",
    "run_continuous_tm_benchmark",
    "run_full_benchmark",
]
