"""Resource profiling for benchmarks"""

import time
import tracemalloc
from typing import Dict, Any


class BenchmarkProfiler:
    """Profile memory and time for benchmark operations"""

    def __init__(self):
        self.checkpoints: Dict[str, Dict[str, float]] = {}
        self.start_time: float = 0.0
        self.baseline_memory: float = 0.0

    def __enter__(self):
        """Start profiling"""
        tracemalloc.start()
        self.start_time = time.perf_counter()
        self.baseline_memory = self._get_memory_mb()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop profiling"""
        tracemalloc.stop()
        return False

    def checkpoint(self, name: str):
        """Record time and memory at this checkpoint"""
        current_time = time.perf_counter()
        elapsed = current_time - self.start_time
        memory_mb = self._get_memory_mb()

        self.checkpoints[name] = {
            "elapsed_time": elapsed,
            "memory_mb": memory_mb,
            "memory_delta_mb": memory_mb - self.baseline_memory,
        }

    def _get_memory_mb(self) -> float:
        """Get current memory usage in MB"""
        current, peak = tracemalloc.get_traced_memory()
        return peak / 1024 / 1024  # Convert to MB

    def get_time_between(self, start_checkpoint: str, end_checkpoint: str) -> float:
        """Get time elapsed between two checkpoints"""
        if start_checkpoint not in self.checkpoints or end_checkpoint not in self.checkpoints:
            return 0.0
        return self.checkpoints[end_checkpoint]["elapsed_time"] - self.checkpoints[start_checkpoint]["elapsed_time"]

    def get_memory_at(self, checkpoint: str) -> float:
        """Get memory usage at checkpoint"""
        if checkpoint not in self.checkpoints:
            return 0.0
        return self.checkpoints[checkpoint]["memory_mb"]

    def get_peak_memory_between(self, start_checkpoint: str, end_checkpoint: str) -> float:
        """Get peak memory between two checkpoints"""
        peak = 0.0
        recording = False

        for name, data in self.checkpoints.items():
            if name == start_checkpoint:
                recording = True
            if recording:
                peak = max(peak, data["memory_mb"])
            if name == end_checkpoint:
                break

        return peak

    def get_results(self) -> Dict[str, Any]:
        """Return all metrics as dictionary"""
        return {
            "checkpoints": self.checkpoints,
            "baseline_memory_mb": self.baseline_memory,
            "total_time": self.checkpoints.get("end", {}).get("elapsed_time", 0.0),
        }
