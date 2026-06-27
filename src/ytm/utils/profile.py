import threading
import warnings
from dataclasses import dataclass
from time import perf_counter

import psutil

try:
    import pynvml

    _HAS_NVML = True
except ImportError:
    _HAS_NVML = False


class _Sampler(threading.Thread):
    """Background thread that polls RAM and VRAM usage at a fixed interval.

    Parameters
    ----------
    pid : int
        PID of the process to monitor.
    poll_rate : int, default=100
        Sampling interval in milliseconds.
    cuda_id : int, optional
        CUDA device index. If ``None``, searches all devices for ``pid``.
    """

    def __init__(self, pid: int, poll_rate: int = 100, cuda_id: int | None = None):
        super().__init__(daemon=False)
        self._process = psutil.Process(pid)
        self._pid = pid
        self._interval = poll_rate / 1000
        self._stop_event = threading.Event()
        self.cpu_samples: list[int] = []
        self.gpu_samples: list[int] = []
        self._gpu_handle = None

        if _HAS_NVML:
            try:
                pynvml.nvmlInit()
                self._gpu_handle = self._find_gpu(cuda_id)
            except pynvml.NVMLError:
                warnings.warn("NVML initialization failed. GPU metrics disabled.")

    def _find_gpu(self, cuda_id: int | None):
        if cuda_id is not None:
            return pynvml.nvmlDeviceGetHandleByIndex(cuda_id)
        count = pynvml.nvmlDeviceGetCount()
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            for proc in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
                if proc.pid == self._pid:
                    return handle
        return None

    def _sample_gpu(self) -> int:
        for proc in pynvml.nvmlDeviceGetComputeRunningProcesses(self._gpu_handle):
            if proc.pid == self._pid:
                return proc.usedGpuMemory
        return 0

    def run(self) -> None:
        """Poll memory until :meth:`stop` is called."""
        while not self._stop_event.is_set():
            self.cpu_samples.append(self._process.memory_info().rss)
            if self._gpu_handle:
                self.gpu_samples.append(self._sample_gpu())
            self._stop_event.wait(self._interval)

    def stop(self) -> None:
        """Signal the thread to stop, join it, and shut down NVML if active."""
        self._stop_event.set()
        self.join()
        if self._gpu_handle:
            pynvml.nvmlShutdown()


class Profiler:
    """Context manager for profiling CPU and GPU memory usage over time.

    Samples RAM (and optionally VRAM) at a fixed poll rate while the block
    executes. Call :meth:`get_profile` or :meth:`summary` after the block
    to inspect results. Requires ``pynvml`` for GPU metrics.

    Parameters
    ----------
    poll_rate : int, default=100
        Sampling frequency in milliseconds.
    cuda_id : int, optional
        Index of the CUDA device to monitor. If ``None``, auto-detects the
        device used by the current process.

    Examples
    --------
    >>> with Profiler() as p:
    ...     run_experiment()
    >>> p.summary()
    """

    def __init__(self, poll_rate: int = 100, cuda_id: int | None = None):
        self._poll_rate = poll_rate
        self._cuda_id = cuda_id

    def __enter__(self):
        self._sampler = _Sampler(pid=psutil.Process().pid, poll_rate=self._poll_rate, cuda_id=self._cuda_id)
        self._sampler.start()
        self._start_time = perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._end_time = perf_counter()
        self._sampler.stop()

    @property
    def _cpu_samples(self) -> list[int]:
        if not self._sampler.cpu_samples:
            raise RuntimeError("No samples collected. Was the Profiler used as a context manager?")
        return self._sampler.cpu_samples

    @property
    def _gpu_samples(self) -> list[int] | None:
        s = self._sampler.gpu_samples
        return s if s else None

    @property
    def elapsed(self) -> float:
        if not hasattr(self, "_start_time"):
            raise RuntimeError("Profiler has not been started yet.")
        if not hasattr(self, "_end_time"):
            raise RuntimeError("Profiler has not been stopped yet.")
        return self._end_time - self._start_time

    def get_profile(self) -> "Profile":
        """Return a :class:`Profile` snapshot after the context exits.

        Returns
        -------
        Profile
            Frozen dataclass with elapsed time and RAM/VRAM statistics.

        Raises
        ------
        RuntimeError
            If called before the context has exited.
        """
        cpu = self._cpu_samples
        gpu = self._gpu_samples
        ram_start = cpu[0]
        return Profile(
            elapsed=self.elapsed,
            ram_start=ram_start,
            ram_peak=max(cpu),
            ram_mean=sum(cpu) // len(cpu),
            ram_delta=cpu[-1] - ram_start,
            vram_start=gpu[0] if gpu else None,
            vram_peak=max(gpu) if gpu else None,
            vram_mean=sum(gpu) // len(gpu) if gpu else None,
            vram_delta=gpu[-1] - gpu[0] if gpu else None,
        )

    def summary(self) -> None:
        """Print a formatted summary table of the profile to stdout."""
        p = self.get_profile()
        p.summary()


def _fmt_bytes(b: int | float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(b) < 1024:
            return f"{b:.2f} {unit}"
        b /= 1024
    return f"{b:.2f} TB"


@dataclass(frozen=True)
class Profile:
    """Immutable snapshot of profiling results produced by :class:`Profiler`.

    All memory values are in bytes. VRAM fields are ``None`` if no GPU was
    detected or ``pynvml`` is not installed.
    """
    elapsed: float
    ram_start: int
    ram_peak: int
    ram_mean: int
    ram_delta: int
    vram_start: int | None = None
    vram_peak: int | None = None
    vram_mean: int | None = None
    vram_delta: int | None = None

    def summary(self) -> None:
        """Print elapsed time and peak/mean/delta RAM and VRAM to stdout."""
        rows = [
            ("Elapsed time", f"{self.elapsed:.3f} s"),
            ("RAM start", _fmt_bytes(self.ram_start)),
            ("RAM peak", _fmt_bytes(self.ram_peak)),
            ("RAM mean", _fmt_bytes(self.ram_mean)),
            ("RAM delta", _fmt_bytes(self.ram_delta)),
        ]
        if self.vram_peak is not None:
            rows += [
                ("VRAM start", _fmt_bytes(self.vram_start)),
                ("VRAM peak", _fmt_bytes(self.vram_peak)),
                ("VRAM mean", _fmt_bytes(self.vram_mean)),
                ("VRAM delta", _fmt_bytes(self.vram_delta)),
            ]
        key_width = max(len(r[0]) for r in rows)
        print(f"{'Metric':<{key_width}}  Value")
        print(f"{'─' * key_width}  {'─' * 12}")
        for key, val in rows:
            print(f"{key:<{key_width}}  {val}")
        print(f"{'─' * key_width}  {'─' * 12}\n")
