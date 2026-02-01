from .cpu import CPUBackend as CPUBackend
try:
    from .cuda import CUDABackend as CUDABackend
except ImportError:
    CUDABackend = None


