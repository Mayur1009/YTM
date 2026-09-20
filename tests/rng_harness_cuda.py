"""The RNG header compiled for the device, so CPU and CUDA streams can be compared draw for draw."""

import functools

import numpy as np

from ytm._core.utils import read_file

from .rng_harness import CORE, STUB

KERNELS = r"""
extern "C" __global__ void k_uniform(ull key, uint start, int n, float* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { uint c = start + (uint)i; out[i] = rand_uniform(key, &c); }
}
extern "C" __global__ void k_geom(ull key, uint start, int n, float p, float* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { uint c = start + (uint)i; out[i] = geom_sample(key, &c, p); }
}
extern "C" __global__ void k_hash(ull seed, ull a, ull b, ull* out) { out[0] = rng_hash(seed, a, b); }
"""


@functools.cache
def _mod():
    import cupy as cp

    code = STUB + "".join(read_file(CORE / f) for f in ("cuda.h", "common.h", "rng.h")) + KERNELS
    return cp.RawModule(code=code, backend="nvrtc", options=())


def _launch(name, n, args):
    import cupy as cp

    _mod().get_function(name)(((n + 255) // 256,), (256,), args)
    cp.cuda.Stream.null.synchronize()


def uniforms(key: int, n: int, start: int = 0) -> np.ndarray:
    import cupy as cp

    out = cp.empty(n, dtype=cp.float32)
    _launch("k_uniform", n, (np.uint64(key), np.uint32(start), np.int32(n), out))
    return out.get()


def geom(key: int, n: int, p: float, start: int = 0) -> np.ndarray:
    import cupy as cp

    out = cp.empty(n, dtype=cp.float32)
    _launch("k_geom", n, (np.uint64(key), np.uint32(start), np.int32(n), np.float32(p), out))
    return out.get()


def rng_hash(seed: int, a: int, b: int) -> int:
    import cupy as cp

    out = cp.empty(1, dtype=cp.uint64)
    _launch("k_hash", 1, (np.uint64(seed), np.uint64(a), np.uint64(b), out))
    return int(out.get()[0])
