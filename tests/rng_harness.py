"""Compiles the shared RNG headers into a standalone lib so the C can be tested directly.

The functions in `rng.h` are `static inline`, so they are not exported. This adds thin wrappers
that fill numpy buffers, which keeps the per-call ctypes overhead out of the statistics.
"""

import ctypes
import functools
import pathlib

import numpy as np

from ytm._core.backends.toolchain import Toolchain
from ytm._core.utils import read_file

CORE = pathlib.Path(__file__).parents[1] / "src" / "ytm" / "_core" / "backends"

# Only the defines the shared headers actually reference.
STUB = """
#define S 10.0f
#define CLASSES 10
#define TOTAL_CLAUSES 1000
#define COALESCED 0
#define INCLUDE_STATE 128
#define HEIGHT 28
#define WIDTH 28
#define DEPTH 1
#define PATCH_WIDTH 10
#define STRIDE_Y 1
#define STRIDE_X 1
#define N_PATCHES_Y 19
#define N_PATCHES_X 19
#define N_PATCHES 361
#define N_RAW_PATCH_FEATS 100
#define N_LITERALS 272
#define PATCH_HEIGHT 10
#define FBOUND_T uint32_t
#define NFEAT_T uint32_t
#define PBOUND_T uint32_t
#define NLITS_T uint32_t
"""

WRAPPERS = """
void w_draw_uniform(ull key, uint start, int n, float* out) {
    uint counter = start;
    for (int i = 0; i < n; ++i) out[i] = rand_uniform(key, &counter);
}

void w_draw_raw(ull key, uint start, int n, unsigned int* out) {
    for (int i = 0; i < n; ++i) {
        uint counter = start + (uint)i;
        out[i] = (unsigned int)(mix64(key ^ (ull)counter) >> 32);
    }
}

void w_draw_geom(ull key, uint start, int n, float p, float* out) {
    uint counter = start;
    for (int i = 0; i < n; ++i) out[i] = geom_sample(key, &counter, p);
}

unsigned long long w_rng_hash(ull seed, ull a, ull b) { return rng_hash(seed, a, b); }

// how far one geom_sample call advances the counter
unsigned int w_geom_counter(ull key, float p) {
    uint counter = 0;
    geom_sample(key, &counter, p);
    return counter;
}
unsigned long long w_mix64(ull x) { return mix64(x); }
"""


@functools.cache
def _lib() -> ctypes.CDLL:
    code = STUB + "".join(read_file(CORE / f) for f in ("cpu.h", "common.h", "rng.h")) + WRAPPERS
    lib = Toolchain(openmp=False).compile(code)
    lib.w_rng_hash.restype = ctypes.c_uint64
    lib.w_geom_counter.restype = ctypes.c_uint32
    lib.w_mix64.restype = ctypes.c_uint64
    return lib


def uniforms(key: int, n: int, start: int = 0) -> np.ndarray:
    """`n` consecutive draws from one key, as `rand_uniform` would produce them in a loop."""
    out = np.empty(n, dtype=np.float32)
    _lib().w_draw_uniform(
        ctypes.c_uint64(key), ctypes.c_uint32(start), ctypes.c_int(n), out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    )
    return out


def raw32(key: int, n: int, start: int = 0) -> np.ndarray:
    """The top 32 bits of the hash, before the float conversion."""
    out = np.empty(n, dtype=np.uint32)
    _lib().w_draw_raw(
        ctypes.c_uint64(key), ctypes.c_uint32(start), ctypes.c_int(n), out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
    )
    return out


def geom(key: int, n: int, p: float, start: int = 0) -> np.ndarray:
    out = np.empty(n, dtype=np.float32)
    _lib().w_draw_geom(
        ctypes.c_uint64(key),
        ctypes.c_uint32(start),
        ctypes.c_int(n),
        ctypes.c_float(p),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    return out


def rng_hash(seed: int, a: int, b: int) -> int:
    return _lib().w_rng_hash(*(ctypes.c_uint64(v) for v in (seed, a, b)))


def mix64(x: int) -> int:
    return _lib().w_mix64(ctypes.c_uint64(x))


def geom_counter(key: int, p: float) -> int:
    """How many uniforms one `geom_sample` call consumed."""
    return _lib().w_geom_counter(ctypes.c_uint64(key), ctypes.c_float(p))
