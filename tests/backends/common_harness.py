"""Exposes the `common.h` helpers so they can be called directly.

They are `INLINE_FN`, so nothing exports them. The wrappers below are the only way to reach the
real C rather than a Python restatement of it.
"""

import ctypes
import functools
import pathlib
import subprocess
import tempfile

import numpy as np

from ytm._core._device_checks import run_compiler
from ytm._core.backends.cpu import read_file
from ytm._core.config import BaseTMConfig
from ytm._core.device_config import DeviceConfig

CORE = pathlib.Path(__file__).parents[2] / "src" / "ytm" / "_core" / "backends"

WRAPPERS = """
int w_get_feature_value(const int* X, int py, int px, int fid) {
    return get_feature_value(X, py, px, fid);
}

int w_match_patch(const int* X, int py, int px, const int* feat_bounds, const int* bounded_feat_ids, int n) {
    return match_patch(X, py, px, feat_bounds, bounded_feat_ids, n) ? 1 : 0;
}
"""


@functools.cache
def _lib(header: str) -> ctypes.CDLL:
    dev = DeviceConfig(device="cpu:1")
    assert dev._compiler is not None
    code = header + "".join(read_file(CORE / f) for f in ("cpu.h", "common.h")) + WRAPPERS

    with tempfile.NamedTemporaryFile(suffix=".c", mode="w", delete=False) as f:
        f.write(code)
        c_file = f.name
    so_file = c_file.replace(".c", ".so")

    try:
        run_compiler([dev._compiler] + dev._compiler_flags + [c_file, "-o", so_file])
    except subprocess.CalledProcessError as e:  # pragma: no cover
        raise RuntimeError(f"common harness failed to compile:\n{e.stderr.decode()}") from None
    return ctypes.CDLL(so_file)


def lib_for(cfg: BaseTMConfig) -> ctypes.CDLL:
    """One lib per model geometry, since the helpers are compiled against its defines."""
    return _lib(cfg._header)


_int_p = ctypes.POINTER(ctypes.c_int32)


def get_feature_value(cfg: BaseTMConfig, X: np.ndarray, py: int, px: int, fid: int) -> int:
    X = np.ascontiguousarray(X, dtype=np.int32)
    return lib_for(cfg).w_get_feature_value(X.ctypes.data_as(_int_p), ctypes.c_int(py), ctypes.c_int(px), ctypes.c_int(fid))


def match_patch(cfg: BaseTMConfig, X: np.ndarray, py: int, px: int, bounds: np.ndarray, ids: np.ndarray) -> bool:
    X = np.ascontiguousarray(X, dtype=np.int32)
    bounds = np.ascontiguousarray(bounds, dtype=np.int32)
    ids = np.ascontiguousarray(ids, dtype=np.int32)
    return bool(
        lib_for(cfg).w_match_patch(
            X.ctypes.data_as(_int_p),
            ctypes.c_int(py),
            ctypes.c_int(px),
            bounds.ctypes.data_as(_int_p),
            ids.ctypes.data_as(_int_p),
            ctypes.c_int(len(ids)),
        )
    )
