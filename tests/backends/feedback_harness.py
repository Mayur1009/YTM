"""Exposes the `feedback.c` helpers so they can be called directly.

They are `INLINE_FN`, so nothing exports them. One lib is compiled per model geometry, since the
ranges and the boost behaviour come from the generated defines.
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
void w_inc_lits(int s, int e, int o, TA_STATE_T* ta) { inc_lits(s, e, o, ta); }
void w_dec_lits(int s, int e, int o, TA_STATE_T* ta) { dec_lits(s, e, o, ta); }

void w_prob_inc_lits(ull k, float p, int s, int e, int o, TA_STATE_T* ta) {
    uint c = 0;
    prob_inc_lits(k, &c, p, s, e, o, ta);
}

void w_prob_dec_lits(ull k, float p, int s, int e, int o, TA_STATE_T* ta) {
    uint c = 0;
    prob_dec_lits(k, &c, p, s, e, o, ta);
}

void w_t1a_incs(ull k, int s, int e, int o, TA_STATE_T* ta) { uint c = 0; t1a_incs(k, &c, s, e, o, ta); }
void w_t1a_decs(ull k, int s, int e, int o, TA_STATE_T* ta) { uint c = 0; t1a_decs(k, &c, s, e, o, ta); }

void w_type1a_fb(ull k, const int* Xe, int py, int px, const int* fmins, const int* loffsets, TA_STATE_T* ta) {
    uint c = 0;
    type1a_fb(k, &c, Xe, py, px, fmins, loffsets, ta);
}

void w_type1b_fb(ull k, TA_STATE_T* ta) { uint c = 0; type1b_fb(k, &c, ta); }

void w_type2_fb(const int* Xe, int py, int px, const int* fmins, const int* loffsets, TA_STATE_T* ta) {
    type2_fb(Xe, py, px, fmins, loffsets, ta);
}
"""


@functools.cache
def _lib(header: str) -> ctypes.CDLL:
    dev = DeviceConfig(device="cpu:1")
    assert dev._compiler is not None
    code = header + "".join(read_file(CORE / f) for f in ("cpu.h", "common.h", "rng.h", "feedback.h", "feedback.c")) + WRAPPERS

    with tempfile.NamedTemporaryFile(suffix=".c", mode="w", delete=False) as f:
        f.write(code)
        c_file = f.name
    so_file = c_file.replace(".c", ".so")

    try:
        run_compiler([dev._compiler] + dev._compiler_flags + [c_file, "-o", so_file])
    except subprocess.CalledProcessError as e:  # pragma: no cover
        raise RuntimeError(f"feedback harness failed to compile:\n{e.stderr.decode()}") from None
    return ctypes.CDLL(so_file)


def lib_for(cfg: BaseTMConfig) -> ctypes.CDLL:
    return _lib(cfg._header)


_int_p = ctypes.POINTER(ctypes.c_int32)


def _ta_p(cfg: BaseTMConfig):
    return ctypes.POINTER(np.ctypeslib.as_ctypes_type(cfg._ta_dtype))


def states(cfg: BaseTMConfig, fill: int) -> np.ndarray:
    """One clause worth of TA states."""
    return np.full(cfg._n_literals, fill, dtype=cfg._ta_dtype)


def _range_call(name, cfg, ta, start, end, offset, key=None, prob=None):
    lib, fn = lib_for(cfg), None
    fn = getattr(lib, name)
    args = [ctypes.c_uint64(key)] if key is not None else []
    if prob is not None:
        args.append(ctypes.c_float(prob))
    args += [ctypes.c_int(start), ctypes.c_int(end), ctypes.c_int(offset), ta.ctypes.data_as(_ta_p(cfg))]
    fn(*args)
    return ta


def inc_lits(cfg, ta, start, end, offset=0):
    return _range_call("w_inc_lits", cfg, ta, start, end, offset)


def dec_lits(cfg, ta, start, end, offset=0):
    return _range_call("w_dec_lits", cfg, ta, start, end, offset)


def prob_inc_lits(cfg, ta, prob, start, end, offset=0, key=1):
    return _range_call("w_prob_inc_lits", cfg, ta, start, end, offset, key=key, prob=prob)


def prob_dec_lits(cfg, ta, prob, start, end, offset=0, key=1):
    return _range_call("w_prob_dec_lits", cfg, ta, start, end, offset, key=key, prob=prob)


def t1a_incs(cfg, ta, start, end, offset=0, key=1):
    return _range_call("w_t1a_incs", cfg, ta, start, end, offset, key=key)


def t1a_decs(cfg, ta, start, end, offset=0, key=1):
    return _range_call("w_t1a_decs", cfg, ta, start, end, offset, key=key)


def _patch_args(cfg, X):
    return (
        np.ascontiguousarray(X, dtype=np.int32).ctypes.data_as(_int_p),
        np.ascontiguousarray(cfg._feat_mins, dtype=np.int32).ctypes.data_as(_int_p),
        np.ascontiguousarray(cfg._literal_offsets, dtype=np.int32).ctypes.data_as(_int_p),
    )


def type1a_fb(cfg, ta, X, py, px, key=1):
    p_X, p_fmins, p_loff = _patch_args(cfg, X)
    lib_for(cfg).w_type1a_fb(ctypes.c_uint64(key), p_X, ctypes.c_int(py), ctypes.c_int(px), p_fmins, p_loff, ta.ctypes.data_as(_ta_p(cfg)))
    return ta


def type1b_fb(cfg, ta, key=1):
    lib_for(cfg).w_type1b_fb(ctypes.c_uint64(key), ta.ctypes.data_as(_ta_p(cfg)))
    return ta


def type2_fb(cfg, ta, X, py, px):
    p_X, p_fmins, p_loff = _patch_args(cfg, X)
    lib_for(cfg).w_type2_fb(p_X, ctypes.c_int(py), ctypes.c_int(px), p_fmins, p_loff, ta.ctypes.data_as(_ta_p(cfg)))
    return ta
