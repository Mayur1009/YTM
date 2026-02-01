import subprocess
import tempfile
import pathlib
import ctypes
import numpy as np
from ..utils import read_file
from .base import Backend

from ctypes import POINTER, c_uint32, c_int, c_float, c_int32

int32_p = POINTER(c_int32)
uint32_p = POINTER(c_uint32)
int_p = POINTER(c_int)
float_p = POINTER(c_float)


class CPUBackend(Backend):
    def __init__(self, header: str = "", seed: int | None = None, compile_flags: list[str] | None = None):
        self.header = header
        self.seed = seed
        self.compile_flags = compile_flags if compile_flags is not None else []
        self._compile_and_load()
        self._set_rng(self.seed)

    def _compile_and_load(self):
        current_dir = pathlib.Path(__file__).parent
        code_str = read_file("impl.c", current_dir)

        with tempfile.TemporaryDirectory() as tmpdir:
            src_file = pathlib.Path(tmpdir).joinpath("impl.c")
            out_file = pathlib.Path(tmpdir).joinpath("impl.so")

            with open(src_file, "w") as f:
                f.write(self.header + "\n" + code_str)

            compile_cmd = ["gcc", "-shared", "-fPIC", "-O3", "-march=native", *self.compile_flags, "-o", str(out_file), str(src_file)]
            subprocess.run(compile_cmd, check=True)
            self._lib = ctypes.CDLL(str(out_file))

    def _set_rng(self, seed: int | None):
        # Don't know how RNG works for CPU yet. Maybe I need the implement XOR...RNG in C?
        # For now RNG is just an int storing the current RNG state.
        self.rng_dev = np.random.randint(0, 2**32 - 1) if seed is None else seed

    def _get_rng(self):
        return self.rng_dev

    def allocate(self, size: int, nbytes: int) -> np.ndarray:
        return np.zeros(size, dtype=np.uint32)

    def to_device(self, dev, host: np.ndarray) -> None:
        np.copyto(dev, host)

    def to_host(self, host: np.ndarray, dev) -> None:
        np.copyto(host, dev)

    def memset(self, dev: np.ndarray, value: int, size: int) -> None:
        dev.fill(value)

    def encode_batch(self, X: np.ndarray, encoded_X: np.ndarray, N: int, n_patches: int) -> None:
        X_ct = X.ctypes.data_as(int32_p)
        encoded_X_ct = encoded_X.ctypes.data_as(uint32_p)
        self._lib.encode_batch(X_ct, encoded_X_ct, c_int(N))
