import ctypes.util
import os
import pathlib
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
import weakref
from ctypes import CDLL, RTLD_GLOBAL, c_int

_LIB_SUFFIX = ".dll" if platform.system() == "Windows" else ".so"
_OMP_PROBE = """
#include <omp.h>
int ytm_omp_probe(int requested) {
    int restore = omp_get_max_threads();
    omp_set_num_threads(requested);

    int seen = 0;
#pragma omp parallel reduction(max : seen)
    { seen = omp_get_num_threads(); }

    omp_set_num_threads(restore);
    return seen;
}
"""


class Toolchain:
    def __init__(self):
        self.req_cflags = ["-fPIC"]
        self.def_cflags = ["-O3", "-march=native", "-mtune=native"]

        self.req_ldflags = ["-shared"]
        if platform.system() == "Darwin" and pathlib.Path("/usr/bin/ld").exists():
            self.req_ldflags.append("-fuse-ld=/usr/bin/ld")
        self.def_ldflags = ["-lm"]

        self.tempdir = pathlib.Path(tempfile.mkdtemp(prefix="ytm-"))
        weakref.finalize(self, shutil.rmtree, self.tempdir, ignore_errors=True)

        self.compiler = self._resolve_compiler()
        self.comp_name = self._compiler_family()
        self.cflags: list[str] = self._resolve_cflags()
        self.ldflags: list[str] = self._resolve_ldflags()

        self.omp_failures: list[str] = []
        self.is_openmp_working = self._resolve_openmp()

    def _resolve_compiler(self) -> list[str]:
        if cc := os.environ.get("CC"):
            return shlex.split(cc)
        for name in ("clang", "gcc", "cc"):
            if shutil.which(name):
                return [name]
        raise OSError("`device='cpu'` needs a C compiler. Set $CC, or put `clang` or `gcc` on the PATH.")

    def _compiler_family(self):
        proc = subprocess.run([*self.compiler, "--version"], capture_output=True, check=False)
        first = proc.stdout.decode(errors="replace").strip().splitlines()
        return "clang" if first and "clang" in first[0].lower() else "gcc"

    def _resolve_cflags(self) -> list[str]:
        return [*self.req_cflags, *self.def_cflags, *shlex.split(os.environ.get("CFLAGS", ""))]

    def _resolve_ldflags(self) -> list[str]:
        return [*self.req_ldflags, *self.def_ldflags, *shlex.split(os.environ.get("LDFLAGS", ""))]

    def _discover_runtime(self) -> pathlib.Path | None:
        src = self.tempdir / "discover.c"
        out = self.tempdir / f"discover{_LIB_SUFFIX}"
        src.write_text(_OMP_PROBE)
        cmd = [*self.compiler, *self.cflags, "-fopenmp", *self.ldflags, "-v", str(src), "-o", str(out)]
        proc = subprocess.run(cmd, capture_output=True, check=False)
        if proc.returncode != 0:
            return None

        tokens = proc.stderr.decode(errors="replace").split()
        names = [t[2:] for t in tokens if t.startswith("-l") and t[2:] in ("omp", "gomp", "iomp5")]
        dirs = [pathlib.Path(t[2:]) for t in tokens if t.startswith("-L") and len(t) > 2]

        for name in names:
            for directory in dirs:
                found = [p for p in sorted(directory.glob(f"lib{name}.*")) if p.is_file() and (".so" in p.name or p.suffix == ".dylib")]
                if found:
                    return found[0].resolve()
            if located := ctypes.util.find_library(name):
                return pathlib.Path(located)
        return None

    def _find_runtime_by_name(self) -> pathlib.Path | None:
        base = "omp" if self.comp_name == "clang" else "gomp"
        if located := ctypes.util.find_library(base):
            return pathlib.Path(located)

        for name in (f"lib{base}.dylib", f"lib{base}.so.1", f"lib{base}.so"):
            proc = subprocess.run([*self.compiler, f"-print-file-name={name}"], capture_output=True, check=False)
            found = pathlib.Path(proc.stdout.decode(errors="replace").strip())
            if found.is_absolute() and found.exists():
                return found.resolve()
        return None

    def _resolve_openmp(self) -> bool:
        self.omp_runtime = self._discover_runtime() or self._find_runtime_by_name()

        if self.omp_runtime is None:
            self.omp_failures.append(f"  no OpenMP runtime found for {' '.join(self.compiler)} ({self.comp_name})")
            return False

        try:
            CDLL(str(self.omp_runtime), mode=RTLD_GLOBAL)
        except OSError as e:
            self.omp_failures.append(f"  {self.omp_runtime}: will not load ({e})")
            return False

        expected = min(2, os.cpu_count() or 1)
        include = f"-I{pathlib.Path(sys.prefix) / 'include'}"
        omp_ldflags = ["-Wl,-undefined,dynamic_lookup"] if platform.system() == "Darwin" else []

        for i, omp_cflags in enumerate((["-Xpreprocessor", "-fopenmp", include], ["-fopenmp", include])):
            what = f"  {' '.join(omp_cflags)}"
            try:
                lib = self._build(_OMP_PROBE, [*self.cflags, *omp_cflags], [*self.ldflags, *omp_ldflags], f"omp{i}")
                lib.ytm_omp_probe.restype = c_int
                lib.ytm_omp_probe.argtypes = [c_int]
                seen = lib.ytm_omp_probe(expected)
            except Exception as e:
                self.omp_failures.append(f"{what}: {e}")
                continue

            if seen < expected:
                self.omp_failures.append(f"{what}: ran on {seen} thread(s), expected at least {expected}")
                continue

            self.cflags += omp_cflags
            self.ldflags += omp_ldflags
            return True

        return False

    @staticmethod
    def _run(cmd: list[str]) -> None:
        proc = subprocess.run(cmd, capture_output=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"{' '.join(cmd)}\n{proc.stderr.decode(errors='replace').strip()}")

    def _build(self, code: str, cflags: list[str], ldflags: list[str], suffix="") -> CDLL:
        d = self.tempdir / f"ytm{suffix}"
        src, obj, lib = d.with_suffix(".c"), d.with_suffix(".o"), d.with_suffix(_LIB_SUFFIX)
        src.write_text(code)
        self._run([*self.compiler, "-c", *cflags, str(src), "-o", str(obj)])
        self._run([*self.compiler, *ldflags, str(obj), "-o", str(lib)])
        return CDLL(str(lib))

    def compile(self, code: str) -> CDLL:
        return self._build(code, self.cflags, self.ldflags)
