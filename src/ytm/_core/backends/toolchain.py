"""C compiler setup and build cache for the cpu backend.

The compiler and flags are picked once per machine by building small probe libraries, and the
choice is cached. Compiled kernels are cached too, keyed by their source, the compiler, the flags
and the host name, so a cache shared between machines never loads a library built for another CPU.

Environment variables
---------------------
CC
    Compiler command. Default: the first of ``clang``, ``gcc``, ``cc`` on the PATH.
CFLAGS, LDFLAGS
    Extra compile and link flags, appended after the defaults.
YTM_CACHE_DIR
    Cache directory. Default: ``$XDG_CACHE_HOME/ytm`` or ``~/.cache/ytm``.
YTM_NO_CACHE
    Set to turn both caches off.
YTM_CACHE_SRC
    Set to also write each kernel's C source next to its cached library.
OMP_PREFIX, HOMEBREW_PREFIX
    macOS only: extra places to look for ``libomp``.
"""

import hashlib
import itertools
import json
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
_PROBE = "int ytm_probe(int a) {return 1;}"
_OMP_PROBE = """
#include <omp.h>
#define NC 4
int ytm_probe(int requested) {
    float acc[NC] = {0};
#pragma omp parallel for schedule(dynamic) reduction(+ : acc[ : NC]) num_threads(requested)
    for (int i = 0; i < 256; ++i) acc[i % NC] += 1.0f;

    int seen = 0;
#pragma omp parallel reduction(max : seen) num_threads(requested)
    { seen = omp_get_num_threads(); }

    return acc[0] == 64.0f && seen >= requested;
}
"""


def _flag_candidates() -> list[tuple[list[str], list[str]]]:
    cflags = ["-fPIC", "-O3", "-march=native", "-mtune=native", *shlex.split(os.environ.get("CFLAGS", ""))]
    ldflags = ["-shared", "-lm", *shlex.split(os.environ.get("LDFLAGS", ""))]
    out = [(list(cflags), list(ldflags))]
    if platform.system() == "Darwin" and pathlib.Path("/usr/bin/ld").exists():
        out.append((list(cflags), ldflags + ["-fuse-ld=/usr/bin/ld"]))
    return out


def _darwin_openmp_prefixes() -> list[pathlib.Path]:
    if platform.system() != "Darwin":
        return []

    brew = os.environ.get("HOMEBREW_PREFIX", None)
    raw = [
        os.environ.get("OMP_PREFIX"),
        os.environ.get("CONDA_PREFIX"),
        sys.prefix,
        f"{brew}/opt/libomp" if brew is not None else None,
        "/usr/local/opt/libomp",
        "/usr/local",
    ]

    out: list[pathlib.Path] = []
    for entry in raw:
        if entry is not None:
            prefix = pathlib.Path(entry)
            if prefix not in out and (prefix / "include" / "omp.h").exists() and (prefix / "lib" / "libomp.dylib").exists():
                out.append(prefix)
    return out


def _omp_flag_candidates() -> list[tuple[list[str], list[str], pathlib.Path | None]]:
    if platform.system() != "Darwin":
        return [(["-fopenmp"], [], None)]

    out: list[tuple[list[str], list[str], pathlib.Path | None]] = [(["-Xpreprocessor", "-fopenmp"], ["-Wl,-undefined,dynamic_lookup"], None)]
    for prefix in _darwin_openmp_prefixes():
        out.append(
            (
                ["-Xpreprocessor", "-fopenmp", f"-I{prefix / 'include'}"],
                ["-Wl,-undefined,dynamic_lookup"],
                prefix / "lib" / "libomp.dylib",
            )
        )
    return out


def _resolve_cachedir() -> pathlib.Path | None:
    if os.environ.get("YTM_NO_CACHE"):
        return None
    if directory := os.environ.get("YTM_CACHE_DIR"):
        cachedir = pathlib.Path(directory)
    elif xdg := os.environ.get("XDG_CACHE_HOME"):
        cachedir = pathlib.Path(xdg) / "ytm"
    elif os.environ.get("HOME"):
        cachedir = pathlib.Path.home() / ".cache" / "ytm"
    else:
        return None

    try:
        cachedir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    return cachedir


def compiler_cmd() -> list[str]:
    if cc := os.environ.get("CC"):
        return shlex.split(cc)
    for name in ("clang", "gcc", "cc"):
        if shutil.which(name):
            return [name]
    raise OSError("`device='cpu'` needs a C compiler. Set $CC, or put `clang` or `gcc` on the PATH.")


class Toolchain:
    """Builds C code into shared libraries for the cpu backend.

    Parameters
    ----------
    openmp : bool, default=True
        Try to build with OpenMP. If no flags work, it builds without, and
        ``is_openmp_working`` is False with the reasons in ``omp_failures``.
    """

    def __init__(self, openmp: bool = True):
        self.openmp = openmp
        self._base_flags = _flag_candidates()
        self._omp_flags = _omp_flag_candidates() if openmp else [([], [], None)]

        self.tempdir = pathlib.Path(tempfile.mkdtemp(prefix="ytm-"))
        weakref.finalize(self, shutil.rmtree, self.tempdir, ignore_errors=True)

        self.save_src = bool(os.environ.get("YTM_CACHE_SRC"))
        self.cachedir = _resolve_cachedir()
        self.comp = compiler_cmd()
        self.omp_failures: list[str] = []
        self._decide_flags()

    def _decide_flags(self) -> None:
        if (cached := self._load_decision()) is not None:
            self.cflags, self.ldflags, self.omp_runtime, self.is_openmp_working = cached
            if not self.is_openmp_working:
                self.omp_failures = [f"  cached decision in {self._cache_path()}, set YTM_NO_CACHE=1 to disable cache and get errors."]
            return

        found = self._resolve_flags(self._omp_flags, self.omp_failures) if self.openmp else None
        self.is_openmp_working = found is not None
        if found is None:
            found = self._resolve_flags([([], [], None)], self.omp_failures)

        base_cf, base_lf = self._base_flags[0]
        self.cflags, self.ldflags, self.omp_runtime = found or (list(base_cf), list(base_lf), None)
        self._cache_toolchain()

    def _cache_key(self) -> str:
        found = shutil.which(self.comp[0])
        stat = pathlib.Path(found).stat() if found else None

        parts = [
            *self.comp,
            found or "",
            f"{stat.st_size}:{stat.st_mtime_ns}" if stat else "",
            platform.system(),
            platform.machine(),
            platform.node(),
            sys.prefix,
            str(self.openmp),
            repr(self._base_flags),
            repr(self._omp_flags),
        ]

        h = hashlib.sha256()
        for part in parts:
            h.update(part.encode())
            h.update(b"\0")  # keeps ("ab", "c") from hashing like ("a", "bc")
        return h.hexdigest()

    def _cache_path(self) -> pathlib.Path | None:
        return None if self.cachedir is None else self.cachedir / f"toolchain-{self._cache_key()}.json"

    def _load_decision(self):
        path = self._cache_path()
        if path is None:
            return None
        try:
            stored = json.loads(path.read_text())
            cflags, ldflags = list(stored["cflags"]), list(stored["ldflags"])
            runtime = stored["omp_runtime"]
            runtime = None if runtime is None else pathlib.Path(runtime)
            working = bool(stored["is_openmp_working"])
        except (OSError, ValueError, KeyError, TypeError):
            return None

        if runtime is not None:
            try:
                CDLL(str(runtime), mode=RTLD_GLOBAL)
            except OSError:
                return None

        return cflags, ldflags, runtime, working

    def _cache_toolchain(self) -> None:
        path = self._cache_path()
        if path is None or (self.is_openmp_working and self.omp_runtime is None and platform.system() == "Darwin"):
            return

        payload = {
            "cflags": self.cflags,
            "ldflags": self.ldflags,
            "omp_runtime": None if self.omp_runtime is None else str(self.omp_runtime),
            "is_openmp_working": self.is_openmp_working,
        }
        try:
            staged = path.with_name(f"{path.name}.{os.getpid()}")
            staged.write_text(json.dumps(payload, indent=2))
            staged.replace(path)
        except OSError:
            pass

    def _run_probe(self, cflags: list[str], ldflags: list[str], tag: str, is_omp: bool) -> bool:
        probe_cflags = [f for f in cflags if not f.startswith(("-O", "-march", "-mtune"))] + ["-O0"]
        lib = self._build(_OMP_PROBE if is_omp else _PROBE, probe_cflags, ldflags, tag)
        lib.ytm_probe.restype = c_int
        lib.ytm_probe.argtypes = [c_int]
        return bool(lib.ytm_probe(min(2, os.cpu_count() or 1)))

    def _resolve_flags(self, candidates, failures: list[str]):
        for tag, ((omp_cf, omp_lf, preload), (base_cf, base_lf)) in enumerate(itertools.product(candidates, self._base_flags)):
            cand_cf, cand_lf = base_cf + omp_cf, base_lf + omp_lf
            is_omp = bool(omp_cf)
            what = f"  {' '.join(cand_cf + cand_lf)}"

            if preload is not None:
                try:
                    CDLL(str(preload), mode=RTLD_GLOBAL)
                except OSError as e:
                    failures.append(f"{what}: {preload} will not load ({e})")
                    continue
            try:
                if not self._run_probe(cand_cf, cand_lf, f"probe{tag}", is_omp):
                    failures.append(f"{what}: built and loaded, but did not work")
                    continue
            except (RuntimeError, OSError, AttributeError) as e:
                failures.append(f"{what}: {e}")
                continue

            return cand_cf, cand_lf, preload
        return None

    @staticmethod
    def _run(cmd: list[str]) -> None:
        proc = subprocess.run(cmd, capture_output=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"{' '.join(cmd)}\n{proc.stderr.decode(errors='replace').strip()}")

    def _compile_to(self, code: str, cflags: list[str], ldflags: list[str], src: pathlib.Path, out: pathlib.Path) -> None:
        src.write_text(code)
        self._run([*self.comp, *cflags, *ldflags, str(src), "-o", str(out)])

    def _build(self, code: str, cflags: list[str], ldflags: list[str], suffix="") -> CDLL:
        d = self.tempdir / f"ytm{suffix}"
        src, lib = d.with_suffix(".c"), d.with_suffix(_LIB_SUFFIX)
        self._compile_to(code, cflags, ldflags, src, lib)
        return CDLL(str(lib))

    def _artifact_key(self, code: str) -> str:
        h = hashlib.sha256()
        for part in (code, *self.comp, *self.cflags, *self.ldflags, platform.machine(), platform.node()):
            h.update(part.encode())
            h.update(b"\0")
        return h.hexdigest()

    def compile(self, code: str) -> CDLL:
        """Build ``code`` into a shared library and load it, reusing a cached build if there is one."""
        if self.cachedir is None:
            return self._build(code, self.cflags, self.ldflags)

        lib = self.cachedir / f"{self._artifact_key(code)}{_LIB_SUFFIX}"
        if self.save_src:
            try:
                lib.with_suffix(".c").write_text(code)
            except OSError:
                pass

        if lib.exists():
            try:
                return CDLL(str(lib))
            except OSError:
                lib.unlink(missing_ok=True)

        try:
            staged = lib.with_name(f"{lib.name}.{os.getpid()}")
            src = self.tempdir / f"{lib.stem}.c"
            self._compile_to(code, self.cflags, self.ldflags, src, staged)
            staged.replace(lib)
        except OSError:
            return self._build(code, self.cflags, self.ldflags)

        return CDLL(str(lib))
