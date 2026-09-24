import hashlib
import os
import pathlib
import platform
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from ctypes import CDLL

from .._device_checks import run_compiler

_SUFFIX = ".dll" if platform.system() == "Windows" else ".so"


def resolve_cache_dir(env: Mapping[str, str] | None = None) -> pathlib.Path:
    """Resolve the cache directory. Does not create it."""
    env = os.environ if env is None else env
    if directory := env.get("YTM_CACHE_DIR"):
        return pathlib.Path(directory)
    if xdg := env.get("XDG_CACHE_HOME"):
        return pathlib.Path(xdg) / "ytm"
    if pathlib.Path.home().exists():
        return pathlib.Path.home() / ".cache" / "ytm"
    return pathlib.Path("/tmp/ytm")


def cache_key(code: str, compiler: str, flags: Sequence[str]) -> str:
    """Return the cache key for one compilation, based on the code, compiler, flags, and machine."""
    h = hashlib.sha256()
    for part in (code, compiler, *flags, platform.machine(), platform.node()):
        h.update(part.encode())
        h.update(b"\0")  # keeps ("ab", "c") from hashing like ("a", "bc")
    return h.hexdigest()


def _compile(code: str, compiler: str, flags: Sequence[str]) -> pathlib.Path:
    """Compile `code`, return the so path."""
    with tempfile.NamedTemporaryFile(suffix=".c", mode="w", delete=False) as f:
        f.write(code)
        c_file = pathlib.Path(f.name)

    with tempfile.NamedTemporaryFile(suffix=_SUFFIX, delete=False) as f:
        so_file = pathlib.Path(f.name)

    try:
        run_compiler([compiler, *flags, str(c_file), "-o", str(so_file)])
    except subprocess.CalledProcessError as e:
        so_file.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to compile. Compiler output:\n{e.stdout.decode()}\nError: {e.stderr.decode()}") from None
    finally:
        c_file.unlink(missing_ok=True)

    return so_file


def load_library(code: str, compiler: str, flags: Sequence[str], env: Mapping[str, str] | None = None) -> CDLL:
    """Return a loaded shared library for `code`, compiling only on a cache miss."""
    env = os.environ if env is None else env
    disable_cache = bool(env.get("YTM_NO_CACHE"))
    key = cache_key(code, compiler, flags)

    cache_usable = not disable_cache

    if cache_usable:
        cache_dir = resolve_cache_dir(env)
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            cache_usable = False

    target = cache_dir / f"{key}{_SUFFIX}" if cache_usable else None
    if target is not None and target.exists():
        try:
            return CDLL(str(target))
        except OSError:
            target.unlink(missing_ok=True)

    so_file = _compile(code, compiler, flags)
    lib = CDLL(str(so_file))

    if target is not None:
        staged = cache_dir / f".{key}.{os.getpid()}"
        shutil.copyfile(so_file, staged)
        staged.replace(target)
        so_file.unlink(missing_ok=True)

        if bool(env.get("YTM_SAVE_SOURCE")):
            with open(cache_dir / f"{key}.c", "w") as f:
                f.write(code)

    return lib
