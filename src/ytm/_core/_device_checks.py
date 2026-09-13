import importlib.util
import os
import platform
import shutil
import subprocess
import tempfile
import warnings

# Tried in order, first one that compiles the probe below wins.
OMP_FLAG_CANDIDATES = {
    "gcc": [["-fopenmp"]],
    "clang": [
        ["-fopenmp"],  # llvm clang, libomp on the default search path
        ["-fopenmp", "-lomp"],  # llvm clang, driver does not auto-link libomp
        ["-Xpreprocessor", "-fopenmp", "-lomp"],  # apple clang + brew/conda libomp
    ],
}

DEFAULT_COMPILE_FLAGS = ["-shared", "-fPIC", "-lm", "-O3", "-march=native", "-mtune=native"]

# Tried in order, first one that links the probe below wins.
LINK_FLAG_CANDIDATES = {
    "clang": [[], ["-fuse-ld=/usr/bin/ld"]] if platform.system() == "Darwin" else [[]],
    "gcc": [[]],
}

_LINK_PROBE = "int main(void) { return 0; }\n"

_OMP_PROBE = """
#include <omp.h>
int main() {
    int total = 0;
#pragma omp parallel for reduction(+ : total)
    for (int i = 0; i < 8; ++i)
        total += omp_get_max_threads();
    return total > 0 ? 0 : 1;
}
"""


def parse_device(device: str) -> tuple[str, int]:
    """Parse device string.

    cuda:N means cuda on gpuid N.
    cpu:N means cpu with N threads.
    """
    kind, _, spec = device.partition(":")
    kind = kind.strip().lower()

    if kind not in ("cpu", "cuda"):
        raise ValueError(f"Unsupported device: {device!r}. Expected 'cpu[:n_threads]' or 'cuda[:gpu_id]'.")

    if spec == "":
        n = 1 if kind == "cpu" else 0
    else:
        try:
            n = int(spec)
        except ValueError:
            raise ValueError(f"Unsupported device: {device!r}. Expected an integer after ':', got {spec!r}.") from None

    n = max(1, n) if kind == "cpu" else max(0, n)
    return kind, n


def check_cuda_available() -> None:
    if importlib.util.find_spec("cupy") is None:
        raise ImportError("`device='cuda'` requires `cupy` to be installed. But `cupy` is not available in the current environment.")


def resolve_cuda_props(gpu_id: int) -> dict[str, int]:
    """Query the device, letting cupy raise if it is missing, busy or the id is invalid."""
    import cupy as cp

    props = cp.cuda.runtime.getDeviceProperties(gpu_id)
    return {
        "max_threads_per_block": props["maxThreadsPerBlock"],
        "multiprocessor_count": props["multiProcessorCount"],
        "warp_size": props["warpSize"],
        "max_grid_size": props["maxGridSize"][0],
    }


def check_compiler_available() -> None:
    if not (shutil.which("clang") or shutil.which("gcc")):
        raise OSError("`device='cpu'` requires `gcc` or `clang` to be available in the PATH. But no suitable compiler was found.")


def select_compiler() -> str:
    if shutil.which("clang"):
        return "clang"
    if shutil.which("gcc"):
        return "gcc"
    raise RuntimeError("No suitable C compiler found (clang or gcc)")


def run_compiler(cmd: list[str]) -> None:
    subprocess.run(cmd, capture_output=True, check=True)


def resolve_link_flags(compiler: str) -> list[str]:
    """Return the first candidate flag set that links the probe below, or [] if none do."""
    candidates = LINK_FLAG_CANDIDATES.get(compiler, [[]])
    failures = []

    with tempfile.NamedTemporaryFile(suffix=".c", mode="w") as f:
        f.write(_LINK_PROBE)
        f.flush()
        out_file = f.name.replace(".c", ".out")

        for flags in candidates:
            try:
                run_compiler([compiler] + flags + [f.name, "-o", out_file])
            except subprocess.CalledProcessError as e:
                failures.append(f"  {' '.join(flags) or '(no extra flags)'}\n{e.stderr.decode().strip()}")
                continue

            os.unlink(out_file)
            return flags

    warnings.warn(
        f"Link check failed for compiler '{compiler}'. Tried:\n" + "\n".join(failures) + "\nProceeding without a linker override."
    )
    return []


def resolve_openmp_flags(compiler: str, link_flags: list[str] | None = None) -> list[str]:
    """Return the first candidate flag set that compiles the OpenMP probe, or [] if none do.

    `link_flags` is whatever `resolve_link_flags` already found for this compiler, so the probe
    doesn't fail for the same linker reason `resolve_link_flags` exists to work around.
    """
    candidates = OMP_FLAG_CANDIDATES.get(compiler, [])
    link_flags = link_flags or []
    failures = []

    with tempfile.NamedTemporaryFile(suffix=".c", mode="w") as f:
        f.write(_OMP_PROBE)
        f.flush()
        out_file = f.name.replace(".c", ".out")

        for flags in candidates:
            try:
                run_compiler([compiler, "-Werror=unknown-pragmas"] + link_flags + flags + [f.name, "-o", out_file])
            except subprocess.CalledProcessError as e:
                failures.append(f"  {' '.join(flags)}\n{e.stderr.decode().strip()}")
                continue

            os.unlink(out_file)
            return flags

    warnings.warn(
        f"OpenMP support check failed for compiler '{compiler}'. Tried:\n" + "\n".join(failures) + "\nProceeding without OpenMP support."
    )
    return []
