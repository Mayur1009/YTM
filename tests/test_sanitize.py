import json
import os
import pathlib
import platform
import shutil
import subprocess
import sys

import pytest

from .conftest import DEVICES

pytestmark = pytest.mark.sanitize

ROOT = pathlib.Path(__file__).parents[1]

CONFIGS = {
    "discrete_flat_binary": dict(backend="discrete", kw=dict(n_clauses=6, s=3.0, dim=[4, 1, 1], n_classes=3)),
    "discrete_flat_therm_uncoalesced": dict(
        backend="discrete", kw=dict(n_clauses=4, s=3.0, dim=[3, 1, 1], n_classes=2, feat_maxs=3, coalesced=False)
    ),
    "discrete_conv": dict(backend="discrete", kw=dict(n_clauses=4, s=3.0, dim=[1, 6, 1], n_classes=2, patch_dim=[1, 3])),
    "discrete_patch_not_image": dict(
        backend="discrete", kw=dict(n_clauses=3, s=3.0, dim=[10, 10, 1], n_classes=2, patch_dim=[8, 8], stride=[5, 5])
    ),
    "guided_delta_l": dict(backend="guided", kw=dict(n_clauses=6, s=3.0, dim=[4, 1, 1], n_classes=3, fb_signal="delta_l")),
    "guided_grad_uncoalesced": dict(
        backend="guided", kw=dict(n_clauses=4, s=3.0, dim=[4, 1, 1], n_classes=2, fb_signal="grad", coalesced=False)
    ),
    "guided_conv_therm": dict(
        backend="guided", kw=dict(n_clauses=4, s=3.0, dim=[1, 6, 1], n_classes=2, patch_dim=[1, 3], feat_maxs=3, fb_signal="grad")
    ),
    "discrete_one_class": dict(backend="discrete", kind="binary", kw=dict(n_clauses=4, s=3.0, dim=[4, 1, 1])),
    "discrete_one_clause": dict(backend="discrete", kw=dict(n_clauses=1, s=3.0, dim=[2, 1, 1], n_classes=2)),
}


def _run(cmd, env_extra=None):
    env = {**os.environ, "TQDM_DISABLE": "1", **(env_extra or {})}
    return subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)


def _libs():
    from ytm._core.device_config import DeviceConfig

    cc = DeviceConfig(device="cpu:1")._compiler
    name = os.path.basename(cc)
    if "gcc" in name:
        libs = [f"lib{n}.so" for n in ("asan", "ubsan")]
    elif "clang" in name:
        # The shared clang ASan runtime bundles the UBSan one, so a single preload covers both.
        libs = [f"libclang_rt.asan-{platform.machine()}.so"]
    else:
        pytest.skip(f"ASan preload is only wired for gcc and clang, compiler is {cc}")
    paths = [subprocess.check_output([cc, f"-print-file-name={n}"], text=True).strip() for n in libs]
    if not all(os.path.isabs(p) and os.path.exists(p) for p in paths):
        pytest.skip(f"{cc} did not report {libs}")
    return ":".join(paths)


def _check(proc):
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert "OK" in proc.stdout
    assert "AddressSanitizer" not in proc.stderr and "runtime error" not in proc.stderr, proc.stderr[-3000:]


@pytest.mark.parametrize("name", list(CONFIGS))
def test_cpu_asan_ubsan_clean(name):
    """Out-of-bounds writes and bad indexing leave plausible-looking numbers; only the sanitizer sees them."""
    env = {"LD_PRELOAD": _libs(), "ASAN_OPTIONS": "detect_leaks=0", "UBSAN_OPTIONS": "print_stacktrace=1"}
    _check(_run([sys.executable, "-m", "tests.sanitize_worker", json.dumps(CONFIGS[name]), "cpu:2"], env))


@pytest.mark.parametrize("name", list(CONFIGS))
def test_cuda_memcheck_clean(name, request):
    """Same workload under compute-sanitizer memcheck (opt-in, slow)."""
    if not request.config.getoption("--cuda-sanitizer"):
        pytest.skip("pass --cuda-sanitizer to run")
    if "cuda" not in DEVICES:
        pytest.skip("no usable cuda device")
    tool = shutil.which("compute-sanitizer") or str(ROOT / ".pixi/envs/cuda/bin/compute-sanitizer")
    if not os.path.exists(tool):
        pytest.skip("compute-sanitizer not found")
    _check(_run([tool, "--error-exitcode", "1", sys.executable, "-m", "tests.sanitize_worker", json.dumps(CONFIGS[name]), "cuda"]))
