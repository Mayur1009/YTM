import dataclasses
import datetime
import importlib.metadata
import json
import pathlib
import platform
import subprocess
import sys

from ytm.utils.profile import Profile

RESULTS = pathlib.Path(__file__).parent / "results"
DISTS = {"pysparse": "PySparseCoalescedTsetlinMachineCUDA"}


def _git(path: str, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=path, check=False, capture_output=True, text=True).stdout.strip()


def _lib_sha(lib: str) -> str:
    info = json.loads(importlib.metadata.distribution(lib).read_text("direct_url.json") or "{}")
    if commit := info.get("vcs_info", {}).get("commit_id"):
        return commit[:7]
    if info.get("url", "").startswith("file://"):
        path = info["url"].removeprefix("file://")
        sha = _git(path, "rev-parse", "--short", "HEAD")
        return f"{sha}-dirty" if sha and _git(path, "status", "--porcelain") else sha
    return ""


def _cpu() -> str:
    if sys.platform == "darwin":
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], check=False, capture_output=True, text=True).stdout.strip()
    for line in pathlib.Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor()


class RunLog:
    def __init__(self, libname: str, expname: str, params: dict, out: str | None = None):
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.path = pathlib.Path(out) if out else RESULTS / expname / f"{libname}_{expname}_{stamp}.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        dist = DISTS.get(libname, libname)
        self.run: dict = {
            "libname": libname,
            "expname": expname,
            "params": params,
            "meta": {
                "lib_version": importlib.metadata.version(dist),
                "lib_sha": _lib_sha(dist),
                "machine": f"{platform.platform()} | {_cpu()}",
                "python": platform.python_version(),
                "date": stamp,
            },
            "epochs": [],
        }

    def write(self, epoch: int, fit: Profile, predict: Profile, metrics: dict) -> None:
        self.run["epochs"].append(
            {"epoch": epoch, "fit": dataclasses.asdict(fit), "predict": dataclasses.asdict(predict), "metrics": metrics}
        )
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.run, indent=2, default=lambda o: o.item()))
        tmp.replace(self.path)
