import argparse
import json
import pathlib
import statistics
from collections import defaultdict

RESULTS = pathlib.Path(__file__).parent / "results"
TABLES = [
    ("Non-coalesced", "CPU", 0, "cpu", "tmu", "TMU"),
    ("Non-coalesced", "CUDA", 0, "cuda", "tmu", "TMU"),
    ("Coalesced", "CPU", 1, "cpu", "tmu", "TMU"),
    ("Coalesced", "CUDA", 1, "cuda", "pysparse", "PySparse"),
]


def device_of(params: dict) -> str:
    if str(params.get("device", "")).startswith("cuda") or params.get("platform") == "CUDA":
        return "cuda"
    return "cpu"


def summarize(run: dict) -> dict:
    epochs = run["epochs"]
    peak = lambda key: max(max(e["fit"][key] or 0, e["predict"][key] or 0) for e in epochs) / 1e6
    return {
        "fit": statistics.mean(e["fit"]["elapsed"] for e in epochs),
        "predict": statistics.mean(e["predict"]["elapsed"] for e in epochs),
        "ram": peak("ram_peak"),
        "vram": peak("vram_peak"),
        "acc": epochs[-1]["metrics"]["acc"] * 100,
    }


def load(dataset: str | None) -> dict:
    groups = defaultdict(list)
    for path in sorted(RESULTS.glob("*/*.json")):
        run = json.loads(path.read_text())
        if dataset and not run["expname"].startswith(dataset):
            continue
        key = (run["libname"], run["expname"], run["params"]["coalesced"], device_of(run["params"]))
        groups[key].append(summarize(run))
    return {key: {m: statistics.mean(r[m] for r in runs) for m in runs[0]} for key, runs in groups.items()}


def cell(a: dict | None, b: dict | None, metric: str, fmt: str) -> str:
    fmt_one = lambda r: format(r[metric], fmt) if r else ""
    return f"{fmt_one(a)} / {fmt_one(b)}"


def print_tables(results: dict) -> None:
    for arch, dev_name, coalesced, device, other, other_name in TABLES:
        expnames = sorted({k[1] for k in results if k[2] == coalesced and k[3] == device})
        if not expnames:
            continue
        mem, mem_name = ("vram", "Peak VRAM MB") if device == "cuda" else ("ram", "Peak RAM MB")
        names = f"(YTM / {other_name})"
        print(f"## {arch}\n\n### {dev_name}\n")
        print(f"| Experiment | Fit s/epoch {names} | Predict s {names} | {mem_name} {names} | Acc % {names} |")
        print("|---|---|---|---|---|")
        for exp in expnames:
            a, b = results.get(("ytm", exp, coalesced, device)), results.get((other, exp, coalesced, device))
            cells = [cell(a, b, "fit", ".2f"), cell(a, b, "predict", ".2f"), cell(a, b, mem, ".0f"), cell(a, b, "acc", ".2f")]
            print(f"| {exp} | " + " | ".join(cells) + " |")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?")
    args = parser.parse_args()
    print_tables(load(args.dataset))
