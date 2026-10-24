import os
import pathlib
from dataclasses import fields
from enum import IntEnum

from tqdm import tqdm

from .device_config import DeviceConfig


def enum_to_header(prefix: str, enum_cls: type[IntEnum]) -> str:
    return "".join(f"#define {prefix}_{m.name} {m.value}\n" for m in enum_cls)


def read_file(path: pathlib.Path | str) -> str:
    with open(path) as f:
        return f.read()


class Feedback(IntEnum):
    """Feedback codes shared by the C side and the python side. The header is generated from this."""

    NONE = 0
    T1A = 1
    T1B = 2
    T2 = 3



def split_device_kwargs(opt: dict) -> tuple[dict, dict]:
    """Split a kwargs dict into the model config fields and the DeviceConfig fields."""
    names = {f.name for f in fields(DeviceConfig)}
    dev_kw = {k: opt.pop(k) for k in list(opt) if k in names}
    return opt, dev_kw


TQDM_KWARGS: dict = {
    "leave": False,
    "dynamic_ncols": True,
    "bar_format": "{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
    "disable": os.environ.get("TQDM_DISABLE", "0") not in ("0", "", "false", "False"),
}


def tqdm_config(**kwargs) -> None:
    """Change the progress bar's tqdm options, e.g. `tqdm_config(bar_format="...")`."""
    TQDM_KWARGS.update(kwargs)


def tqdm_disable() -> None:
    """Turn the progress bar off entirely."""
    TQDM_KWARGS["disable"] = True


def tqdm_bar(iterable, **kwargs):
    args = {**TQDM_KWARGS, **kwargs}
    return tqdm(iterable, **args)
