import os
import pathlib
from dataclasses import fields
from enum import IntEnum

import numpy as np
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


def prepare_X(cfg, X: np.ndarray) -> np.ndarray:
    """Check X against the feature bounds and shift it to `[0, therm_bits]`, shaped `(N, *cfg._dim)`."""
    # number of features match
    assert np.prod(X.shape[1:]) == np.prod(cfg._dim), f"Expected input features to match dim {cfg._dim}, but got {X.shape[1:]}"

    # Check if values are in provided bounds
    if cfg._patch_is_image:
        lo, hi, axes = cfg._feat_mins.reshape(cfg._dim), cfg._feat_maxs.reshape(cfg._dim), 0
    else:
        lo, hi, axes = cfg._feat_mins[: cfg._dim[2]], cfg._feat_maxs[: cfg._dim[2]], (0, 1, 2)
    Xv = np.asarray(X).reshape(X.shape[0], *cfg._dim)
    assert np.all(Xv.min(axis=axes) >= lo), f"X has values below feat_mins, min is {int(Xv.min())}"
    assert np.all(Xv.max(axis=axes) <= hi), f"X has values above feat_maxs, max is {int(Xv.max())}"

    # Shift X, so that the model always sees X in [0, therm_bits]
    return np.asarray((Xv - lo) if np.any(lo) else Xv, dtype=cfg._fbound_dtype, order="C")


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
