"""Progress bar settings for YTM.

YTM shows a tqdm progress bar during ``fit`` and inference (``score``, ``predict``, ``transform``, ...). The tqdm
options used for every bar live in ``KWARGS``. Change them with ``set_config``, or turn the bars off with
``disable`` or the ``TQDM_DISABLE=1`` environment variable. ``bar`` wraps any iterable in a progress bar with
these settings.
"""

import os

from tqdm import tqdm

__all__ = ["KWARGS", "bar", "disable", "set_config"]

KWARGS: dict = {
    "leave": False,
    "dynamic_ncols": True,
    "bar_format": "{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
    "disable": os.environ.get("TQDM_DISABLE", "0") not in ("0", "", "false", "False"),
}
"""tqdm options used for every YTM progress bar."""


def set_config(**kwargs) -> None:
    """Change the progress bar's tqdm options, e.g. `set_config(bar_format="...")`."""
    KWARGS.update(kwargs)


def disable() -> None:
    """Turn the progress bar off entirely."""
    KWARGS["disable"] = True


def bar(iterable, **kwargs):
    """Make a TQDM progress bar."""
    args = {**KWARGS, **kwargs}
    return tqdm(iterable, **args)
