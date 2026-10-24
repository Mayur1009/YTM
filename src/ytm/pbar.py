import os

from tqdm import tqdm

KWARGS: dict = {
    "leave": False,
    "dynamic_ncols": True,
    "bar_format": "{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
    "disable": os.environ.get("TQDM_DISABLE", "0") not in ("0", "", "false", "False"),
}


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
