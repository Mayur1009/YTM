from time import time

import numpy as np

from .binarizer import Binarizer as Binarizer
from .profile import Profiler as Profiler

try:
    from rich import box
    from rich.console import Console
    from rich.table import Table

    _console = Console()
    _RICH = True
except ImportError:
    _RICH = False


class Timer:
    """Context manager for measuring elapsed wall-clock time.

    Examples
    --------
    >>> with Timer() as timer:
    ...     do_work()
    >>> print(timer.elapsed)
    """

    start_time: float
    end_time: float

    def __enter__(self):
        if hasattr(self, "end_time"):
            del self.end_time
        self.start_time = time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time()

    @property
    def elapsed(self) -> float:
        """Elapsed time in seconds. Must be accessed after the context exits.

        Returns
        -------
        float
            Wall-clock seconds between ``__enter__`` and ``__exit__``.

        Raises
        ------
        RuntimeError
            If accessed before the context has exited.
        """
        if not hasattr(self, "end_time"):
            raise RuntimeError("elapsed must be accessed after context is ended.")
        return self.end_time - self.start_time


def print_table(id: str, rows: dict[str, dict], force_no_rich: bool = False) -> None:
    """Print a formatted table to the console.

    Column headers are derived from all keys across all row dicts.
    Missing values are shown as ``-``. Uses ``rich`` if available, otherwise
    falls back to plain ASCII.

    Parameters
    ----------
    id : str
        Label for the top-left header cell (e.g. epoch identifier).
    rows : dict of str to dict
        Mapping of row name to a dict of column values.
    force_no_rich : bool, default=False
        If True, use plain ASCII output even if ``rich`` is installed.

    Examples
    --------
    >>> print_table("Epoch 1/10", {
    ...     "Train": {"Acc": "95.00%", "Fit Time": "1.23s"},
    ...     "Test":  {"Acc": "92.00%"},
    ... })
    """
    headers = list(dict.fromkeys(k for row in rows.values() for k in row))

    if _RICH and not force_no_rich:
        table = Table(box=box.ASCII_DOUBLE_HEAD, header_style="bold dim", border_style="bright_black")
        table.add_column(id, style="bold dim")
        for h in headers:
            table.add_column(h, justify="right")
        for name, row in rows.items():
            table.add_row(f"[bold dim]{name}[/]", *(str(row.get(h, "-")) for h in headers))
        _console.print(table)
    else:
        table = [[id, *headers]] + [[name, *(str(row.get(h, "-")) for h in headers)] for name, row in rows.items()]
        widths = [max(len(str(cell)) for cell in col) for col in zip(*table)]
        sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
        lines = ["|" + "|".join(f" {cell!s:>{w}} " for cell, w in zip(line, widths)) + "|" for line in table]
        print(sep)
        print(lines[0])
        print(sep)
        for line in lines[1:]:
            print(line)
        print(sep)


def norm_asymmetric(arr: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Zero-anchored per-sample normalization. Scales output to ``[-1, 1]``.

    For each sample independently, negative values are scaled by that sample's
    own ``|min|`` so the most negative value maps to ``-1``. Positive values
    are scaled by that sample's own ``max`` so the most positive value maps to
    ``1``. Zero stays zero.

    Required to properly scale interpretation maps of the TM, especially for images.

    Parameters
    ----------
    arr : ndarray of shape (N, ...)
        Input array. The first axis is the sample axis.
    eps : float, default=1e-9
        Added to each divisor to avoid division by zero.

    Returns
    -------
    ndarray of shape (N, ...)
        Normalized copy of ``arr``. The input is not changed.

    Examples
    --------
    >>> norm_asymmetric(np.array([[-2.0, 0.0, 1.0, 4.0]]))
    array([[-1.  ,  0.  ,  0.25,  1.  ]])
    """
    assert arr.ndim >= 2, "expected (N, *feat_shape)"
    axis = tuple(range(1, arr.ndim))
    out = arr.copy()
    neg_min = out.min(axis=axis, keepdims=True)
    pos_max = out.max(axis=axis, keepdims=True)
    out = np.where(out < 0, out / (-neg_min + eps), out)
    out = np.where(out > 0, out / (pos_max + eps), out)
    return out
