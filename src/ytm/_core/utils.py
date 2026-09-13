from collections.abc import Sequence
from dataclasses import dataclass, fields
from enum import IntEnum
from typing import Any

import numpy as np
from tqdm import tqdm

from .device_config import DeviceConfig

try:
    from rich import box
    from rich.console import Console, Group
    from rich.panel import Panel
    from rich.table import Table

    _console = Console()
    _RICH = True
except ImportError:
    _RICH = False


def enum_to_header(prefix: str, enum_cls: type[IntEnum]) -> str:
    return "".join(f"#define {prefix}_{m.name} {m.value}\n" for m in enum_cls)


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


def tqdm_bar(iterable, **kwargs):
    args = {
        "leave": False,
        "dynamic_ncols": True,
        "bar_format": "{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
    }
    args.update(kwargs)
    return tqdm(iterable, **args)


@dataclass(kw_only=True)
class FitBuffers:
    X: Any  # (n_samples, *dim) the epoch's inputs, cast for the kernels
    Y: Any  # (n_samples, n_classes) targets
    clause_drop_mask: Any  # (total_clauses,) one mask for the whole epoch, per Drop Clause
    selected_pids: Any  # (total_clauses,) patch each clause matched on, -1 for none
    votes: Any  # (n_classes,) weighted vote sum for the current sample


@dataclass
class PackedClauses:
    clause_feat_bounds: Any  # (total_clauses, n_raw_patch_feats, 2) closed [lower, upper]
    clause_position_bounds: Any  # (total_clauses, 4) closed [min_y, max_y, min_x, max_x]
    bounded_feat_ids: Any  # (total_clauses, n_raw_patch_feats) features that constrain anything
    n_bounded_feats: Any  # (total_clauses,) how many of the above are in use
    clause_density: Any  # (total_clauses,) included literals, -1 marks a contradiction
    is_clause_synced: Any  # (total_clauses,) 0 when the clause needs repacking


@dataclass
class ClauseInfo:
    """Packed clause representation returned by :meth:`BaseTM.get_clauses`.

    Attributes
    ----------
    feature_bounds : ndarray of shape (n_clause_banks, n_clauses, n_raw_patch_feats * 2)
        Closed ``[lower, upper]`` feature inclusion bounds per clause.
    position_bounds : ndarray of shape (n_clause_banks, n_clauses, 4) or None
        Closed ``[min_y, max_y, min_x, max_x]`` position bounds per clause.
        ``None`` when position literals are disabled and input has a single patch.
    clause_density : ndarray of shape (n_clause_banks, n_clauses), dtype int
        Number of included literals per clause. ``-1`` marks an invalid
        clause (contains a contradiction).
    """

    feature_bounds: np.ndarray
    position_bounds: np.ndarray | None
    clause_density: np.ndarray

    def _title(self, bank: int, clause: int) -> str:
        density = int(self.clause_density[bank, clause])
        if density < 0:
            state = "CONTRADICTION"
        elif density == 0:
            state = "density=0  (matches everything)"
        else:
            state = f"density={density}"
        return f"clause ({bank}, {clause})  {state}"

    def _position(self, bank: int, clause: int, le: str) -> list[tuple[str, bool]]:
        """One entry per axis, paired with whether that axis is inverted."""
        if self.position_bounds is None:
            return []
        y0, y1, x0, x1 = (int(v) for v in self.position_bounds[bank, clause])
        return [(f"{y0} {le} y {le} {y1}", y0 > y1), (f"{x0} {le} x {le} {x1}", x0 > x1)]

    def _features(self, bank: int, clause: int, feat_names: Sequence[str] | None, le: str) -> list[tuple[str, bool]]:
        """One entry per feature, paired with whether its interval is inverted."""
        bounds = self.feature_bounds[bank, clause].reshape(-1, 2)
        names = feat_names if feat_names is not None else [f"X{i}" for i in range(len(bounds))]
        out = []
        for name, (lo, hi) in zip(names, bounds, strict=True):
            lo, hi = int(lo), int(hi)
            out.append((f"{name} = {lo}" if lo == hi else f"{lo} {le} {name} {le} {hi}", lo > hi))
        return out

    def _plain(self, bank: int, clause: int, feat_names: Sequence[str] | None) -> str:
        """Ascii rendering, used when rich is not installed. `X` marks an inverted interval."""
        cells = [(t + ("  X" if bad else ""), bad) for t, bad in self._features(bank, clause, feat_names, "<=")]
        width = max(len(t) for t, _ in cells)
        sep = "+" + "+".join(["-" * (width + 2)] * len(cells)) + "+"
        row = "|" + "|".join(f" {t:<{width}} " for t, _ in cells) + "|"

        lines = [self._title(bank, clause)]
        pos = self._position(bank, clause, "<=")
        if pos:
            lines.append("  position  " + "    ".join(t + ("  X" if bad else "") for t, bad in pos))
        lines += ["  " + sep, "  " + row, "  " + sep]
        return "\n".join(lines)

    def _panel(self, bank: int, clause: int, feat_names: Sequence[str] | None):
        table = Table(box=box.SQUARE, show_header=False, border_style="bright_black", pad_edge=False)
        cells = self._features(bank, clause, feat_names, "≤")
        for _ in cells:
            table.add_column(justify="center", no_wrap=True)
        table.add_row(*(f"[red]{t}[/]" if bad else t for t, bad in cells))

        body = []
        pos = self._position(bank, clause, "≤")
        if pos:
            body.append("position   " + "    ".join(f"[red]{t}[/]" if bad else t for t, bad in pos))
        body.append(table)

        density = int(self.clause_density[bank, clause])
        colour = "red" if density < 0 else ("yellow" if density == 0 else "green")
        return Panel(Group(*body), title=f"[{colour}]{self._title(bank, clause)}[/]", title_align="left")

    def to_string(self, bank: int, clause: int, feat_names: Sequence[str] | None = None) -> str:
        """The clause at `(bank, clause)` as plain text. Features are named `X0, X1, ...` unless given."""
        return self._plain(bank, clause, feat_names)

    def pprint(self, bank: int, clause: int, feat_names: Sequence[str] | None = None) -> None:
        """Print the clause at `(bank, clause)`."""
        if not _RICH:
            print(self._plain(bank, clause, feat_names))
            return
        _console.print(self._panel(bank, clause, feat_names))
