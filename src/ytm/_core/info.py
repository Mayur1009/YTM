from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, NamedTuple

import numpy as np

if TYPE_CHECKING:
    from .config import BaseTMConfig


class Symbols(NamedTuple):
    and_: str
    not_: str
    le: str
    ge: str
    contra: str


UNICODE_SYMBOLS = Symbols("∧", "¬", "≤", "≥", "⊥")
ASCII_SYMBOLS = Symbols("&", "~", "<=", ">=", "!")

Justify = Literal["left", "right", "center"]


@dataclass
class ClauseInfo:
    """Clauses returned by :meth:`BaseTM.get_clauses`.

    Attributes
    ----------
    feature_bounds : ndarray of shape (n_clause_banks, n_clauses, n_raw_patch_feats * 2)
        Closed ``[lower, upper]`` feature inclusion bounds per clause.
    position_bounds : ndarray of shape (n_clause_banks, n_clauses, 4) or None
        Closed ``[min_row, max_row, min_col, max_col]`` patch position bounds per clause.
        ``None`` when the input has a single patch.
    has_contra : ndarray of shape (n_clause_banks, n_clauses), dtype int8
        ``1`` when the clause is unsatisfiable (contains a contradiction).
    clause_len : ndarray of shape (n_clause_banks, n_clauses), dtype int
        Number of included literals per clause. ``0`` means the clause is
        empty and thus a tautology.
    clause_weights : ndarray of shape (n_classes, n_clauses)
        Weight of each clause per class. Non-coalesced: row ``j`` holds class ``j``'s own clauses.
        Coalesced: every row covers the same shared clauses.
    config : BaseTMConfig
        Config of the model the clauses come from.
    """

    feature_bounds: np.ndarray
    position_bounds: np.ndarray | None
    has_contra: np.ndarray
    clause_len: np.ndarray
    clause_weights: np.ndarray
    config: "BaseTMConfig" = field(repr=False)

    def clause_to_str(
        self,
        class_id: int,
        clause_id: int,
        feat_names: list[str] | None = None,
        include_weight: bool = True,
        ascii: bool = False,
    ) -> str:
        """One clause as text.

        Parameters
        ----------
        class_id : int
            Class the clause belongs to. Ignored for coalesced model.
        clause_id : int
            Clause ID.
        feat_names : list of str, optional
            One name per raw patch feature. Defaults to ``X0, X1, ...``.
        include_weight : bool, default=True
            Prefix the returned string with ``W[...]:``, the clause weight/s.
        ascii : bool, default=False
            Use ascii ``& ~ <= >=`` instead of unicode ``∧ ¬ ≤ ≥``.

        Returns
        -------
        str
            E.g. ``"W[+2]: ¬x0 ∧ x1"``, ``"W[-1, +2, +3]: (x2 ≥ 6)"``, ``"W[+1]: (empty)"``.

        Examples
        --------
        Binary features, not coalesced, one weight per clause:

        >>> info.clause_to_str(1, 1)
        'W[-20]: X0 ∧ X1'
        >>> info.clause_to_str(1, 0, feat_names=["a", "b", "c"], ascii=True)
        'W[-6]: ~a & ~b'
        >>> info.clause_to_str(1, 0, include_weight=False)
        '¬X0 ∧ ¬X1'
        >>> info.clause_to_str(1, 3)
        'W[+10]: (empty)'

        Thermometer features in ``[0, 7]``, coalesced, so one weight per class:

        >>> info.clause_to_str(0, 0)
        'W[+11, -11]: (X0 ≤ 4)'
        >>> info.clause_to_str(0, 2)
        'W[-10, +10]: (5 ≤ X0 ≤ 6)'
        >>> info.clause_to_str(0, 3)
        'W[-10, +10]: (X0 = 7)'
        """
        cfg = self.config
        sym = ASCII_SYMBOLS if ascii else UNICODE_SYMBOLS
        names = feat_names if feat_names is not None else [f"X{i}" for i in range(cfg._n_raw_patch_feats)]
        cells = [text for text, _ in self._clause_cells(class_id, clause_id, names, sym) if text]
        rule = f" {sym.and_} ".join(cells) if cells else "(empty)"
        if not include_weight:
            return rule

        # Coalesced clauses vote for every class, so all weights are shown.
        weights = self.clause_weights
        w = weights[:, clause_id] if cfg.coalesced else weights[class_id : class_id + 1, clause_id]
        return f"W[{', '.join(_weight_str(v) for v in w)}]: {rule}"

    def print(
        self,
        feat_names: list[str] | None = None,
        sort: str = "w0",
        top_k: int | None = None,
        ascii: bool = False,
        force_no_rich: bool = False,
    ) -> None:
        """Print clause banks as tables.

        Used ``rich`` if available, or fallback to simple text output.

        Parameters
        ----------
        feat_names : list of str, optional
            Feature names. Defaults to ``X0, X1, ...``.
        sort : str, default="w0"
            Comma-separated sort keys, applied in order for positive and negative polarities.
            Possible keys :
              - ``wK``: class K |weight|, largest first.
              - ``len``: shortest first.
              - ``!``: contradictions last.
            A ``-`` prefix reverses a key. For non-coalesced ``K`` in ``wK`` will be ignored.
        top_k : int, optional
            Show at most this many rows per section.
        ascii : bool, default=False
            Use ascii ``& ~ <= >= !`` instead of unicode ``∧ ¬ ≤ ≥ ⊥``.
        force_no_rich : bool, default=False
            Plain text table even when ``rich`` is installed.

        Examples
        --------
        Default, largest weight first:
        >>> info.print()

        Shortest clauses first, equal lengths by weight:
        >>> info.print(sort="len,w0")

        Contradictions last, then longest first:
        >>> info.print(sort="!,-len")

        Coalesced: split into positive and negative sections by class 2's weight, shortest first:
        >>> info.print(sort="len,w2")

        Only the 10 largest weights of each section:
        >>> info.print(top_k=10)
        """
        assert top_k is None or top_k >= 1, f"top_k must be None or >= 1, got {top_k}"
        cfg = self.config
        sym = ASCII_SYMBOLS if ascii else UNICODE_SYMBOLS
        names = feat_names if feat_names is not None else [f"X{i}" for i in range(cfg._n_raw_patch_feats)]
        weights = self.clause_weights
        has_pos = self.position_bounds is not None
        # sec_class: class whose weight sign splits the sections, keys: [(key, reversed), ...]
        sec_class, keys = self._parse_sort(sort)

        use_rich = False
        if not force_no_rich:
            try:
                import rich  # noqa: F401

                use_rich = True
            except ImportError:
                pass
        render = _render_rich if use_rich else _render_plain

        def join_slots(cell_lists: list[list[tuple[str, bool]]]) -> list[list[tuple[str, bool]]]:
            # Pad every slot to its widest cell so the `and` symbols line up, drop slots empty in every row.
            # Kept as (text, is_contra) segments so the renderer can mark contradictions.
            widths = [max(len(cl[i][0]) for cl in cell_lists) for i in range(len(cell_lists[0]))]
            out = []
            for cl in cell_lists:
                segs = []
                for (text, bad), w in zip(cl, widths, strict=True):
                    if w == 0:
                        continue
                    if segs:
                        segs.append((f" {sym.and_} ", False))
                    segs.append((text.center(w), bad))
                out.append(segs)
            return out

        def seg_len(segs: list[tuple[str, bool]]) -> int:
            return sum(len(text) for text, _ in segs)

        # One table per clause bank, a single table when coalesced.
        for bank in range(cfg._n_clause_banks):
            # Smaller sorts first, so weights are negated to put the largest first.
            def key_value(name: str, c: int, bank: int = bank) -> float:
                if name == "len":
                    return int(self.clause_len[bank, c])
                if name == "!":
                    return int(self.has_contra[bank, c])
                if name == "w":
                    return -abs(float(weights[bank, c]))
                return -abs(float(weights[int(name[1:]), c]))

            # Keys compared in order, later keys break ties. A "-" key flips its sign.
            def sort_key(c: int) -> tuple:
                return tuple(-key_value(n, c) if rev else key_value(n, c) for n, rev in keys)

            # Positive section first, then negative.
            sec_w = weights[sec_class] if cfg.coalesced else weights[bank]
            ids = range(cfg._n_clauses)
            sections = [sorted((c for c in ids if sec_w[c] >= 0), key=sort_key), sorted((c for c in ids if sec_w[c] < 0), key=sort_key)]

            # Cut each section to top_k, the footer tells how many were hidden.
            footers = [f"showing {top_k} of {len(s)}" if top_k is not None and len(s) > top_k else None for s in sections]
            sections = [s[:top_k] if top_k is not None else s for s in sections]
            shown = [c for s in sections for c in s]

            # First two cells are (row, col) when there are patches, they go in their own column.
            cells = [self._clause_cells(bank, c, names, sym) for c in shown]
            rules = join_slots([cl[2:] if has_pos else cl for cl in cells])
            positions = join_slots([cl[:2] for cl in cells]) if has_pos else None

            if cfg.coalesced:
                # One padded entry per class so the weights line up across rows.
                w_strs = [[_weight_str(v) for v in weights[:, c]] for c in shown]
                w_width = [max(len(ws[k]) for ws in w_strs) for k in range(weights.shape[0])]
                w_col = ["[" + " ".join(v.rjust(w_width[k]) for k, v in enumerate(ws)) + "]" for ws in w_strs]
            else:
                w_col = [_weight_str(weights[bank, c]) for c in shown]

            # Columns: clause id, contradiction mark, length, weight(s), [position], rule.
            rows = []
            for i, c in enumerate(shown):
                row = [str(c), [(sym.contra, True)] if self.has_contra[bank, c] else "_", str(int(self.clause_len[bank, c])), w_col[i]]
                if positions is not None:
                    row.append(positions[i])
                rows.append(row + [rules[i]])

            header = ["ci", sym.contra, "|C|", "W[:, ci]" if cfg.coalesced else f"W[{bank}, ci]"]
            justify: list[Justify] = ["right", "center", "right", "right"]
            if positions is not None:
                header.append("Position (row, col)".center(max(seg_len(p) for p in positions)))
                justify.append("left")
            header.append("C[ci]".center(max(seg_len(r) for r in rules)))
            justify.append("left")

            title = "Clauses (coalesced)" if cfg.coalesced else f"Class: {bank}"
            n_pos = len(sections[0])
            render(title, header, justify, [rows[:n_pos], rows[n_pos:]], footers)

    def _clause_cells(self, class_id: int, clause_id: int, feat_names: list[str], sym: Symbols) -> list[tuple[str, bool]]:
        """Unpadded (text, is_contra) cells of one clause: [row, col] when there are patches, then one per feature, "" when unconstrained."""
        cfg = self.config
        bank = 0 if cfg.coalesced else class_id
        cells = []

        if self.position_bounds is not None:
            r0, r1, c0, c1 = (int(v) for v in self.position_bounds[bank, clause_id])
            cells.append((_bound_str("row", r0, r1, 0, cfg._n_patches_y - 1, sym, False), r0 > r1))
            cells.append((_bound_str("col", c0, c1, 0, cfg._n_patches_x - 1, sym, False), c0 > c1))

        bounds = self.feature_bounds[bank, clause_id].reshape(-1, 2)
        for name, (lo, hi), fmin, fmax in zip(feat_names, bounds, cfg._feat_mins, cfg._feat_maxs, strict=True):
            text = _bound_str(name, int(lo), int(hi), int(fmin), int(fmax), sym, fmax - fmin == 1)
            cells.append((text, bool(lo > hi)))
        return cells

    def _parse_sort(self, sort: str) -> tuple[int, list[tuple[str, bool]]]:
        """Parse `sort` into (class whose weight sign splits the sections, [(key, reversed), ...])."""
        cfg = self.config
        n_classes = cfg.n_classes
        sec_class = None
        keys: dict[str, bool] = {}  # first occurrence wins
        for k in filter(None, (k.strip() for k in (sort or "").split(","))):
            rev, name = k.startswith("-"), k.removeprefix("-")
            if name not in ("len", "!"):
                if not (name.startswith("w") and name[1:].isdigit() and int(name[1:]) < n_classes):
                    raise ValueError(f"unknown sort key {k!r}, expected w0..w{n_classes - 1}, len or !, optionally prefixed with -")
                if sec_class is None:
                    sec_class = int(name[1:])
                if not cfg.coalesced:  # one weight per clause, every wK is the same key
                    name = "w"
            keys.setdefault(name, rev)
        return sec_class or 0, list(keys.items())


def _bound_str(name: str, lo: int, hi: int, fmin: int, fmax: int, sym: Symbols, binary: bool) -> str:
    if (lo, hi) == (fmin, fmax):
        return ""
    if binary:  # lo > hi means both x and not x are included
        return f"({name} {sym.and_} {sym.not_}{name})" if lo > hi else name if lo == fmax else f"{sym.not_}{name}"
    if lo > hi:  # contradiction, keep both bounds so the empty range is visible
        return f"({lo} {sym.le} {name} {sym.le} {hi})"
    if lo == hi:
        return f"({name} = {lo})"
    if lo == fmin:
        return f"({name} {sym.le} {hi})"
    if hi == fmax:
        return f"({name} {sym.ge} {lo})"
    return f"({lo} {sym.le} {name} {sym.le} {hi})"


def _weight_str(w: float) -> str:
    return f"{w:+.0f}" if float(w).is_integer() else f"{w:+.3f}"


def _render_rich(title: str, header: list[str], justify: list[Justify], sections: list[list[list]], footers: list[str | None]) -> None:
    """Cells are str, or (text, is_contra) segments whose contradictory parts are shown red."""
    from rich import box
    from rich.console import Console
    from rich.markup import escape
    from rich.table import Table
    from rich.text import Text

    def cell(v):
        if isinstance(v, str):
            return escape(v)  # rich would read [ci] as markup
        return Text.assemble(*((text, "red") if bad else text for text, bad in v))

    table = Table(title=title, box=box.ASCII_DOUBLE_HEAD, header_style="bold dim", border_style="bright_black")
    for h, j in zip(header, justify, strict=True):
        table.add_column(escape(h), justify=j, no_wrap=True)
    for rows, footer in zip(sections, footers, strict=True):
        if not rows:
            continue
        for row in rows:
            table.add_row(*(cell(v) for v in row))
        if footer:
            table.add_row(*[""] * (len(header) - 1), f"[dim]{footer}[/]")
        table.add_section()
    Console().print(table)


def _render_plain(title: str, header: list[str], justify: list[Justify], sections: list[list[list]], footers: list[str | None]) -> None:
    # (text, is_contra) segments are joined, contradictions are already visible in the text itself
    sections = [[[v if isinstance(v, str) else "".join(text for text, _ in v) for v in r] for r in rows] for rows in sections]
    widths = [max(len(h), *(len(r[i]) for rows in sections for r in rows)) for i, h in enumerate(header)]
    align = {"left": str.ljust, "right": str.rjust, "center": str.center}

    def line(ch: str) -> str:
        return "+" + "+".join(ch * (w + 2) for w in widths) + "+"

    def fmt(vals: list[str]) -> str:
        return "|" + "|".join(f" {align[j](v, w)} " for v, w, j in zip(vals, widths, justify, strict=True)) + "|"

    # = for the table top, bottom and under the header, ~ between sections.
    inner = len(line("-")) - 2
    out = ["+" + "=" * inner + "+", "|" + title.center(inner) + "|", line("-"), fmt(header), line("=")]
    first = True
    for rows, footer in zip(sections, footers, strict=True):
        if not rows:
            continue
        if not first:
            out.append(line("~"))
        first = False
        out += [fmt(r) for r in rows]
        if footer:
            out.append("|" + f" {footer}".ljust(inner) + "|")
    out.append(line("="))
    print("\n".join(out) + "\n")
