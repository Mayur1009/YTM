import abc
from dataclasses import dataclass, fields

import numpy as np

from .backends.base import BaseDevice
from .config import BaseTMConfig
from .device_config import DeviceConfig
from .utils import prepare_X


@dataclass
class ClauseInfo:
    """Packed clause representation returned by :meth:`BaseTM.get_clauses`.

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
        vacuous and fires on every input.
    clause_weights : ndarray of shape (n_classes, n_clauses)
        Weight of each clause per class. Non-coalesced: row ``j`` holds class ``j``'s own clauses.
        Coalesced: every row covers the same shared clauses.
    """

    feature_bounds: np.ndarray
    position_bounds: np.ndarray | None
    has_contra: np.ndarray
    clause_len: np.ndarray
    clause_weights: np.ndarray


class BaseTM(abc.ABC):
    config_cls: type[BaseTMConfig]
    cpu_device_cls: type[BaseDevice]

    def __init__(self, dev: BaseDevice):
        self.dev = dev
        self._rng = np.random.default_rng(self.config.seed)

    @property
    def config(self) -> BaseTMConfig:
        return self.dev.config

    @property
    def device_config(self) -> DeviceConfig:
        return self.dev.device_config

    @staticmethod
    @abc.abstractmethod
    def _cuda_device_cls() -> type[BaseDevice]: ...

    def _build_device(self, config: BaseTMConfig, device_config: DeviceConfig, state: dict | None = None) -> BaseDevice:
        cls = self.cpu_device_cls if device_config.kind == "cpu" else self._cuda_device_cls()
        return cls(config, device_config, state)

    def to(self, device: str, **device_kwargs) -> None:
        """Move the model to another device, in place."""
        self.dev = self._build_device(self.config, DeviceConfig(device=device, **device_kwargs), self.dev.get_state_dict())

    def __getstate__(self) -> dict:
        """Config is saved as its constructor arguments, minus `ta_init`: its only use is at construction."""
        cfg = self.config
        return {
            "config": {f.name: getattr(cfg, f.name) for f in fields(cfg) if f.init and f.name != "ta_init"},
            "params": self.dev.get_state_dict(),
            "rng": self._rng,
        }

    def __setstate__(self, state: dict) -> None:
        """Unpickling always lands on `cpu:1`; use `.to()` afterwards to move it."""
        cfg = self.config_cls(**state["config"])
        self.dev = self._build_device(cfg, DeviceConfig(), state["params"])
        self._rng = state["rng"]

    @abc.abstractmethod
    def _fit(self, X: np.ndarray, Y: np.ndarray, *args, **kwargs): ...

    def _prepare_X(self, X: np.ndarray) -> np.ndarray:
        return prepare_X(self.config, X)

    def score(self, X: np.ndarray, batch_size: int = -1, force_repack: bool = False) -> np.ndarray:
        return self.dev.calc_class_sums(self._prepare_X(X), batch_size, force_repack)

    @abc.abstractmethod
    def to_prob(self, class_sums: np.ndarray) -> np.ndarray:
        """Map class sums from `score`/`predict` to [0, 1]."""

    def transform(self, X: np.ndarray, batch_size: int = -1, force_repack: bool = False) -> np.ndarray:
        return self.dev.transform(self._prepare_X(X), batch_size, force_repack)

    def transform_patchwise(self, X: np.ndarray, batch_size: int = -1, force_repack: bool = False) -> np.ndarray:
        return self.dev.transform_patchwise(self._prepare_X(X), batch_size, force_repack)

    def wic(self, class_id: int, polarity: int, pw_th: float = 0.0, force_repack: bool = False) -> np.ndarray:
        if self.config._n_patches > 1 and not self.config.track_patch_weights:
            raise ValueError("track_patch_weights=True is required for wic() on a convolutional model.")
        return self.dev.wic(class_id, polarity, pw_th, force_repack)

    def wac(self, X: np.ndarray, target_classes: np.ndarray, polarity: int, batch_size: int = -1, force_repack: bool = False) -> np.ndarray:
        return self.dev.wac(self._prepare_X(X), target_classes, polarity, batch_size, force_repack)

    def set_threads(self, n: int) -> None:
        self.dev.set_threads(max(1, n))

    # == getters ==
    def get_weights(self) -> np.ndarray:
        return self.dev.get_weights()

    def get_ta_states(self) -> np.ndarray:
        return self.dev.get_ta_states()

    def get_patch_weights(self) -> np.ndarray:
        return self.dev.get_patch_weights()

    def get_literals(self) -> np.ndarray:
        return np.asarray(self.get_ta_states() >= self.config._include_state, dtype=np.uint8)

    def get_clauses(self, force_repack: bool = True) -> ClauseInfo:
        """Clauses as intervals. Cached; `force_repack=False` returns the cache when there is one, stale after more training."""
        if not force_repack and hasattr(self, "_clauses"):
            return self._clauses

        cfg = self.config
        self.dev.pack_clauses(force_repack=True, full=True)
        packed = self.dev.get_packed_clauses()
        shape = (cfg._n_clause_banks, cfg._n_clauses)

        position_bounds = None
        if cfg._n_patches > 1:
            position_bounds = packed.clause_position_bounds.reshape(*shape, 4)

        # Scatter the compacted bounds back to one entry per feature, unconstrained ones spanning
        # the full range, then undo the internal zero basing so the intervals are in user units.
        n_feat = cfg._n_raw_patch_feats
        feature_bounds = np.empty((cfg._total_clauses, n_feat, 2), dtype=np.int32)
        feature_bounds[:, :, 0] = 0
        feature_bounds[:, :, 1] = cfg._therm_bits

        fids = packed.clause_feat_ids.reshape(cfg._total_clauses, n_feat)
        compact = packed.clause_feat_bounds.reshape(cfg._total_clauses, n_feat, 2)
        used = np.arange(n_feat)[None, :] < packed.clause_n_feats[:, None]
        ci, si = np.nonzero(used)
        feature_bounds[ci, fids[ci, si]] = compact[ci, si]

        feature_bounds = feature_bounds.reshape(*shape, n_feat, 2) + cfg._feat_mins.reshape(-1, 1)

        self._clauses = ClauseInfo(
            feature_bounds=feature_bounds.reshape(*shape, cfg._n_raw_patch_feats * 2),
            position_bounds=position_bounds,
            has_contra=packed.has_contra.reshape(*shape),
            clause_len=packed.clause_len.reshape(*shape),
            clause_weights=self.get_weights(),
        )
        return self._clauses

    # == printing functions ==
    def clause_to_str(
        self,
        class_id: int,
        clause_id: int,
        feat_names: list[str] | None = None,
        include_weight: bool = True,
        ascii: bool = False,
    ) -> str:
        """One clause as text.

        Uses the cached clauses from :meth:`get_clauses`, so call ``get_clauses()``
        after to refresh them.

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
        """
        cfg = self.config
        sym = self._symbols(ascii)
        names = feat_names if feat_names is not None else [f"X{i}" for i in range(cfg._n_raw_patch_feats)]
        clauses = self.get_clauses(force_repack=False)
        cells = [c for c in self._clause_cells(clauses, class_id, clause_id, names, sym) if c]
        rule = f" {sym[0]} ".join(cells) if cells else "(empty)"
        if not include_weight:
            return rule

        # Coalesced clauses vote for every class, so all weights are shown.
        weights = clauses.clause_weights
        w = weights[:, clause_id] if cfg.coalesced else weights[class_id : class_id + 1, clause_id]
        return f"W[{', '.join(self._weight_str(v) for v in w)}]: {rule}"

    def print_clauses(
        self,
        feat_names: list[str] | None = None,
        sort: str = "w0",
        top_k: int | None = None,
        ascii: bool = False,
        force_no_rich: bool = False,
        force_repack: bool = True,
    ) -> None:
        """Print clause banks as tables.

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
        force_repack : bool, default=True
            Force repack the clauses.
        """
        cfg = self.config
        sym = self._symbols(ascii)
        and_, contra = sym[0], sym[4]
        names = feat_names if feat_names is not None else [f"X{i}" for i in range(cfg._n_raw_patch_feats)]
        clauses = self.get_clauses(force_repack=force_repack)
        weights = clauses.clause_weights
        has_pos = clauses.position_bounds is not None
        sec_class, keys = self._sections(sort)

        use_rich = False
        if not force_no_rich:
            try:
                import rich  # noqa: F401

                use_rich = True
            except ImportError:
                pass
        render = self._render_rich if use_rich else self._render_plain

        def join_slots(cell_lists: list[list[str]]) -> list[str]:
            # Pad every slot to its widest cell so the `and` symbols line up, drop slots empty in every row.
            widths = [max(len(cl[i]) for cl in cell_lists) for i in range(len(cell_lists[0]))]
            return [f" {and_} ".join(cell.center(w) for cell, w in zip(cl, widths) if w > 0) for cl in cell_lists]

        for bank in range(cfg._n_clause_banks):

            def key_value(name: str, c: int, bank: int = bank) -> float:
                if name == "len":
                    return int(clauses.clause_len[bank, c])
                if name == "!":
                    return int(clauses.has_contra[bank, c])
                if name == "w":
                    return -abs(float(weights[bank, c]))
                return -float(weights[int(name[1:]), c])

            def sort_key(c: int) -> tuple:
                return tuple(-key_value(n, c) if rev else key_value(n, c) for n, rev in keys)

            # Positive section first, then negative.
            sec_w = weights[sec_class] if cfg.coalesced else weights[bank]
            ids = range(cfg._n_clauses)
            sections = [sorted((c for c in ids if sec_w[c] >= 0), key=sort_key), sorted((c for c in ids if sec_w[c] < 0), key=sort_key)]
            footers = [f"showing {top_k} of {len(s)}" if top_k is not None and len(s) > top_k else None for s in sections]
            sections = [s[:top_k] if top_k is not None else s for s in sections]
            shown = [c for s in sections for c in s]
            if not shown:
                continue

            cells = [self._clause_cells(clauses, bank, c, names, sym) for c in shown]
            rules = join_slots([cl[2:] if has_pos else cl for cl in cells])
            positions = join_slots([cl[:2] for cl in cells]) if has_pos else None

            if cfg.coalesced:
                # One padded entry per class so the weights line up across rows.
                w_strs = [[self._weight_str(v) for v in weights[:, c]] for c in shown]
                w_width = [max(len(ws[k]) for ws in w_strs) for k in range(weights.shape[0])]
                w_col = ["[" + " ".join(v.rjust(w_width[k]) for k, v in enumerate(ws)) + "]" for ws in w_strs]
            else:
                w_col = [self._weight_str(weights[bank, c]) for c in shown]

            rows = {}
            for i, c in enumerate(shown):
                row = [str(c), contra if clauses.has_contra[bank, c] else "_", str(int(clauses.clause_len[bank, c])), w_col[i]]
                if has_pos and positions is not None:
                    row.append(positions[i])
                rows[c] = row + [rules[i]]

            header = ["ci", contra, "|C|", "W[:, ci]" if cfg.coalesced else f"W[{bank}, ci]"]
            justify = ["right", "center", "right", "right"]
            if has_pos and positions is not None:
                header.append("Position (row, col)".center(max(len(p) for p in positions)))
                justify.append("left")
            header.append("C[ci]".center(max(len(r) for r in rules)))
            justify.append("left")

            title = "Clauses (coalesced)" if cfg.coalesced else f"Class: {bank}"
            render(title, header, justify, [[rows[c] for c in s] for s in sections], footers)

    def _clause_cells(self, clauses: ClauseInfo, class_id: int, clause_id: int, feat_names: list[str], sym: tuple[str, ...]) -> list[str]:
        """Unpadded cells of one clause: [row, col] when there are patches, then one per feature, "" when unconstrained."""
        cfg = self.config
        bank = 0 if cfg.coalesced else class_id
        cells = []

        if clauses.position_bounds is not None:
            r0, r1, c0, c1 = (int(v) for v in clauses.position_bounds[bank, clause_id])
            cells.append(self._bound_str("row", r0, r1, 0, cfg._n_patches_y - 1, sym, False))
            cells.append(self._bound_str("col", c0, c1, 0, cfg._n_patches_x - 1, sym, False))

        bounds = clauses.feature_bounds[bank, clause_id].reshape(-1, 2)
        for name, (lo, hi), fmin, fmax in zip(feat_names, bounds, cfg._feat_mins, cfg._feat_maxs, strict=True):
            cells.append(self._bound_str(name, int(lo), int(hi), int(fmin), int(fmax), sym, fmax - fmin == 1))
        return cells

    def _sections(self, sort: str) -> tuple[int, list[tuple[str, bool]]]:
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

    @staticmethod
    def _symbols(ascii: bool) -> tuple[str, str, str, str, str]:
        """(and, not, le, ge, contradiction)"""
        return ("&", "~", "<=", ">=", "!") if ascii else ("∧", "¬", "≤", "≥", "⊥")

    @staticmethod
    def _bound_str(name: str, lo: int, hi: int, fmin: int, fmax: int, sym: tuple[str, ...], binary: bool) -> str:
        and_, not_, le, ge, _ = sym
        if (lo, hi) == (fmin, fmax):
            return ""
        if binary:  # lo > hi means both x and not x are included
            return f"({name} {and_} {not_}{name})" if lo > hi else name if lo == fmax else f"{not_}{name}"
        if lo > hi:  # contradiction, keep both bounds so the empty range is visible
            return f"({lo} {le} {name} {le} {hi})"
        if lo == hi:
            return f"({name} = {lo})"
        if lo == fmin:
            return f"({name} {le} {hi})"
        if hi == fmax:
            return f"({name} {ge} {lo})"
        return f"({lo} {le} {name} {le} {hi})"

    @staticmethod
    def _weight_str(w: float) -> str:
        return f"{w:+.0f}" if float(w).is_integer() else f"{w:+.3f}"

    @staticmethod
    def _render_rich(title: str, header: list[str], justify: list[str], sections: list[list[list[str]]], footers: list[str | None]) -> None:
        from rich import box
        from rich.console import Console
        from rich.markup import escape
        from rich.table import Table

        table = Table(title=title, box=box.SIMPLE_HEAVY)
        for h, j in zip(header, justify, strict=True):
            table.add_column(escape(h), justify=j, no_wrap=True)  # rich would read [ci] as markup
        for rows, footer in zip(sections, footers, strict=True):
            if not rows:
                continue
            for row in rows:
                table.add_row(*(escape(v) for v in row), style="red" if row[1] != "_" else None)
            if footer:
                table.add_row(*[""] * (len(header) - 1), f"[dim]{footer}[/]")
            table.add_section()
        Console().print(table)

    @staticmethod
    def _render_plain(
        title: str, header: list[str], justify: list[str], sections: list[list[list[str]]], footers: list[str | None]
    ) -> None:
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
