try:
    from rich.console import Console
    from rich.table import Table
    import rich.box as box

    _console = Console()
    _RICH = True
except ImportError:
    _RICH = False


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
        col_width = 10
        sep = "+" + "+".join(["-" * (col_width + 2)] * (len(headers) + 1)) + "+"
        header = f"| {id:>{col_width}} |" + "".join(f" {h:>{col_width}} |" for h in headers)
        print(sep)
        print(header)
        print(sep)
        for name, row in rows.items():
            line = f"| {name:>{col_width}} |" + "".join(f" {str(row.get(h, '-')):>{col_width}} |" for h in headers)
            print(line)
        print(sep)
