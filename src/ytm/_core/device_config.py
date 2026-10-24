from dataclasses import dataclass, field, fields
from typing import Literal, TypedDict


def parse_device(device: str) -> tuple[str, int]:
    """Parse a device string into its kind and number.

    Parameters
    ----------
    device : str
        ``"cpu"``, ``"cpu:N"``, ``"cuda"`` or ``"cuda:N"``. For cpu, ``N`` is the number of
        threads, at least 1, default 1. For cuda, ``N`` is the GPU id, at least 0, default 0.

    Returns
    -------
    kind : {"cpu", "cuda"}
        Device kind.

    n : int
        Thread count for cpu, GPU id for cuda.

    Raises
    ------
    ValueError
        If the kind is unknown, or ``N`` is not an integer or out of range.
    """
    kind, _, spec = device.partition(":")
    kind = kind.strip().lower()

    if kind not in ("cpu", "cuda"):
        raise ValueError(f"Unsupported device: {device!r}. Expected 'cpu[:n_threads]' or 'cuda[:gpu_id]'.")

    if spec == "":
        n = 1 if kind == "cpu" else 0
    else:
        try:
            n = int(spec)
        except ValueError:
            raise ValueError(f"Unsupported device: {device!r}. Expected an integer after ':', got {spec!r}.") from None

    if kind == "cpu" and n < 1:
        raise ValueError(f"Unsupported device: {device!r}. cpu needs at least 1 thread, got {n}.")
    if kind == "cuda" and n < 0:
        raise ValueError(f"Unsupported device: {device!r}. cuda GPU id must be >= 0, got {n}.")

    return kind, n


@dataclass
class DeviceConfig:
    """Where and how a model runs.

    Parameters
    ----------
    device : str, default="cpu:1"
        ``"cpu[:n_threads]"`` or ``"cuda[:gpu_id]"``. See :func:`parse_device`.

    block_size : int, default=256
        CUDA threads per block. Rounded down to a multiple of the warp size and capped at the
        GPU maximum. Not used on cpu.
    """

    device: Literal["cpu", "cuda"] | str = "cpu:1"

    # cuda
    block_size: int = 256

    kind: str = field(init=False)
    n: int = field(init=False)

    def __post_init__(self):
        self.kind, self.n = parse_device(self.device)


def split_device_kwargs(opt: dict) -> tuple[dict, dict]:
    """Split a kwargs dict into the model config fields and the DeviceConfig fields."""
    names = {f.name for f in fields(DeviceConfig) if f.init}
    cfg_kw = {k: v for k, v in opt.items() if k not in names}
    dev_kw = {k: v for k, v in opt.items() if k in names}
    return cfg_kw, dev_kw


class T_DeviceConfig(TypedDict, total=False):
    device: Literal["cpu", "cuda"] | str

    # cuda
    block_size: int
