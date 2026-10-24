from dataclasses import dataclass, fields
from typing import Literal, TypedDict


def parse_device(device: str) -> tuple[str, int]:
    """Parse device string.

    cuda:N means cuda on gpuid N.
    cpu:N means cpu with N threads.
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

    n = max(1, n) if kind == "cpu" else max(0, n)
    return kind, n


@dataclass()
class DeviceConfig:
    device: Literal["cpu", "cuda"] | str = "cpu:1"

    # cuda
    block_size: int = 256

    def __post_init__(self):
        self.kind, self.n = parse_device(self.device)


def split_device_kwargs(opt: dict) -> tuple[dict, dict]:
    """Split a kwargs dict into the model config fields and the DeviceConfig fields."""
    names = {f.name for f in fields(DeviceConfig)}
    dev_kw = {k: opt.pop(k) for k in list(opt) if k in names}
    return opt, dev_kw


class T_DeviceConfig(TypedDict, total=False):
    device: Literal["cpu", "cuda"] | str

    # cuda
    block_size: int
