from dataclasses import dataclass
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
    grid_size: int | None = None
    block_size: int = 256
    warps_per_clause: int = 1

    def __post_init__(self):
        self.kind, self.n = parse_device(self.device)


class T_DeviceConfig(TypedDict, total=False):
    device: Literal["cpu", "cuda"] | str

    # cuda
    grid_size: int | None
    block_size: int
    warps_per_clause: int
