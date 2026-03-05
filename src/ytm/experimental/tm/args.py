import numpy as np
from typing import Literal
from dataclasses import dataclass

@dataclass()
class TMArgs:
    n_clauses: int
    T: float | int
    s: float
    dim: tuple[int, int, int]
    n_classes: int
    patch_dim: tuple[int, int] | None = None
    q: float = 1.0
    weighted: bool = True
    max_weight: float = float(np.finfo(np.float32).max)
    coalesced: bool = True
    negated_literals: bool = True
    position_literals: bool = True
    negative_clauses: bool = True
    allow_polarity_change: bool = True
    max_included_literals: int = -1
    n_states: int = 256
    include_state: int = -1
    skip_t1a_fb: bool = False
    skip_t1b_fb: bool = False
    skip_t2_fb: bool = False
    seed: int = np.random.randint(0, 1 << 30)

    # Device specific arguments
    device: Literal["cpu", "cuda"] = "cpu"
    n_threads: int = 1
    grid_size: int | None = None
    block_size: int = 128

    def __post_init__(self):
        if self.patch_dim is None:
            self.patch_dim = (self.dim[0], self.dim[1])

        if self.include_state == -1:
            self.include_state = self.n_states // 2

        if self.seed == 0:
            self.seed = 1


