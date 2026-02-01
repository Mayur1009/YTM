from dataclasses import dataclass, field
from typing import Literal, TypedDict
import numpy as np

@dataclass
class TMOpt:
    backend: str = "cpu"
    backend_args: dict = field(default_factory=dict)
    q: float = 1.0
    patch_dim: tuple[int, int] | None = None
    number_of_ta_states: int = 256
    max_included_literals: int | None = None
    negated_literals: bool = True
    init_neg_weights: bool = False
    negative_polarity: bool = False
    encode_loc: bool = False
    coalesced: bool = False
    weighted: bool = False
    max_weight: float = float(np.iinfo(np.int32).max)
    allow_polarity_change: bool = False
    initial_weight: float = 1.0
    initial_state: int | Literal["random"] | Literal["middle"] = "middle"
    include_state: int | Literal["middle"] = "middle"
    type1a_fb: bool = True
    type1b_fb: bool = True
    type2_fb: bool = True
    seed: int | None = None

class TMOptArgs(TypedDict, total=False):
    backend: str
    backend_args: dict
    q: float
    patch_dim: tuple[int, int] | None
    number_of_ta_states: int
    max_included_literals: int | None
    negated_literals: bool
    init_neg_weights: bool
    negative_polarity: bool
    encode_loc: bool
    coalesced: bool
    weighted: bool
    max_weight: float
    allow_polarity_change: bool
    initial_weight: float
    initial_state: int | Literal["random"] | Literal["middle"]
    include_state: int | Literal["middle"]
    type1a_fb: bool
    type1b_fb: bool
    type2_fb: bool
    seed: int | None
