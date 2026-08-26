import importlib.util
import shutil
from dataclasses import dataclass, field
from typing import Literal, TypedDict

import numpy as np

ACT_FN_CODES = {"softmax": 0, "sigmoid": 1, "identity": 2}
LOSS_FN_CODES = {"ce": 0, "mse": 1, "mae": 2, "sce": 3, "asl": 4, "tversky": 5, "huber": 6}


@dataclass()
class TMArgs:
    n_clauses: int
    s: float
    dim: tuple[int, int, int]
    n_classes: int
    feat_mins: int | np.ndarray = 0
    feat_maxs: int | np.ndarray = 1
    patch_dim: tuple[int, int] = (0, 0)
    stride: tuple[int, int] = (1, 1)
    lr: float = 0.1
    lambda_: float | tuple[float, float] = 1.0
    act_fn: Literal["softmax", "sigmoid", "identity"] = "softmax"
    loss_fn: Literal["ce", "sce", "mse", "mae", "huber", "tversky", "asl"] = "ce"
    loss_fn_kwargs: dict = field(default_factory=dict)
    weighted: bool = True
    bias: bool = False
    max_weight: float = float(np.finfo(np.float32).max)
    coalesced: bool = True
    negated_literals: bool = True
    position_literals: bool = True
    negative_clauses: bool = True
    allow_polarity_change: bool = True
    max_includes: int = -1
    n_states: int = 256
    include_state: int = -1
    ta_init: Literal["random", "middle", "random_include"] | str | int = "random_include"
    weight_init: Literal["random"] | float = "random"
    bias_init: Literal["random"] | float = "random"
    skip_t1a_fb: bool = False
    skip_t1b_fb: bool = False
    skip_t2_fb: bool = False
    track_patch_weights: bool = True
    boost_tp_fb: bool = True
    seed: int = -1

    # Device specific arguments
    device: Literal["cpu", "cuda"] = "cpu"
    n_threads: int = 1
    compile_flags: None | list[str] = None
    grid_size: int | None = None
    block_size: int = 256
    warps_per_clause: int = 1

    def __post_init__(self):
        if self.device == "cuda" and importlib.util.find_spec("cupy") is None:
            raise ImportError("`device='cuda'` requires `cupy` to be installed. But `cupy` is not available in the current environment.")

        if self.device == "cpu" and not (shutil.which("gcc") or shutil.which("clang")):
            raise OSError("`device='cpu'` requires `gcc` or `clang` to be available in the PATH. But no suitable compiler was found.")

        if self.act_fn not in ACT_FN_CODES:
            raise ValueError(f"act_fn must be one of {set(ACT_FN_CODES)}, got '{self.act_fn}'")

        if self.loss_fn not in LOSS_FN_CODES:
            raise ValueError(f"loss_fn must be one of {set(LOSS_FN_CODES)}, got '{self.loss_fn}'")

        self.n_threads = max(1, self.n_threads)
        self.warps_per_clause = max(1, self.warps_per_clause)

        self.n_clauses = max(1, int(self.n_clauses))

        self.s = max(1.0, float(self.s))

        self.patch_dim = (
            self.dim[0] if self.patch_dim[0] <= 0 or self.patch_dim[0] > self.dim[0] else self.patch_dim[0],
            self.dim[1] if self.patch_dim[1] <= 0 or self.patch_dim[1] > self.dim[1] else self.patch_dim[1],
        )

        if self.include_state == -1:
            self.include_state = self.n_states // 2
        else:
            self.include_state = min(self.include_state, self.n_states - 1)

        if self.seed < 0:
            self.seed = np.random.randint(0, 1 << 30)
        elif self.seed == 0:
            self.seed = 1
        else:
            self.seed = int(self.seed)

        n_feat = self.patch_dim[0] * self.patch_dim[1] * self.dim[2]

        if np.isscalar(self.feat_mins):
            self.feat_mins = np.full(n_feat, self.feat_mins, dtype=np.int32)
        else:
            self.feat_mins = np.asarray(self.feat_mins, dtype=np.int32)
            assert self.feat_mins.shape == (n_feat,), f"feat_mins must have shape ({n_feat},), got {self.feat_mins.shape}"

        if np.isscalar(self.feat_maxs):
            self.feat_maxs = np.full(n_feat, self.feat_maxs, dtype=np.int32)
        else:
            self.feat_maxs = np.asarray(self.feat_maxs, dtype=np.int32)
            assert self.feat_maxs.shape == (n_feat,), f"feat_maxs must have shape ({n_feat},), got {self.feat_maxs.shape}"

        if isinstance(self.lambda_, tuple):
            assert len(self.lambda_) == 2, f"lambda_ tuple must be (lambda_plus, lambda_minus), got {len(self.lambda_)} elements"
            self.lambda_plus = float(self.lambda_[0])
            self.lambda_minus = float(self.lambda_[1])
        else:
            self.lambda_plus = float(self.lambda_)
            self.lambda_minus = float(self.lambda_)

        if isinstance(self.ta_init, int):
            assert 0 <= self.ta_init <= self.n_states - 1, (
                f"ta_init must be within 0 and n_states, 'middle' or 'random', got {self.ta_init}."
            )

        if self.weighted and isinstance(self.weight_init, float):
            assert self.weight_init > 0, f"weight_init must be positive float, or 'random', got {self.weight_init}"

        assert isinstance(self.bias_init, float) or self.bias_init == "random", (
            f"bias_init must be 'random' or float, got {self.bias_init}"
        )


class T_args(TypedDict, total=False):
    feat_mins: int | np.ndarray
    feat_maxs: int | np.ndarray
    patch_dim: tuple[int, int]
    stride: tuple[int, int]
    lr: float
    lambda_: float | tuple[float, float]
    act_fn: Literal["softmax", "sigmoid", "identity"]
    loss_fn: Literal["ce", "sce", "mse", "mae", "huber", "tversky", "asl"]
    loss_fn_kwargs: dict
    weighted: bool
    bias: bool
    max_weight: float
    coalesced: bool
    negated_literals: bool
    position_literals: bool
    negative_clauses: bool
    allow_polarity_change: bool
    max_includes: int
    n_states: int
    include_state: int
    ta_init: Literal["random", "middle", "random_include"] | str | int
    weight_init: Literal["random"] | float
    bias_init: Literal["random"] | float
    skip_t1a_fb: bool
    skip_t1b_fb: bool
    skip_t2_fb: bool
    track_patch_weights: bool
    boost_tp_fb: bool
    seed: int
    device: Literal["cpu", "cuda"]
    n_threads: int
    compile_flags: list[str]
    grid_size: int | None
    block_size: int
    warps_per_clause: int
