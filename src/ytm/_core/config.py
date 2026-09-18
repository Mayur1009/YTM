from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypedDict

import numpy as np

from .utils import Feedback, enum_to_header


def _get_unsinged_type(val: int):
    if val <= (1 << 8):
        _dtype, _ctype = np.uint8, "uint8_t"
    elif val <= (1 << 16):
        _dtype, _ctype = np.uint16, "uint16_t"
    else:
        _dtype, _ctype = np.uint32, "uint32_t"

    return _dtype, _ctype


@dataclass()
class BaseTMConfig:
    n_clauses: int
    s: float
    dim: int | tuple[int] | tuple[int, int] | tuple[int, int, int]
    n_classes: int

    # discrete input
    feat_mins: int | Sequence[int] | np.ndarray = 0
    feat_maxs: int | Sequence[int] | np.ndarray = 1

    # convolution
    patch_dim: tuple[int, int] | None = None
    stride: tuple[int, int] = (1, 1)

    # conlutional interpretability
    track_patch_weights: bool = True

    # clause bank
    coalesced: bool = True
    negative_clauses: bool = True

    # literals
    negated_literals: bool = True
    position_literals: bool = True
    max_includes: int | None = None

    # TA states
    n_states: int = 256
    include_state: int | None = None
    ta_init: Literal["random", "middle", "random_include"] | str | int = "middle"

    # clause weights
    weighted: bool = True
    max_weight: float = float(1 << 30)
    allow_polarity_change: bool = True

    # feedback
    skip_t1a_fb: bool = False
    skip_t1b_fb: bool = False
    skip_t2_fb: bool = False
    boost_tp_inc: bool = True
    boost_tp_dec: bool = False

    # Random state
    seed: int | None = None

    def __post_init__(self):
        self._check_seed()
        self._check_params()
        self._check_feat_bounds()
        self._derive_vars()
        self._derive_types()
        self._build_header()

    def _check_seed(self):
        """Set a random seed when seed <= 0 or None, else use the provided seed."""
        if self.seed is None or self.seed <= 0:
            self.seed = int(np.random.randint(1, 1 << 31))
        else:
            self.seed = int(self.seed)

    def _check_params(self):
        """Clamp scalar params and validate the init options."""
        self._n_clauses = max(1, int(self.n_clauses))
        self._s = max(1.0, float(self.s))

        dim = (self.dim,) if isinstance(self.dim, int) else tuple(self.dim)
        assert 1 <= len(dim) <= 3, f"dim must be an int or a tuple of length 1, 2 or 3, got {self.dim}"
        assert all(isinstance(d, (int, np.integer)) for d in dim), f"dim entries must be ints, got {self.dim}"

        # (H,) -> (H, 1, 1), (H, W) -> (H, W, 1), (H, W, C) stays as is.
        padded = [int(d) for d in dim] + [1] * (3 - len(dim))
        self._dim: tuple[int, int, int] = (padded[0], padded[1], padded[2])

        # patch dim
        if self.patch_dim is None:
            self._patch_dim = (self._dim[0], self._dim[1])
        else:
            self._patch_dim = (
                self._dim[0] if self.patch_dim[0] <= 0 or self.patch_dim[0] > self._dim[0] else self.patch_dim[0],
                self._dim[1] if self.patch_dim[1] <= 0 or self.patch_dim[1] > self._dim[1] else self.patch_dim[1],
            )

        # Stride
        assert len(self.stride) == 2, f"stride must be a tuple of length 2, got {self.stride}"
        assert all(isinstance(st, (int, np.integer)) and st > 0 for st in self.stride), (
            f"stride entries must be positive ints, got {self.stride}"
        )
        self._stride = (int(self.stride[0]), int(self.stride[1]))

        # number of patches and raw features
        self._n_patches_y = ((self._dim[0] - self._patch_dim[0]) // self._stride[0]) + 1
        self._n_patches_x = ((self._dim[1] - self._patch_dim[1]) // self._stride[1]) + 1
        self._n_patches = self._n_patches_y * self._n_patches_x
        self._n_raw_patch_feats = self._patch_dim[0] * self._patch_dim[1] * self._dim[2]

        # TA states, need at least an include and an exclude state
        assert self.n_states >= 2, f"n_states must be at least 2, got {self.n_states}"

        # Include state, None or -1 means the middle state
        if self.include_state is None or self.include_state == -1:
            self._include_state = self.n_states // 2
        else:
            assert 0 <= self.include_state <= self.n_states - 1, (
                f"include_state must be within 0 and n_states - 1 ({self.n_states - 1}), None or -1, got {self.include_state}"
            )
            self._include_state = self.include_state

        # TA initial state
        if isinstance(self.ta_init, str) and self.ta_init.startswith("random:"):
            band = self.ta_init[len("random:") :]
            assert band.isdigit(), f"ta_init 'random:N' needs a non negative int N, got {self.ta_init!r}"
        else:
            assert self.ta_init in ("middle", "random", "random_include") or (
                isinstance(self.ta_init, int) and 0 <= self.ta_init <= self.n_states - 1
            ), (
                f"ta_init must be 'middle', 'random', 'random_include', 'random:N', "
                f"or an int within 0 and n_states - 1 ({self.n_states - 1}), got {self.ta_init}"
            )

    def _resolve_feat_bound(self, name: str, value: int | Sequence[int] | np.ndarray) -> np.ndarray:
        n_feat = self._n_raw_patch_feats
        depth = self._dim[2]

        if np.ndim(value) == 0:
            return np.full(n_feat, value, dtype=np.int32)

        arr = np.asarray(value, dtype=np.int32, order="C")

        if self._n_patches > 1:
            assert arr.shape == (depth,), f"{name} should either be a scalar, or tuple of length dim[2]."
            return np.tile(arr, self._patch_dim[0] * self._patch_dim[1])

        assert arr.size == n_feat, f"{name} must be a scalar or have {n_feat} entries, got shape {arr.shape}"

        return arr.reshape(-1)

    def _check_feat_bounds(self):
        """Broadcast feat bounds to one int32 entry per raw patch feature."""
        self._feat_mins = self._resolve_feat_bound("feat_mins", self.feat_mins)
        self._feat_maxs = self._resolve_feat_bound("feat_maxs", self.feat_maxs)
        self._therm_bits = self._feat_maxs - self._feat_mins
        self._all_binary_feats = bool(np.all(self._therm_bits == 1))

        assert np.all(self._feat_maxs >= self._feat_mins), (
            f"feat_maxs must be >= feat_mins for every feature, violated at "
            f"{np.flatnonzero(self._feat_maxs < self._feat_mins)[:8].tolist()}"
        )

    def _derive_vars(self):
        """Calculate variables from the params, so that they can be used later."""
        # Clause banks and total clauses. Coalesced shares one bank across all classes.
        self._n_clause_banks = 1 if self.coalesced else self.n_classes
        self._total_clauses = self._n_clause_banks * self._n_clauses

        self._n_position_feats = (self._n_patches_y - 1) + (self._n_patches_x - 1)
        self._n_patch_feats = int(np.sum(self._therm_bits))

        # literal offsets for thermometer encoded features
        self._literal_offsets = np.zeros(self._n_raw_patch_feats + 1, dtype=np.int32)
        self._literal_offsets[1:] = np.cumsum(self._therm_bits)

        # final number of actual literals
        self._n_literals = self._n_patch_feats + self._n_position_feats
        if self.negated_literals:
            self._n_literals *= 2

        # Clause budget
        if self.max_includes is None or self.max_includes <= 0 or self.max_includes > self._n_literals:
            self._max_includes = self._n_literals
        else:
            self._max_includes = self.max_includes

    def _derive_types(self):
        # TA states
        self._ta_dtype, self._ta_ctype = _get_unsinged_type(self.n_states)

        # therm bits
        self._fbound_dtype, self._fbound_ctype = _get_unsinged_type(int(self._therm_bits.max()) + 1)

        # number of features
        self._nfeat_dtype, self._nfeat_ctype = _get_unsinged_type(self._n_raw_patch_feats + 1)

        # number of patches
        self._npatches_dtype, self._npatches_ctype = _get_unsinged_type(self._n_patches)

        # patch index bounds
        self._pbound_dtype, self._pbound_ctype = _get_unsinged_type(max(self._n_patches_y, self._n_patches_x))

        # number of literals
        self._nlits_dtype, self._nlits_ctype = _get_unsinged_type(self._n_literals + 1)

        self._therm_bits = self._therm_bits.astype(self._fbound_dtype)
        self._literal_offsets = self._literal_offsets.astype(self._nlits_dtype)

    def _build_header(self):
        self._header = f"""
#define TOTAL_CLAUSES {self._total_clauses}
#define CLASSES {self.n_classes}
#define S {float(self._s)}f

#define HEIGHT {self._dim[0]}
#define WIDTH {self._dim[1]}
#define DEPTH {self._dim[2]}
#define PATCH_HEIGHT {self._patch_dim[0]}
#define PATCH_WIDTH {self._patch_dim[1]}
#define STRIDE_Y {self._stride[0]}
#define STRIDE_X {self._stride[1]}
#define N_PATCHES_Y {self._n_patches_y}
#define N_PATCHES_X {self._n_patches_x}
#define N_PATCHES {self._n_patches}

#define N_RAW_PATCH_FEATS {self._n_raw_patch_feats}
#define N_PATCH_FEATS {self._n_patch_feats}
#define N_POSITION_FEATS {self._n_position_feats}
#define N_LITERALS {self._n_literals}
#define MAX_INCLUDED_LITERALS {self._max_includes}
#define NEGATED_LITERALS {int(self.negated_literals)}
#define POSITION_LITERALS {int(self.position_literals)}
#define ALL_BINARY_FEATS {int(self._all_binary_feats)}

#define INCLUDE_STATE {self._include_state}
#define MAX_TA_STATE {self.n_states - 1}

#define COALESCED {int(self.coalesced)}
#define NEGATIVE_CLAUSES {int(self.negative_clauses)}
#define WEIGHTED {int(self.weighted)}
#define MAX_WEIGHT {float(self.max_weight)}f
#define ALLOW_POLARITY_CHANGE {int(self.allow_polarity_change)}
#define TRACK_PATCH_WEIGHTS {int(self.track_patch_weights)}

#define TYPE1A_FB {int(not self.skip_t1a_fb)}
#define TYPE1B_FB {int(not self.skip_t1b_fb)}
#define TYPE2_FB {int(not self.skip_t2_fb)}
#define BOOST_TP_INC {int(self.boost_tp_inc)}
#define BOOST_TP_DEC {int(self.boost_tp_dec)}

#define TA_STATE_T {self._ta_ctype}
#define FBOUND_T {self._fbound_ctype}
#define PBOUND_T {self._pbound_ctype}
#define NFEAT_T {self._nfeat_ctype}
#define NPATCHES_T {self._npatches_ctype}
#define NLITS_T {self._nlits_ctype}

{enum_to_header("FB", Feedback)}
"""


class T_BaseTMConfig(TypedDict, total=False):
    # discrete input
    feat_mins: int | Sequence[int] | np.ndarray
    feat_maxs: int | Sequence[int] | np.ndarray

    # convolution
    patch_dim: tuple[int, int] | None
    stride: tuple[int, int]

    # conlutional interpretability
    track_patch_weights: bool

    # clause bank
    coalesced: bool
    negative_clauses: bool

    # literals
    negated_literals: bool
    position_literals: bool
    max_includes: int | None

    # TA states
    n_states: int
    include_state: int | None
    ta_init: Literal["random", "middle", "random_include"] | str | int

    # clause weights
    weighted: bool
    max_weight: float
    allow_polarity_change: bool

    # feedback
    skip_t1a_fb: bool
    skip_t1b_fb: bool
    skip_t2_fb: bool
    boost_tp_inc: bool
    boost_tp_dec: bool

    # Random state
    seed: int | None
