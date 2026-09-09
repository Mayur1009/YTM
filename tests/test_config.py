from dataclasses import asdict, fields

import numpy as np
import pytest

from ytm._core.config import BaseTMConfig

DEFAULTS: dict = {"n_clauses": 10, "s": 10.0, "dim": (28, 28), "n_classes": 10}

# Configs whose derived var invariants should hold regardless of the shape of the input.
DERIVED_CASES: list[dict] = [
    {},
    {"dim": 784},
    {"dim": (28, 28, 3), "feat_maxs": 255},
    {"dim": (28, 28), "patch_dim": (10, 10), "stride": (2, 2)},
    {"dim": (28, 28), "negated_literals": False, "position_literals": False},
    {"feat_mins": np.arange(784), "feat_maxs": np.arange(784) + 4},
]


def make_config(**kwargs) -> BaseTMConfig:
    return BaseTMConfig(**{**DEFAULTS, **kwargs})


class TestDim:
    @pytest.mark.parametrize(
        "dim, expected",
        [
            (784, (784, 1, 1)),
            ((784,), (784, 1, 1)),
            ((28, 28), (28, 28, 1)),
            ((28, 28, 3), (28, 28, 3)),
            (np.array([28, 28, 3]), (28, 28, 3)),
        ],
    )
    def test_padded_to_three(self, dim, expected):
        assert make_config(dim=dim)._dim == expected

    @pytest.mark.parametrize("dim", [(), (1, 2, 3, 4), (28.5, 28), ("28", 28), (None, 28)])
    def test_rejects_bad_shapes_and_types(self, dim):
        with pytest.raises(AssertionError):
            make_config(dim=dim)


class TestPatchDim:
    @pytest.mark.parametrize(
        "patch_dim, expected",
        [
            (None, (28, 28)),
            ((0, 0), (28, 28)),
            ((10, 10), (10, 10)),
            ((99, 5), (28, 5)),
        ],
    )
    def test_falls_back_to_full_input(self, patch_dim, expected):
        assert make_config(patch_dim=patch_dim)._patch_dim == expected


class TestStride:
    @pytest.mark.parametrize("stride", [(1, 1), (2, 3), (28, 28), np.array([2, 3])])
    def test_kept_as_plain_ints(self, stride):
        _stride = make_config(stride=stride)._stride
        assert _stride == (int(stride[0]), int(stride[1]))
        assert all(type(st) is int for st in _stride)

    @pytest.mark.parametrize("stride", [(1,), (1, 2, 3), (0, 1), (1, 0), (-2, 1), (1.5, 1)])
    def test_rejects_non_positive_ints(self, stride):
        with pytest.raises(AssertionError):
            make_config(stride=stride)


class TestNStates:
    def test_allows_two(self):
        assert make_config(n_states=2)._include_state == 1

    @pytest.mark.parametrize("n_states", [1, 0, -4])
    def test_rejects_fewer_than_two(self, n_states):
        with pytest.raises(AssertionError):
            make_config(n_states=n_states)


class TestIncludeState:
    @pytest.mark.parametrize("include_state, expected", [(None, 128), (-1, 128), (0, 0), (200, 200), (255, 255)])
    def test_defaults_to_middle(self, include_state, expected):
        assert make_config(include_state=include_state)._include_state == expected

    @pytest.mark.parametrize("include_state", [256, 300, -2, -5])
    def test_rejects_out_of_range(self, include_state):
        with pytest.raises(AssertionError):
            make_config(include_state=include_state)


class TestTaInit:
    @pytest.mark.parametrize("ta_init", ["middle", "random", "random_include", "random:5", "random:0", 0, 128, 255])
    def test_accepts_known_forms(self, ta_init):
        assert make_config(ta_init=ta_init).ta_init == ta_init

    @pytest.mark.parametrize("ta_init", ["random:", "random:-5", "random:abc", "random:1.5", "foo", "", 256, -1])
    def test_rejects_unknown_forms(self, ta_init):
        with pytest.raises(AssertionError):
            make_config(ta_init=ta_init)


class TestWeightInit:
    @pytest.mark.parametrize("weight_init", ["random", "random:2.5", "random:1", 1.0, 2, 0.5])
    def test_accepts_known_forms(self, weight_init):
        assert make_config(weight_init=weight_init).weight_init == weight_init

    @pytest.mark.parametrize("weight_init", ["random:", "random:0", "random:-2", "random:abc", "foo", "", 0, 0.0, -1.5])
    def test_rejects_non_positive_and_unknown_forms(self, weight_init):
        with pytest.raises(AssertionError):
            make_config(weight_init=weight_init)


class TestBiasInit:
    @pytest.mark.parametrize("bias", [True, False])
    @pytest.mark.parametrize("bias_init", ["random", 0.0, 2.0, 3])
    def test_accepts_known_forms_regardless_of_bias(self, bias, bias_init):
        assert make_config(bias=bias, bias_init=bias_init).bias_init == bias_init

    @pytest.mark.parametrize("bias_init", ["zeros", "", None])
    def test_rejects_unknown_forms(self, bias_init):
        with pytest.raises(AssertionError):
            make_config(bias_init=bias_init)


class TestFeatBounds:
    def test_broadcasts_scalars(self):
        cfg = make_config(feat_mins=0, feat_maxs=255)
        assert cfg._feat_mins.shape == (784,)
        assert cfg._feat_maxs.shape == (784,)
        assert cfg._feat_mins.dtype == np.int32
        assert cfg._feat_maxs.dtype == np.int32
        assert np.all(cfg._feat_maxs == 255)

    def test_follows_patch_dim(self):
        assert make_config(dim=(28, 28, 3), patch_dim=(10, 10))._feat_mins.shape == (300,)

    def test_accepts_arrays(self):
        feat_maxs = np.arange(784)
        cfg = make_config(feat_maxs=feat_maxs)
        assert np.array_equal(cfg._feat_maxs, feat_maxs)
        assert cfg._feat_maxs.dtype == np.int32

    @pytest.mark.parametrize("bad", [np.zeros(5), np.zeros(785), np.zeros((28, 28))])
    def test_rejects_wrong_shape(self, bad):
        with pytest.raises(AssertionError):
            make_config(feat_mins=bad)
        with pytest.raises(AssertionError):
            make_config(feat_maxs=bad)


class TestClauseParams:
    @pytest.mark.parametrize("n_clauses, expected", [(10, 10), (1, 1), (0, 1), (-5, 1)])
    def test_n_clauses_clamped_to_at_least_one(self, n_clauses, expected):
        assert make_config(n_clauses=n_clauses)._n_clauses == expected

    @pytest.mark.parametrize("s, expected", [(10.0, 10.0), (1.0, 1.0), (0.5, 1.0), (-3.0, 1.0)])
    def test_s_clamped_to_at_least_one(self, s, expected):
        assert make_config(s=s)._s == expected


class TestMaxIncludes:
    @pytest.mark.parametrize("max_includes, expected", [(None, 1568), (-1, 1568), (0, 1568), (9999, 1568), (20, 20)])
    def test_clamps_to_n_literals(self, max_includes, expected):
        cfg = make_config(max_includes=max_includes)
        assert cfg._n_literals == 1568
        assert cfg._max_includes == expected


class TestSeed:
    @pytest.mark.parametrize("seed", [None, 0, -1, -99])
    def test_unset_or_non_positive_draws_a_seed(self, seed):
        drawn = make_config(seed=seed).seed
        assert isinstance(drawn, int)
        assert drawn > 0

    def test_explicit_seed_is_kept(self):
        assert make_config(seed=42).seed == 42

    def test_two_default_configs_get_different_seeds(self):
        assert make_config().seed != make_config().seed

    def test_drawn_seed_survives_asdict(self):
        """Resolved in place so a reloaded model keeps the same kernel RNG stream."""
        cfg = make_config(seed=None)
        assert asdict(cfg)["seed"] == cfg.seed

    def test_round_trip_reuses_the_drawn_seed(self):
        cfg = make_config(seed=None)
        assert BaseTMConfig(**asdict(cfg)).seed == cfg.seed


class TestDerivedVars:
    DERIVED = (
        "_n_clause_banks",
        "_total_clauses",
        "_n_patches_y",
        "_n_patches_x",
        "_n_patches",
        "_n_raw_patch_feats",
        "_n_position_feats",
        "_therm_bits",
        "_n_patch_feats",
        "_literal_offsets",
        "_lit_to_fid",
        "_n_literals",
        "_max_includes",
        "_header",
    )

    @pytest.mark.parametrize("kwargs", DERIVED_CASES)
    def test_every_derived_var_is_set(self, kwargs):
        cfg = make_config(**kwargs)
        assert [name for name in self.DERIVED if not hasattr(cfg, name)] == []

    @pytest.mark.parametrize("kwargs", DERIVED_CASES)
    def test_therm_bits_span_the_feature_range(self, kwargs):
        cfg = make_config(**kwargs)
        assert np.array_equal(cfg._therm_bits, cfg._feat_maxs - cfg._feat_mins)
        assert cfg._therm_bits.shape == (cfg._n_raw_patch_feats,)
        assert cfg._n_patch_feats == int(cfg._therm_bits.sum())

    def test_no_convolution(self):
        cfg = make_config()
        assert cfg._n_clause_banks == 1
        assert cfg._total_clauses == 10
        assert (cfg._n_patches_y, cfg._n_patches_x, cfg._n_patches) == (1, 1, 1)
        assert cfg._n_raw_patch_feats == 784
        assert cfg._n_position_feats == 0
        assert cfg._n_patch_feats == 784
        assert cfg._n_literals == 1568

    def test_strided_convolution(self):
        cfg = make_config(dim=(28, 28, 1), patch_dim=(10, 10), stride=(2, 2), feat_maxs=255, coalesced=False)
        assert cfg._n_clause_banks == 10
        assert cfg._total_clauses == 100
        assert (cfg._n_patches_y, cfg._n_patches_x, cfg._n_patches) == (10, 10, 100)
        assert cfg._n_raw_patch_feats == 100
        assert cfg._n_position_feats == 18
        assert cfg._n_patch_feats == 25500
        assert cfg._n_literals == 51036

    @pytest.mark.parametrize("coalesced, expected", [(True, 1), (False, 10)])
    def test_clause_banks_follow_coalesced(self, coalesced, expected):
        cfg = make_config(coalesced=coalesced)
        assert cfg._n_clause_banks == expected
        assert cfg._total_clauses == cfg._n_clause_banks * cfg._n_clauses

    def test_negated_literals_doubles_literal_count(self):
        with_neg = make_config(negated_literals=True)._n_literals
        without = make_config(negated_literals=False)._n_literals
        assert with_neg == 2 * without

    @pytest.mark.parametrize("kwargs", DERIVED_CASES)
    def test_literal_offsets_are_a_prefix_sum(self, kwargs):
        cfg = make_config(**kwargs)
        offsets = cfg._literal_offsets
        assert offsets.shape == (cfg._n_raw_patch_feats + 1,)
        assert offsets[0] == 0
        assert offsets[-1] == cfg._n_patch_feats
        assert np.all(np.diff(offsets) == cfg._therm_bits)

    @pytest.mark.parametrize("kwargs", DERIVED_CASES)
    def test_lit_to_fid_covers_every_literal_once(self, kwargs):
        cfg = make_config(**kwargs)
        assert cfg._lit_to_fid.shape == (cfg._n_patch_feats,)
        counts = np.bincount(cfg._lit_to_fid, minlength=cfg._n_raw_patch_feats)
        assert np.array_equal(counts, cfg._therm_bits)

    @pytest.mark.parametrize("kwargs", DERIVED_CASES)
    def test_c_side_arrays_are_contiguous_int32(self, kwargs):
        cfg = make_config(**kwargs)
        for name in ("_feat_mins", "_feat_maxs", "_therm_bits", "_literal_offsets", "_lit_to_fid"):
            arr = getattr(cfg, name)
            assert arr.dtype == np.int32, name
            assert arr.flags.c_contiguous, name


class TestConventions:
    """Fields are user input and serialize, underscore attributes are derived and do not."""

    def test_no_field_is_underscore_prefixed(self):
        assert [f.name for f in fields(BaseTMConfig) if f.name.startswith("_")] == []

    def test_asdict_has_no_derived_keys(self):
        assert [k for k in asdict(make_config()) if k.startswith("_")] == []

    @pytest.mark.parametrize(
        "name",
        ["_dim", "_patch_dim", "_stride", "_n_clauses", "_s", "_include_state", "_max_includes", "_n_literals"],
    )
    def test_derived_attributes_are_absent_from_asdict(self, name):
        cfg = make_config()
        assert hasattr(cfg, name)
        assert name not in asdict(cfg)

    def test_toolchain_state_is_not_serialized(self):
        """A model saved on one machine must re-resolve the compiler on another."""
        serialized = asdict(make_config())
        for name in ("_compiler", "_compiler_flags", "_omp_flags", "_n_threads", "_gpu_id", "_device_kind"):
            assert name not in serialized

    def test_user_input_is_not_mutated(self):
        cfg = make_config(dim=784, patch_dim=None, include_state=None, max_includes=None, feat_maxs=255)
        assert cfg.dim == 784
        assert cfg.patch_dim is None
        assert cfg.include_state is None
        assert cfg.max_includes is None
        assert cfg.feat_maxs == 255


def parse_defines(header: str) -> dict[str, str]:
    out = {}
    for line in header.splitlines():
        parts = line.strip().split(None, 2)
        if parts[:1] == ["#define"]:
            out[parts[1]] = parts[2] if len(parts) > 2 else ""
    return out


class TestHeader:
    @pytest.mark.parametrize("kwargs", DERIVED_CASES)
    def test_every_define_has_a_value(self, kwargs):
        defines = parse_defines(make_config(**kwargs)._header)
        assert defines
        assert [name for name, value in defines.items() if value == ""] == []

    @pytest.mark.parametrize("kwargs", DERIVED_CASES)
    def test_no_python_literals_leak_in(self, kwargs):
        """`True`, `False` and `None` are not valid C and would fail to compile."""
        defines = parse_defines(make_config(**kwargs)._header)
        assert [n for n, v in defines.items() if v in ("True", "False", "None")] == []

    @pytest.mark.parametrize("kwargs", DERIVED_CASES)
    def test_float_defines_are_float_literals(self, kwargs):
        defines = parse_defines(make_config(**kwargs)._header)
        for name in ("S", "MAX_WEIGHT"):
            assert defines[name].endswith("f"), name
            float(defines[name][:-1])

    def test_defines_match_the_derived_values(self):
        cfg = make_config(dim=(28, 28, 3), patch_dim=(10, 10), stride=(2, 2), feat_maxs=255, coalesced=False)
        defines = parse_defines(cfg._header)
        expected = {
            "TOTAL_CLAUSES": cfg._total_clauses,
            "CLASSES": cfg.n_classes,
            "HEIGHT": cfg._dim[0],
            "WIDTH": cfg._dim[1],
            "DEPTH": cfg._dim[2],
            "PATCH_HEIGHT": cfg._patch_dim[0],
            "PATCH_WIDTH": cfg._patch_dim[1],
            "STRIDE_Y": cfg._stride[0],
            "STRIDE_X": cfg._stride[1],
            "N_PATCHES_Y": cfg._n_patches_y,
            "N_PATCHES_X": cfg._n_patches_x,
            "N_PATCHES": cfg._n_patches,
            "N_RAW_PATCH_FEATS": cfg._n_raw_patch_feats,
            "N_PATCH_FEATS": cfg._n_patch_feats,
            "N_POSITION_FEATS": cfg._n_position_feats,
            "N_LITERALS": cfg._n_literals,
            "MAX_INCLUDED_LITERALS": cfg._max_includes,
            "INCLUDE_STATE": cfg._include_state,
            "MAX_TA_STATE": cfg.n_states - 1,
        }
        assert {k: int(defines[k]) for k in expected} == expected

    @pytest.mark.parametrize("value", [True, False])
    def test_bools_render_as_zero_or_one(self, value):
        defines = parse_defines(make_config(bias=value, coalesced=value)._header)
        assert defines["BIAS"] == str(int(value))
        assert defines["COALESCED"] == str(int(value))

    @pytest.mark.parametrize("value", [True, False])
    def test_skip_flags_are_inverted(self, value):
        defines = parse_defines(make_config(skip_t1a_fb=value)._header)
        assert defines["TYPE1A_FB"] == str(int(not value))
