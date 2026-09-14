import pathlib
import re
from dataclasses import asdict

import pytest

from ytm._core.config import BaseTMConfig
from ytm._discrete.config import RegressionConfig, TMConfig

DEFAULTS: dict = {"n_clauses": 8, "s": 10.0, "dim": (6, 6, 1), "n_classes": 3, "feat_maxs": 3, "seed": 7}

C_SOURCES = list(pathlib.Path("src/ytm/_core/backends").glob("*.[ch]")) + list(pathlib.Path("src/ytm/_discrete/backends").glob("*.c"))


def config_variants():
    """One of each config class, with the options that change the emitted header."""
    return [
        BaseTMConfig(**DEFAULTS),
        BaseTMConfig(**{**DEFAULTS, "dim": (28, 28, 1), "patch_dim": (10, 10), "stride": (2, 3)}),
        BaseTMConfig(**{**DEFAULTS, "coalesced": True, "negated_literals": False, "position_literals": False}),
        TMConfig(**DEFAULTS, T=50.0),
        RegressionConfig(**{**DEFAULTS, "n_classes": 1}, T=(0.0, 50.0), y_range=(-5.0, 15.0)),
    ]


class TestRoundTrip:
    """`asdict` is how a model is rebuilt, so it has to carry exactly the user fields and no more."""

    @pytest.mark.parametrize("cfg", config_variants(), ids=lambda c: type(c).__name__)
    def test_rebuilding_from_asdict_gives_an_identical_header(self, cfg):
        """The header is what the C is compiled from, so an identical header means an identical model."""
        assert type(cfg)(**asdict(cfg))._header == cfg._header

    @pytest.mark.parametrize("cfg", config_variants(), ids=lambda c: type(c).__name__)
    def test_asdict_carries_no_derived_fields(self, cfg):
        """Derived values are machine and version specific, so shipping them would pin a saved model."""
        assert [k for k in asdict(cfg) if k.startswith("_")] == []


class TestInputNormalisation:
    """Several spellings of the same input must land on the same resolved value."""

    @pytest.mark.parametrize(
        "spellings, resolved",
        [
            ([6, (6,), (6, 1), (6, 1, 1)], (6, 1, 1)),  # a length 6 vector
            ([(6, 6), (6, 6, 1)], (6, 6, 1)),  # a 6x6 image
        ],
        ids=["vector", "image"],
    )
    def test_dim_spellings_agree(self, spellings, resolved):
        """Trailing axes are padded with 1, so shorter spellings must reach the same resolved shape."""
        cfgs = [BaseTMConfig(**{**DEFAULTS, "dim": d}) for d in spellings]
        assert {c._dim for c in cfgs} == {resolved}
        assert len({c._header for c in cfgs}) == 1

    def test_patch_dim_none_covers_the_whole_image(self):
        whole = BaseTMConfig(**{**DEFAULTS, "dim": (8, 8, 1), "patch_dim": None})
        explicit = BaseTMConfig(**{**DEFAULTS, "dim": (8, 8, 1), "patch_dim": (8, 8)})
        assert whole._patch_dim == explicit._patch_dim == (8, 8)
        assert whole._n_patches == explicit._n_patches == 1

    def test_patch_dim_larger_than_dim_is_clamped_to_it(self):
        clamped = BaseTMConfig(**{**DEFAULTS, "dim": (8, 8, 1), "patch_dim": (99, 99)})
        assert clamped._patch_dim == (8, 8)

    @pytest.mark.parametrize("seed", [None, 0, -5])
    def test_unset_seed_becomes_a_usable_positive_one(self, seed):
        """The seed is stored, so it has to be a concrete value rather than a request for randomness."""
        cfg = BaseTMConfig(**{**DEFAULTS, "seed": seed})
        assert isinstance(cfg.seed, int) and cfg.seed > 0
        assert asdict(cfg)["seed"] == cfg.seed


def test_every_macro_used_in_an_if_is_emitted():
    """An undefined macro in `#if` is silently 0, so a missing one compiles the wrong branch without
    any error. Macros used as values fail loudly at compile time instead, so they need no test."""
    used = set()
    for src in C_SOURCES:
        body = src.read_text()
        body = body[body.index("#endif") :] if "IS_NEOVIM_CLANGD_ENV" in body else body  # skip the lsp stubs
        for line in re.findall(r"^\s*#(?:el)?if\s+(.+)$", body, re.MULTILINE):
            used |= set(re.findall(r"\b[A-Z][A-Z0-9_]{2,}\b", line))

    emitted = set(re.findall(r"#define (\w+)", BaseTMConfig(**DEFAULTS)._header)) | set(
        re.findall(r"#define (\w+)", TMConfig(**DEFAULTS, T=50.0)._header)
    )
    derived_in_c = {"N_PATCHES"}  # common.h derives this one itself

    assert used - emitted - derived_in_c == set(), "macro read by the C but never defined"
