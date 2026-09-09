import numpy as np
import pytest

from ytm._core.backends.cpu import CPUDevice
from ytm._core.config import BaseTMConfig
from ytm._core.device_config import DeviceConfig

DEFAULTS: dict = {"n_clauses": 4, "s": 10.0, "dim": (8, 8), "n_classes": 2, "seed": 1, "feat_maxs": 7}


class Device(CPUDevice):
    """Concrete only so the shared compile and pointer setup can be exercised."""

    def fit_epoch(self, X, Y, clause_drop_p, batch_size, **kwargs): ...
    def fit_sample(self, X, Y, e, **kwargs): ...
    def infer(self, X, batch_size): ...
    def transform(self, X, batch_size, force_repack=False): ...
    def transform_patchwise(self, X, batch_size, force_repack=False): ...
    def wic(self, class_id, polarity, pw_th=0.0, force_repack=False): ...
    def wac(self, X, target_classes, polarity, force_repack=False): ...


def make_device(device: str = "cpu:1", **kwargs) -> Device:
    return Device(BaseTMConfig(**{**DEFAULTS, **kwargs}), DeviceConfig(device=device))


@pytest.fixture(scope="module")
def dev() -> Device:
    """One compile for every test that only reads, compiling takes ~0.5s."""
    return make_device()


class TestBuildCode:
    def test_starts_with_the_generated_header(self, dev):
        assert dev._build_code().startswith(dev.config._header)

    @pytest.mark.parametrize("marker", ["INLINE_FN", "S_INV", "mix64", "geom_sample", "pack_clauses", "wic"])
    def test_shared_sources_are_included(self, dev, marker):
        assert marker in dev._build_code()

    def test_subclasses_can_append(self):
        class Extended(Device):
            def _build_code(self) -> str:
                return super()._build_code() + "\nint extra_marker(void) { return 42; }\n"

        d = Extended(BaseTMConfig(**DEFAULTS), DeviceConfig())
        assert d.lib.extra_marker() == 42

    def test_reads_sources_independently_of_cwd(self, tmp_path, monkeypatch, dev):
        before = dev._build_code()
        monkeypatch.chdir(tmp_path)
        assert dev._build_code() == before


class TestCompile:
    @pytest.mark.parametrize("symbol", ["pack_clauses", "wic", "wac_sample", "set_num_threads"])
    def test_entry_points_are_exported(self, dev, symbol):
        assert hasattr(dev.lib, symbol)

    def test_bad_code_raises_with_the_compiler_output(self, dev):
        with pytest.raises(RuntimeError, match="Failed to compile"):
            dev._compile_code("this is not c")

    def test_header_values_reach_the_binary(self):
        """A define that changes the model must produce a different object."""
        a, b = make_device(n_clauses=4), make_device(n_clauses=8)
        assert "#define TOTAL_CLAUSES 4" in a._build_code()
        assert "#define TOTAL_CLAUSES 8" in b._build_code()


class TestPointers:
    POINTERS = (
        "p_clause_feat_bounds", "p_clause_position_bounds", "p_bounded_feat_ids", "p_n_bounded_feats",
        "p_clause_density", "p_is_clause_synced", "p_ta_states", "p_clause_weights", "p_bias",
        "p_patch_weights", "p_feat_mins", "p_feat_maxs", "p_literal_offsets",
    )

    def test_every_array_has_a_pointer(self, dev):
        assert [n for n in self.POINTERS if not getattr(dev, n, None)] == []

    def test_pointers_alias_the_live_arrays(self, dev):
        """C writes through these, so they must not point at a copy."""
        assert dev.p_ta_states.contents.value == dev.ta_states.flat[0]
        dev.ta_states[0, 0] += 1
        assert dev.p_ta_states.contents.value == dev.ta_states[0, 0]
        dev.ta_states[0, 0] -= 1


class TestSetThreads:
    @pytest.mark.parametrize("n", [1, 2, 4, 0, -3])
    def test_accepts_any_count(self, dev, n):
        dev.set_threads(n)

    def test_resolved_thread_count_is_applied_at_init(self):
        assert make_device(device="cpu:2").device_config._n_threads in (1, 2)


class TestPackClauses:
    def test_all_excluded_gives_empty_clauses(self, dev):
        dev.pack_clauses(force_repack=True)
        pc = dev.get_packed_clauses()
        assert np.all(pc.clause_density == 0)
        assert np.all(pc.n_bounded_feats == 0)

    def test_marks_every_clause_synced(self, dev):
        dev.packed_clauses.is_clause_synced.fill(0)
        dev.pack_clauses()
        assert np.all(dev.packed_clauses.is_clause_synced == 1)

    def test_synced_clauses_are_skipped(self):
        """A clause edited without clearing its flag keeps the stale packing."""
        d = make_device()
        d.pack_clauses(force_repack=True)
        d.ta_states[0, d.config._n_position_feats + 3] = d.config._include_state

        d.pack_clauses()
        assert d.get_packed_clauses().clause_density[0] == 0

    def test_force_repack_rebuilds_synced_clauses(self):
        d = make_device()
        d.pack_clauses(force_repack=True)
        d.ta_states[0, d.config._n_position_feats + 3] = d.config._include_state

        d.pack_clauses(force_repack=True)
        assert d.get_packed_clauses().clause_density[0] == 1

    def test_an_included_literal_becomes_a_lower_bound(self):
        """Thermometer bit b included means value > min + b, so the bound starts at b + 1."""
        d = make_device()
        cfg = d.config
        d.ta_states[0, cfg._n_position_feats + 3] = cfg._include_state
        d.pack_clauses(force_repack=True)

        pc = d.get_packed_clauses()
        assert pc.clause_density[0] == 1
        assert pc.n_bounded_feats[0] == 1
        assert pc.clause_feat_bounds[0][0].tolist() == [4, 7]

    def test_a_contradiction_is_marked_invalid(self):
        """Requiring value > 5 and value <= 2 at once can never fire."""
        d = make_device()
        cfg = d.config
        half = cfg._n_literals // 2
        d.ta_states[0, cfg._n_position_feats + 5] = cfg._include_state
        d.ta_states[0, cfg._n_position_feats + 2 + half] = cfg._include_state
        d.pack_clauses(force_repack=True)

        assert d.get_packed_clauses().clause_density[0] == -1

    def test_other_clauses_are_untouched(self):
        d = make_device()
        cfg = d.config
        d.ta_states[0, cfg._n_position_feats + 3] = cfg._include_state
        d.pack_clauses(force_repack=True)

        pc = d.get_packed_clauses()
        assert pc.clause_density[0] == 1
        assert np.all(pc.clause_density[1:] == 0)

