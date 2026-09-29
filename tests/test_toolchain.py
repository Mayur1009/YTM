"""Behaviour of the CPU toolchain: flag resolution, the two caches, and how each fails."""

import ctypes
import json
import os
import pathlib

import pytest

from ytm._core.backends import toolchain as tc
from ytm._core.backends.toolchain import Toolchain, compiler_cmd

CODE = "int answer(void) { return 42; }"


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("YTM_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("YTM_NO_CACHE", raising=False)
    monkeypatch.delenv("YTM_CACHE_SRC", raising=False)
    return tmp_path


def _answer(lib) -> int:
    lib.answer.restype = ctypes.c_int
    return lib.answer()


class TestFlags:
    def test_openmp_on_resolves_openmp_flags(self, cache):
        t = Toolchain(openmp=True)
        assert t.is_openmp_working
        assert any("openmp" in f for f in t.cflags)

    def test_openmp_off_still_builds(self, cache):
        """The serial flags have to be probed too: on macOS the first base candidate cannot link."""
        t = Toolchain(openmp=False)
        assert not t.is_openmp_working
        assert not any("openmp" in f for f in t.cflags)
        assert _answer(t.compile(CODE)) == 42

    def test_user_cflags_come_last(self, cache, monkeypatch):
        monkeypatch.setenv("CFLAGS", "-DYTM_TEST_MARKER")
        t = Toolchain(openmp=False)
        assert "-DYTM_TEST_MARKER" in t.cflags
        assert t.cflags.index("-DYTM_TEST_MARKER") > t.cflags.index("-fPIC")

    def test_compiler_cmd_honours_cc(self, monkeypatch):
        monkeypatch.setenv("CC", "some-compiler -flag")
        assert compiler_cmd() == ["some-compiler", "-flag"]

    def test_compiler_cmd_falls_back_to_path(self, monkeypatch):
        monkeypatch.delenv("CC", raising=False)
        assert compiler_cmd()[0] in ("clang", "gcc", "cc")


class TestCompile:
    def test_compiles_and_runs(self, cache):
        assert _answer(Toolchain(openmp=False).compile(CODE)) == 42

    def test_second_compile_does_not_invoke_the_compiler(self, cache, monkeypatch):
        t = Toolchain(openmp=False)
        t.compile(CODE)
        monkeypatch.setattr(t, "_run", lambda cmd: pytest.fail(f"recompiled: {cmd}"))
        assert _answer(t.compile(CODE)) == 42

    def test_different_source_is_a_different_artifact(self, cache):
        t = Toolchain(openmp=False)
        t.compile(CODE)
        t.compile("int answer(void) { return 7; }")
        assert len(list(cache.glob("*.so"))) == 2

    def test_works_without_a_cache(self, tmp_path, monkeypatch):
        monkeypatch.setenv("YTM_NO_CACHE", "1")
        t = Toolchain(openmp=False)
        assert t.cachedir is None
        assert _answer(t.compile(CODE)) == 42

    def test_save_src_writes_on_a_cache_hit(self, cache, monkeypatch):
        monkeypatch.setenv("YTM_CACHE_SRC", "1")
        t = Toolchain(openmp=False)
        t.compile(CODE)
        for f in cache.glob("*.c"):
            f.unlink()
        t.compile(CODE)  # served from cache, must still drop the source
        assert [f.read_text() for f in cache.glob("*.c")] == [CODE]

    def test_no_staging_files_left_behind(self, cache):
        Toolchain(openmp=False).compile(CODE)
        assert not list(cache.glob(f"*.{os.getpid()}"))


class TestDecisionCache:
    def test_cold_writes_then_warm_reads_without_probing(self, cache, monkeypatch):
        Toolchain(openmp=False)
        assert list(cache.glob("toolchain-*.json"))

        monkeypatch.setattr(Toolchain, "_resolve_flags", lambda *a: pytest.fail("re-probed on a cache hit"))
        assert Toolchain(openmp=False).cflags

    def test_key_tracks_cflags(self, cache, monkeypatch):
        before = Toolchain(openmp=False)._cache_key()
        monkeypatch.setenv("CFLAGS", "-DSOMETHING")
        assert Toolchain(openmp=False)._cache_key() != before

    @pytest.mark.parametrize("content", ["{ not json", json.dumps({"cflags": []}), json.dumps([])])
    def test_unusable_decision_is_re_resolved(self, cache, content):
        path = Toolchain(openmp=False)._cache_path()
        path.write_text(content)
        assert _answer(Toolchain(openmp=False).compile(CODE)) == 42

    def test_decision_naming_a_dead_runtime_is_re_resolved(self, cache):
        t = Toolchain(openmp=False)
        stored = json.loads(t._cache_path().read_text())
        stored["omp_runtime"] = "/nonexistent/libomp.so"
        t._cache_path().write_text(json.dumps(stored))

        recovered = Toolchain(openmp=False)
        assert recovered.omp_runtime is None
        assert _answer(recovered.compile(CODE)) == 42
        assert json.loads(t._cache_path().read_text())["omp_runtime"] is None


class TestArtifactCache:
    def test_corrupt_artifact_is_rebuilt(self, cache):
        """Planted before anything loads it: overwriting a mapped .so is a bus error, not a test."""
        t = Toolchain(openmp=False)
        planted = cache / f"{t._artifact_key(CODE)}.so"
        planted.write_bytes(b"not a shared library")
        assert _answer(t.compile(CODE)) == 42

    def test_unwritable_cache_dir_degrades(self, tmp_path, monkeypatch):
        blocked = tmp_path / "blocked"
        blocked.mkdir()
        blocked.chmod(0o500)
        monkeypatch.setenv("YTM_CACHE_DIR", str(blocked / "inner"))
        try:
            t = Toolchain(openmp=False)
            assert t.cachedir is None
            assert _answer(t.compile(CODE)) == 42
        finally:
            blocked.chmod(0o700)


class TestProbe:
    def test_probe_exercises_what_the_kernels_use(self):
        """sklearn ships an older libomp that satisfies a simple probe but not __kmpc_dispatch_*."""
        assert "schedule(dynamic)" in tc._OMP_PROBE
        assert "reduction(+ : acc[ : NC])" in tc._OMP_PROBE
