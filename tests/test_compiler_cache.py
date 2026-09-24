import ctypes
import itertools
import multiprocessing
import os
import pathlib
import platform
import tempfile

import numpy as np
import pytest

from ytm._core.backends import _compiler_cache as cc
from ytm._core.device_config import DeviceConfig
from ytm.discrete import BinaryTM

CODE = "int f(void) { return 1; }\n"
FLAGS = ["-shared", "-fPIC", "-O3"]
SOURCE = "int answer(void) { return 42; }\n"


def real_toolchain():
    """The compiler and flags a real cpu device resolves, so probe sources actually build."""
    dev = DeviceConfig(device="cpu:1")
    assert dev._compiler is not None
    return dev._compiler, [*dev._compiler_flags, *dev._omp_flags]


@pytest.fixture
def cache(monkeypatch, tmp_path):
    """Redirect the cache to a temp dir without touching the environment."""
    monkeypatch.setattr(cc, "resolve_cache_dir", lambda env=None: tmp_path)
    yield tmp_path


def artifacts(directory: pathlib.Path) -> list[pathlib.Path]:
    return list(directory.glob(f"*{cc._SUFFIX}"))


def _build_in_child(directory: str, queue) -> None:
    """Load the probe library in a fresh process. Module-level so `spawn` can pickle it."""
    compiler, flags = real_toolchain()
    lib = cc.load_library(SOURCE, compiler, flags, env={"YTM_CACHE_DIR": directory})
    lib.answer.restype = ctypes.c_int
    queue.put(lib.answer())


class TestCacheKey:
    def test_is_a_sha256_hexdigest(self):
        key = cc.cache_key(CODE, "clang", FLAGS)
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)

    def test_is_stable_across_calls(self):
        assert cc.cache_key(CODE, "clang", FLAGS) == cc.cache_key(CODE, "clang", FLAGS)

    def test_source_change_changes_key(self):
        assert cc.cache_key(CODE + "\n", "clang", FLAGS) != cc.cache_key(CODE, "clang", FLAGS)

    def test_compiler_change_changes_key(self):
        assert cc.cache_key(CODE, "gcc", FLAGS) != cc.cache_key(CODE, "clang", FLAGS)

    def test_flag_change_changes_key(self):
        assert cc.cache_key(CODE, "clang", [*FLAGS, "-fopenmp"]) != cc.cache_key(CODE, "clang", FLAGS)

    def test_flag_order_changes_key(self):
        # A different argument order is a different compilation; it must not collide.
        assert cc.cache_key(CODE, "clang", ["-O3", "-fPIC"]) != cc.cache_key(CODE, "clang", ["-fPIC", "-O3"])

    def test_sequence_type_does_not_change_key(self):
        # Callers may pass a tuple; only the contents are meaningful.
        assert cc.cache_key(CODE, "clang", tuple(FLAGS)) == cc.cache_key(CODE, "clang", FLAGS)

    def test_separator_prevents_field_smearing(self):
        # Without a separator these two would concatenate to the same bytes.
        assert cc.cache_key("ab", "c", FLAGS) != cc.cache_key("a", "bc", FLAGS)

    def test_host_is_part_of_the_key(self, monkeypatch):
        # -march=native makes artifacts CPU-specific and the cache may sit on a shared
        # filesystem, so two hosts must never read each other's entries.
        base = cc.cache_key(CODE, "clang", FLAGS)
        monkeypatch.setattr(platform, "node", lambda: "some-other-host")
        assert cc.cache_key(CODE, "clang", FLAGS) != base

    def test_machine_is_part_of_the_key(self, monkeypatch):
        base = cc.cache_key(CODE, "clang", FLAGS)
        monkeypatch.setattr(platform, "machine", lambda: "s390x")
        assert cc.cache_key(CODE, "clang", FLAGS) != base


class TestCacheDir:
    """`cache_dir` takes the environment as an argument, so tests never mutate the real one."""

    def test_ytm_cache_dir_wins(self, tmp_path):
        env = {"YTM_CACHE_DIR": str(tmp_path / "chosen"), "XDG_CACHE_HOME": str(tmp_path / "xdg")}
        assert cc.resolve_cache_dir(env) == tmp_path / "chosen"

    def test_falls_back_to_xdg(self, tmp_path):
        assert cc.resolve_cache_dir({"XDG_CACHE_HOME": str(tmp_path / "xdg")}) == tmp_path / "xdg" / "ytm"

    def test_falls_back_to_home_cache(self):
        assert cc.resolve_cache_dir({}) == pathlib.Path.home() / ".cache" / "ytm"

    def test_does_not_create_the_directory(self, tmp_path):
        target = tmp_path / "not-yet"
        cc.resolve_cache_dir({"YTM_CACHE_DIR": str(target)})
        assert not target.exists()

    def test_defaults_to_the_real_environment(self):
        assert cc.resolve_cache_dir() == cc.resolve_cache_dir(dict(os.environ))


class TestLoadLibrary:
    def test_compiles_and_returns_a_working_library(self, cache):
        compiler, flags = real_toolchain()
        lib = cc.load_library(SOURCE, compiler, flags)
        lib.answer.restype = ctypes.c_int
        assert lib.answer() == 42

    def test_disk_hit_does_not_invoke_the_compiler(self, cache, monkeypatch):
        compiler, flags = real_toolchain()
        cc.load_library(SOURCE, compiler, flags)

        monkeypatch.setattr(cc, "run_compiler", lambda cmd: pytest.fail(f"recompiled: {cmd}"))
        lib = cc.load_library(SOURCE, compiler, flags)
        lib.answer.restype = ctypes.c_int
        assert lib.answer() == 42

    def test_leaves_one_artifact_and_no_source(self, cache):
        compiler, flags = real_toolchain()
        cc.load_library(SOURCE, compiler, flags)
        assert len(artifacts(cache)) == 1
        assert list(cache.glob("*.c")) == []

    def test_save_source_keeps_the_c_file(self, cache):
        compiler, flags = real_toolchain()
        cc.load_library(SOURCE, compiler, flags, env={"YTM_SAVE_SOURCE": "1"})
        assert len(list(cache.glob("*.c"))) == 1

    def test_no_cache_writes_nothing_to_the_cache_dir(self, cache):
        compiler, flags = real_toolchain()
        lib = cc.load_library(SOURCE, compiler, flags, env={"YTM_NO_CACHE": "1"})
        lib.answer.restype = ctypes.c_int
        assert lib.answer() == 42
        assert artifacts(cache) == []


class TestFailureModes:
    def test_compile_failure_raises_and_leaves_nothing_behind(self, cache):
        compiler, flags = real_toolchain()
        with pytest.raises(RuntimeError, match="Failed to compile"):
            cc.load_library("this is not c\n", compiler, flags)
        assert list(cache.iterdir()) == []

    def test_unusable_cache_dir_still_returns_a_library(self, monkeypatch, tmp_path):
        # Caching is an optimization; losing it must not cost the caller a model.
        blocked = tmp_path / "blocked"
        blocked.write_text("i am a file, not a directory")
        monkeypatch.setattr(cc, "resolve_cache_dir", lambda env=None: blocked)

        compiler, flags = real_toolchain()
        lib = cc.load_library(SOURCE, compiler, flags)
        lib.answer.restype = ctypes.c_int
        assert lib.answer() == 42

    def test_truncated_artifact_is_recompiled(self, cache):
        # Models what an interrupted earlier run leaves behind: a bad file at the right
        # key, found by a process that has never loaded it. (Corrupting a file this
        # process already mapped would instead give a SIGBUS, because dlopen hands back
        # its cached handle for that path rather than re-reading the file.)
        compiler, flags = real_toolchain()
        stale = cache / f"{cc.cache_key(SOURCE, compiler, flags)}{cc._SUFFIX}"
        stale.write_bytes(b"not an object file")

        lib = cc.load_library(SOURCE, compiler, flags)
        lib.answer.restype = ctypes.c_int
        assert lib.answer() == 42

    def test_concurrent_builders_do_not_corrupt_the_entry(self, cache):
        # Four processes race on one key. There is no lock: identical inputs give identical
        # bytes, so the loser of the rename wastes work but cannot install a wrong artifact.
        ctx = multiprocessing.get_context("spawn")
        queue = ctx.Queue()
        procs = [ctx.Process(target=_build_in_child, args=(str(cache), queue)) for _ in range(4)]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=120)

        assert sorted(queue.get(timeout=5) for _ in procs) == [42] * len(procs)
        assert len(artifacts(cache)) == 1

    def test_recovery_does_not_loop_forever(self, cache, monkeypatch):
        # If the freshly built artifact also fails to load, surface the error rather than retrying.
        compiler, flags = real_toolchain()
        monkeypatch.setattr(cc, "CDLL", lambda path: (_ for _ in ()).throw(OSError("bad image")))
        with pytest.raises(OSError, match="bad image"):
            cc.load_library(SOURCE, compiler, flags)


MODEL = {"n_clauses": 8, "dim": (3, 1, 1), "s": 2.0, "T": 4}


class TestCPUDeviceIntegration:
    def _xor(self):
        X = np.array(list(itertools.product((0, 1), repeat=3)), dtype=np.int32)
        return X, (X.sum(1) % 2).astype(np.uint32)

    def _train(self, seed):
        X, Y = self._xor()
        tm = BinaryTM(seed=seed, **MODEL)
        for _ in range(30):
            tm.fit(X, Y)
        preds, _ = tm.predict(X)
        return float(np.mean(preds == Y)), int(np.asarray(tm.get_ta_states()).sum())

    def test_model_from_a_warm_cache_matches_a_cold_one(self, cache):
        cold = self._train(seed=1)
        assert self._train(seed=1) == cold

    def test_second_construction_does_not_invoke_the_compiler(self, cache, monkeypatch):
        BinaryTM(seed=1, **MODEL)

        monkeypatch.setattr(cc, "run_compiler", lambda cmd: pytest.fail(f"recompiled: {cmd}"))
        BinaryTM(seed=2, **MODEL)  # the seed is not part of the generated source

    def test_construction_leaves_no_stray_files_in_tmp(self, cache):
        tmp = pathlib.Path(tempfile.gettempdir())
        before = set(tmp.glob(f"tmp*{cc._SUFFIX}"))
        BinaryTM(seed=3, **MODEL)
        assert set(tmp.glob(f"tmp*{cc._SUFFIX}")) == before
