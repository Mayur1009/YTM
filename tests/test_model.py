import copy
import ctypes
import pickle

import numpy as np
import pytest

from .conftest import DEVICES
from .support import discrete, guided, set_clauses, set_weights


def _data(n=120, f=16, classes=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 2, size=(n, f), dtype=np.int32)
    Y = ((X[:, : f // 2].sum(1) > X[:, f // 2 :].sum(1)).astype(int) + (X[:, 0] == 1)) % classes
    return X, Y


def _trained(make, device, epochs=4):
    X, Y = _data()
    tm = make(device)
    for _ in range(epochs):
        tm.fit(X, Y)
    fresh = make(device)
    # discrete learns through TA states; guided may move only weights or only states, so either counts as "training did something"
    assert not np.array_equal(tm.get_ta_states(), fresh.get_ta_states()) or not np.array_equal(tm.get_weights(), fresh.get_weights())
    return tm, X


def _discrete(device):
    return discrete("multi", device, n_clauses=32, T=30.0, s=5.0, dim=(4, 4, 1), n_classes=3, feat_maxs=1)


def _guided(device):
    return guided("multi", device, n_clauses=32, s=5.0, dim=(4, 4, 1), n_classes=3, feat_maxs=1)


@pytest.mark.parametrize("make", [_discrete, _guided], ids=["discrete", "guided"])
@pytest.mark.parametrize("clone", [pickle, copy], ids=["pickle", "deepcopy"])
def test_predictions_survive_a_round_trip(device, make, clone):
    """A trained model, not a fresh one: an untrained one hides anything that confuses initial arrays with loaded ones."""
    tm, X = _trained(make, device)
    back = pickle.loads(pickle.dumps(tm)) if clone is pickle else copy.deepcopy(tm)
    assert np.array_equal(back.predict(X)[0], tm.predict(X)[0])


def test_the_c_side_sees_the_loaded_arrays():
    """Comparing python arrays is not enough: `load_state_dict` once left the ctypes pointers on the old buffers."""
    tm, _ = _trained(_discrete, "cpu:1")
    back = pickle.loads(pickle.dumps(tm))
    for name in ("ta_states", "clause_weights", "patch_weights"):
        arr, ptr = getattr(back.dev, name), getattr(back.dev, f"p_{name}")
        assert arr.ctypes.data == ctypes.cast(ptr, ctypes.c_void_p).value, name


def test_predictions_survive_a_device_move():
    """`to("cpu:2")` rebuilds the device and loads the state; predictions must not change."""
    tm, X = _trained(_discrete, "cpu:1")
    before = tm.predict(X)[0]
    moved = pickle.loads(pickle.dumps(tm))
    moved.to("cpu:2")
    assert moved.device_config.device == "cpu:2"
    assert np.array_equal(moved.predict(X)[0], before)


@pytest.mark.skipif("cuda" not in DEVICES, reason="no usable cuda device")
def test_cpu_and_cuda_agree_on_the_same_state():
    """Same TA states and weights must give the same votes on both backends (float32 reduction noise only)."""
    tm, X = _trained(_discrete, "cpu:1")
    on_gpu = pickle.loads(pickle.dumps(tm))
    on_gpu.to("cuda")
    assert np.allclose(on_gpu.score(X), tm.score(X), atol=1e-4)
    assert np.array_equal(on_gpu.predict(X)[0], tm.predict(X)[0])


def test_a_reloaded_model_continues_its_rng_instead_of_replaying():
    """Both generators are stored as `bit_generator.state`; reseeding would replay shuffles and drop masks."""
    tm, _ = _trained(_discrete, "cpu:1")
    fresh = _discrete("cpu:1")
    assert tm._rng.bit_generator.state != fresh._rng.bit_generator.state  # training advanced both generators
    assert tm.dev._rng.bit_generator.state != fresh.dev._rng.bit_generator.state
    a, b = pickle.loads(pickle.dumps(tm)), pickle.loads(pickle.dumps(tm))
    assert a._rng.bit_generator.state == tm._rng.bit_generator.state
    assert a.dev._rng.bit_generator.state == tm.dev._rng.bit_generator.state
    assert np.array_equal(a._rng.random(5), b._rng.random(5))


def test_loading_state_invalidates_the_packed_clauses(device):
    """Packed clauses are cached; `load_state_dict` must mark them stale or predictions keep using the old clause."""
    tm = discrete("multi", device)
    set_weights(tm, np.array([[3, 0, 0, 0], [0, 0, 0, 0]]))
    set_clauses(tm, {0: [0]})
    X = np.array([[1, 0, 0, 0]])
    assert tm.score(X)[0, 0] == 3  # packs the clause "x0"
    state = tm.dev.get_state_dict()
    state["ta_states"] = state["ta_states"].copy()
    state["ta_states"][0, :] = tm.config._include_state - 1
    state["ta_states"][0, 1] = tm.config._include_state  # now the clause is "x1"
    tm.dev.load_state_dict(state)
    assert tm.score(X)[0, 0] == 0  # no force_repack: the load itself must have invalidated the pack


def test_set_ta_states_helper_invalidates_the_pack(device):
    """The test helper's own contract: hand-set states must show up in the next score without `force_repack`."""
    tm = discrete("multi", device)
    set_weights(tm, np.array([[3, 0, 0, 0], [0, 0, 0, 0]]))
    set_clauses(tm, {0: [0]})
    X = np.array([[0, 0, 0, 0]])
    assert tm.score(X)[0, 0] == 0
    set_clauses(tm, {})
    assert tm.score(X)[0, 0] == 3
