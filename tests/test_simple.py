import numpy as np
import pytest

from ytm import simple, tm


def _copy_state(src, dst) -> None:
    dst.states[:] = np.asarray(src.get_ta_states()).reshape(dst.states.shape)
    dst.weights[:] = np.asarray(src.get_weights()).reshape(dst.weights.shape)


def _xor(n, n_features=12, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 2, size=(n, n_features), dtype=np.uint32)
    return X, X[:, 0] ^ X[:, 1]


@pytest.mark.parametrize("patch_dim", [None, (2, 2)], ids=["flat", "conv"])
@pytest.mark.parametrize("coalesced", [True, False], ids=["coalesced", "banks"])
@pytest.mark.parametrize(
    "flags",
    [
        {},
        {"negated_literals": False},
        {"position_literals": False},
        {"negative_clauses": False},
    ],
    ids=["default", "no_negated", "no_position", "no_negative"],
)
def test_score_matches_ytm_with_the_same_state(device, patch_dim, coalesced, flags):
    kw = {"patch_dim": patch_dim, "coalesced": coalesced, **flags}
    ref = tm.MultiClassTM(30, 15, 3.9, (4, 4, 1), 3, ta_init="random", seed=3, device=device, **kw)
    sim = simple.MultiClassTM(30, 15, 3.9, (4, 4), 3, seed=3, **kw)
    _copy_state(ref, sim)

    X = np.random.default_rng(1).integers(0, 2, size=(200, 4, 4), dtype=np.uint32)
    np.testing.assert_array_equal(sim.score(X), ref.score(X.reshape(-1, 4, 4, 1)))


def test_regression_predict_matches_ytm_with_the_same_state(device):
    ref = tm.RegressionTM(40, 100, 3.0, 16, (-0.5, 4.0), ta_init="random", seed=3, device=device)
    sim = simple.RegressionTM(40, 100, 3.0, 16, (-0.5, 4.0), seed=3)
    _copy_state(ref, sim)

    X = np.random.default_rng(1).integers(0, 2, size=(200, 16), dtype=np.uint32)
    sim_preds, sim_cs = sim.predict(X)
    ref_preds, ref_cs = ref.predict(X)
    np.testing.assert_array_equal(sim_cs, ref_cs)
    np.testing.assert_allclose(sim_preds, ref_preds, rtol=1e-6)


def test_empty_clauses_output_one():
    sim = simple.MultiClassTM(8, 15, 3.9, 6, 2, seed=0)
    X = np.random.default_rng(0).integers(0, 2, size=(10, 6), dtype=np.uint32)
    assert sim.clause_outputs(X).all()


def _one_clause(cls=simple.BinaryTM, **kw):
    sim = cls(2, 10, 1.0, 2, boost_tp_inc=True, boost_tp_dec=True, seed=0, **kw)
    sim.states[:] = sim.include_state - 1
    sim.states[0, 0, 0] = sim.include_state
    sim.weights[:] = 1.0
    x = sim._encode_X(np.array([[1, 0]]))[0, 0]
    co = np.array([True, False])
    return sim, np.stack([x, x]), co


def test_type_ia_moves_the_weight_away_from_zero_and_includes_true_literals():
    sim, X, co = _one_clause()
    sim.feedback(0, X, co, np.zeros(2, dtype=bool), 1, 1.0)
    inc = sim.include_state
    assert sim.weights[0, 0] == 2.0
    np.testing.assert_array_equal(sim.states[0, 0], [inc + 1, inc - 2, inc - 2, inc])


def test_clause_at_the_literal_budget_gets_type_ib_instead_of_type_ia():
    sim, X, co = _one_clause(max_includes=1)
    sim.feedback(0, X, co, np.zeros(2, dtype=bool), 1, 1.0)
    inc = sim.include_state
    assert sim.weights[0, 0] == 1.0
    np.testing.assert_array_equal(sim.states[0, 0], [inc - 1, inc - 2, inc - 2, inc - 2])


@pytest.mark.parametrize(
    "cls, kw, expected",
    [
        (simple.MultiClassTM, {"n_classes": 1}, -1.0),
        (simple.BinaryTM, {}, 1.0),
        (simple.MultiClassTM, {"n_classes": 1, "negative_clauses": False}, 1.0),
    ],
    ids=["polarity_flips", "binary_keeps_polarity", "no_negative_clauses"],
)
def test_type_ii_moves_the_weight_towards_zero(cls, kw, expected):
    sim, X, co = _one_clause(cls, **kw)
    sim.feedback(0, X, co, np.zeros(2, dtype=bool), -1, 1.0)
    inc = sim.include_state
    assert sim.weights[0, 0] == expected
    np.testing.assert_array_equal(sim.states[0, 0], [inc, inc, inc, inc - 1])


def test_dropped_clauses_get_no_feedback():
    X, Y = _xor(200)
    sim = simple.MultiClassTM(10, 15, 3.9, 12, 2, seed=0)
    states, weights = sim.states.copy(), sim.weights.copy()
    sim.fit(X, Y, clause_drop_p=1.0)
    np.testing.assert_array_equal(sim.states, states)
    np.testing.assert_array_equal(sim.weights, weights)


@pytest.mark.parametrize("cls", [simple.MultiClassTM, simple.BinaryTM], ids=["multiclass", "binary"])
def test_learns_xor(cls):
    X, Y = _xor(2000)
    X_test, Y_test = _xor(1000, seed=1)
    sim = cls(20, 15, 3.9, 12, **({"n_classes": 2} if cls is simple.MultiClassTM else {}), seed=1)
    for _ in range(10):
        sim.fit(X, Y)
    assert (sim.predict(X_test)[0] == Y_test).mean() >= 0.95


def test_multioutput_learns_xor_and_and():
    X, y = _xor(2000)
    X_test, y_test = _xor(1000, seed=1)
    Y, Y_test = np.stack([y, X[:, 2] & X[:, 3]], 1), np.stack([y_test, X_test[:, 2] & X_test[:, 3]], 1)
    sim = simple.MultiOutputTM(20, 15, 3.9, 12, 2, seed=1)
    for _ in range(10):
        sim.fit(X, Y)
    assert ((sim.predict(X_test)[0] == Y_test).mean(axis=0) >= 0.95).all()


def test_regression_beats_the_mean():
    rng = np.random.default_rng(0)
    X = rng.integers(0, 2, size=(3000, 16), dtype=np.uint32)
    y = 2 * X[:, 0] * X[:, 1] + X[:, 2] + 0.5 * (X[:, 3] ^ X[:, 4])
    sim = simple.RegressionTM(100, 500, 3.0, 16, (0.0, 3.5), seed=0)
    for _ in range(10):
        sim.fit(X, y)
    mae = np.abs(sim.predict(X)[0] - y).mean()
    assert mae < 0.3 * np.abs(y - y.mean()).mean()
