import numpy as np
from tqdm import tqdm


class BaseTM:
    def __init__(self, n_clause, T, s, q, n_class, dim, n_ta_states=256, seed=None):
        self.n_clause = n_clause
        self.T = T
        self.s = s
        self.s_inv = 1 / s
        self.q = q
        self.n_class = n_class
        self.dim = dim
        self.n_ta_states = n_ta_states
        assert self.n_ta_states <= 256, "n_ta_states must be less than or equal to 256."

        self.dim = dim

        self.half_state = (self.n_ta_states // 2) - 1  # If ta_state > half_state, then the literal is included.
        self.max_state = self.n_ta_states - 1  # Maximum state value.

        self.n_features = self.dim[0] * self.dim[1] * self.dim[2]
        self.n_literals = 2 * self.n_features

        self.rng = np.random.default_rng(seed)

        self.states = np.empty((self.n_clause, self.n_literals), dtype=np.uint8)
        self.weights = np.empty((self.n_clause, self.n_class), dtype=np.float32)
        self._reset_states()
        self._reset_weights()

    def _reset_states(self):
        self.states.fill(self.half_state)

    def _reset_weights(self):
        self.weights = self.rng.integers(0, 2, size=(self.n_clause, self.n_class))
        self.weights = (2 * self.weights - 1).astype(np.float32)

    def _encode_X(
        self,
        X: np.ndarray[tuple[int, int], np.dtype[np.bool]],
    ) -> np.ndarray[tuple[int, int], np.dtype[np.bool]]:
        X = X.reshape(X.shape[0], -1)  # Flatten the input to a 2D array
        return np.hstack((X, ~X))

    def _eval_clauses_sample(
        self,
        encoded_X: np.ndarray[tuple[int], np.dtype[np.bool]],
    ) -> np.ndarray[tuple[int], np.dtype[np.bool]]:
        literals = self.states > self.half_state
        # co = literals @ encoded_X.T
        # return co.T == np.sum(literals, axis=1)
        return ~(literals @ ~encoded_X)

    def calc_class_sums(self, clause_outputs: np.ndarray, clip: bool) -> np.ndarray:
        cs = clause_outputs.astype(np.float32) @ self.weights
        return np.clip(cs, -self.T, self.T) if clip else cs

    def _eval_clauses_multisample(
        self, encoded_X: np.ndarray[tuple[int, int], np.dtype[np.bool]]
    ) -> np.ndarray[tuple[int], np.dtype[np.bool]]:
        literals = (self.states > self.half_state).astype(np.float32)
        n_includes = np.sum(literals, axis=1)
        x = encoded_X.astype(np.float32)
        co = literals @ x.T
        co = co.T == n_includes
        return co

    def t1a(
        self,
        clause_filter: np.ndarray[tuple[int], np.dtype[np.bool]],
        X: np.ndarray[tuple[int], np.dtype[np.bool]],
        inc: int,
        cl: int,
    ) -> None:
        self.weights[clause_filter, cl] += inc
        cpy = self.states[clause_filter]

        literal_filter = (cpy < self.max_state) & X
        self.states[clause_filter] += literal_filter.astype(np.uint8)

        s_filter = (self.rng.uniform(size=cpy.shape) <= self.s_inv) & ~X
        literal_filter = (cpy > 0) & s_filter
        self.states[clause_filter] -= literal_filter.astype(np.uint8)

    def t1b(self, clause_filter: np.ndarray[tuple[int], np.dtype[np.bool]]):
        cpy = self.states[clause_filter]
        literal_filter = (cpy > 0) & (self.rng.uniform(size=cpy.shape) <= self.s_inv)
        self.states[clause_filter] -= literal_filter.astype(np.uint8)

    def t2(
        self,
        clause_filter: np.ndarray[tuple[int], np.dtype[np.bool]],
        X: np.ndarray[tuple[int], np.dtype[np.bool]],
        dec: int,
        cl: int,
    ) -> None:
        self.weights[clause_filter, cl] -= dec
        literal_filter = ~X
        self.states[clause_filter] += literal_filter.astype(np.uint8)

    def feedback(
        self,
        X: np.ndarray[tuple[int], np.dtype[np.bool]],
        class_id: int,
        prob: float,
        co: np.ndarray[tuple[int], np.dtype[np.bool]],
        true_class: bool,
    ) -> None:
        if true_class:
            pos_pol = self.weights[:, class_id] >= 0
            neg_pol = self.weights[:, class_id] < 0
            inc = 1
        else:
            pos_pol = self.weights[:, class_id] < 0
            neg_pol = self.weights[:, class_id] >= 0
            inc = -1

        prob_filter = self.rng.uniform(size=co.shape) <= prob

        filter1a = pos_pol & co & prob_filter
        filter1b = pos_pol & ~co & prob_filter
        filter2 = neg_pol & co & prob_filter

        if np.any(filter1a):
            self.t1a(filter1a, X, inc, class_id)

        if np.any(filter1b):
            self.t1b(filter1b)

        if np.any(filter2):
            self.t2(filter2, X, -1 * inc, class_id)

    def _fit_sample(self, encoded_X: np.ndarray[tuple[int], np.dtype[np.bool]], y: int):
        co = self._eval_clauses_sample(encoded_X)
        cs = self.calc_class_sums(co, clip=True)

        prob = (self.T - cs[y]) / (2 * self.T)
        self.feedback(encoded_X, y, prob, co, True)
        other_class: int = self.rng.choice(np.delete(np.arange(self.n_class), y), size=1)[0]
        prob = (self.T + cs[other_class]) / (2 * self.T)

        self.feedback(encoded_X, other_class, prob, co, False)

    def fit(
        self,
        X: np.ndarray[tuple[int, int], np.dtype[np.bool]],
        y: np.ndarray[tuple[int], np.dtype[np.uint]],
        shuffle: bool = False,
    ) -> None:
        encoded_X = self._encode_X(X)
        iota = np.arange(X.shape[0])
        if shuffle:
            self.rng.shuffle(iota)
        for i in tqdm(iota, desc="Fitting", total=X.shape[0], leave=False, dynamic_ncols=True):
            self._fit_sample(encoded_X[i], y[i])

    def predict(self, X: np.ndarray[tuple[int, int], np.dtype[np.bool]], return_cs: bool = False):
        X = self._encode_X(X)
        clause_outputs = self._eval_clauses_multisample(X)

        class_sums = self.calc_class_sums(clause_outputs, False)
        preds = np.argmax(class_sums, axis=1)
        if return_cs:
            return preds, class_sums
        return preds
