import numpy as np


class BaseTM:
    """Weighted Tsetlin Machine in pure numpy.

    Each clause is a team of Tsetlin Automata (TA), one TA per literal. A TA with ``n_states`` states
    includes its literal when its state is ``>= include_state``, else it excludes it. A clause is the
    AND of its included literals, so an empty clause always outputs 1. Each clause votes for or
    against each class through a signed integer weight, the sign being its polarity. The weighted
    votes of the active clauses give the class sum.

    Learning follows the TM feedback loop, one sample at a time:

    - Type I feedback is given to clauses whose polarity agrees with the target. Type Ia, for
      clauses that output 1, memorizes the input: true literals are reinforced towards include, and
      the clause weight moves one step away from 0. Type Ib, for clauses that output 0, forgets:
      every literal moves towards exclude with probability ``1/s``.
    - Type II feedback is given to clauses that output 1 but whose polarity disagrees with the
      target (false positives). Excluded false literals move towards include, so the clause stops
      matching the input, and the clause weight moves one step towards 0.
    - Each clause is selected for feedback with probability ``|prob|``, where ``prob`` is the
      distance of the clipped class sum from its target ``T_max`` or ``T_min``, scaled to [-1, 1].

    Parameters
    ----------
    n_clauses : int
        Clauses per clause bank.

    T : float or tuple of (float, float)
        Vote threshold. A float ``T`` gives targets ``[-T, T]``. A tuple gives ``[T_min, T_max]``.

    s : float
        Specificity, at least 1.0.

    dim : int or tuple of int
        Input shape: ``D``, ``(H, W)`` or ``(H, W, C)``. Inputs are binary.

    n_classes : int
        Number of classes.

    q : float, default=1.0
        Negative class sampling rate. The target class always gets feedback, each other class gets
        feedback with probability ``q / (n_classes - 1)``.

    coalesced : bool, default=True
        One clause bank shared by all classes with per-class weights (Coalesced TM). False gives each
        class its own clause bank.

    patch_dim : tuple of (int, int), default=None
        Convolution window ``(h, w)`` (Convolutional TM). None uses the whole input as one patch.

    negative_clauses : bool, default=True
        Start half the clauses with negative polarity (voting against a class). False keeps every
        weight at 1 or above, so clauses only vote for a class.

    negated_literals : bool, default=True
        Append the negation of each literal. Doubles the literal count.

    position_literals : bool, default=True
        Add thermometer-coded patch position literals. Only used when there is more than one patch.

    max_includes : int, default=None
        Literal budget of a clause. None, <= 0 or more than the literal count means no limit. A
        clause at the budget gets Type Ib instead of Type Ia feedback.

    n_states : int, default=256
        TA states per literal.

    ta_init : "middle" or int, default="middle"
        Initial TA state. ``"middle"`` is just below include.

    boost_tp_inc : bool, default=True
        Type Ia increments for true literals always happen, instead of with probability ``(s-1)/s``.

    boost_tp_dec : bool, default=False
        Type Ia decrements for false literals always happen, instead of with probability ``1/s``.

    allow_polarity_change : bool, default=True
        Let Type II feedback push a weight across 0, flipping the clause polarity.

    seed : int, default=None
        RNG seed. None picks a random seed.
    """

    def __init__(
        self,
        n_clauses: int,
        T: float | tuple[float, float],
        s: float,
        dim: int | tuple[int, ...],
        n_classes: int,
        q: float = 1.0,
        coalesced: bool = True,
        patch_dim: tuple[int, int] | None = None,
        negative_clauses: bool = True,
        negated_literals: bool = True,
        position_literals: bool = True,
        max_includes: int | None = None,
        n_states: int = 256,
        ta_init: str | int = "middle",
        boost_tp_inc: bool = True,
        boost_tp_dec: bool = False,
        allow_polarity_change: bool = True,
        seed: int | None = None,
    ):
        self.n_clauses = n_clauses
        self.T_min, self.T_max = (float(T[0]), float(T[1])) if isinstance(T, (tuple, list)) else (-float(T), float(T))
        self.s = s
        self.q = q
        self.n_classes = n_classes
        self.coalesced = coalesced
        self.negative_clauses = negative_clauses
        self.negated_literals = negated_literals
        self.n_states = n_states
        self.boost_tp_inc = boost_tp_inc
        self.boost_tp_dec = boost_tp_dec
        self.allow_polarity_change = allow_polarity_change
        self.rng = np.random.default_rng(seed)

        self.include_state = n_states // 2  # TA state >= include_state means the literal is included.
        self.max_state = n_states - 1

        # Input and patch geometry
        dim = (dim,) if isinstance(dim, int) else tuple(dim)
        self.dim = dim + (1,) * (3 - len(dim))  # pad dim with 1 so that it always is 3-tuple.
        H, W, C = self.dim
        self.patch_dim = (H, W) if patch_dim is None else tuple(patch_dim)
        h, w = self.patch_dim
        self.n_patches_y, self.n_patches_x = H - h + 1, W - w + 1
        self.n_patches = self.n_patches_y * self.n_patches_x

        # Literals of a patch: thermometer coded patch position (only with more than one patch), the
        # patch pixels, and optionally the negation of all of them.
        use_position = position_literals and self.n_patches > 1
        self.n_position_feats = (self.n_patches_y - 1) + (self.n_patches_x - 1) if use_position else 0
        self.n_features = self.n_position_feats + h * w * C
        self.n_literals = 2 * self.n_features if negated_literals else self.n_features

        if max_includes is None or max_includes <= 0 or max_includes > self.n_literals:
            max_includes = self.n_literals
        self.max_includes = max_includes

        self.n_banks = 1 if coalesced else n_classes
        self.states = np.empty((self.n_banks, n_clauses, self.n_literals), dtype=np.uint32)
        self.weights = np.empty((n_classes, n_clauses), dtype=np.float32)
        self._reset_states(ta_init)
        self._reset_weights()

    def _reset_states(self, ta_init: str | int) -> None:
        init = self.include_state - 1 if ta_init == "middle" else int(ta_init)
        assert 0 <= init <= self.max_state, f"ta_init must be in [0, {self.max_state}], got {init}"
        self.states.fill(init)

    def _reset_weights(self) -> None:
        # Half the clauses start with negative polarity. Coalesced shares clauses, so each class gets a
        # random assignment of polarities to clauses.
        pol = np.ones(self.n_clauses, dtype=np.float32)
        if self.negative_clauses:
            pol[self.n_clauses // 2 :] = -1.0
        for c in range(self.n_classes):
            self.weights[c] = self.rng.permutation(pol) if self.coalesced else pol

    def _bank(self, class_id: int) -> int:
        """Helper to get clause bank id."""
        return 0 if self.coalesced else class_id

    def _encode_X(self, X: np.ndarray) -> np.ndarray:
        """Turn inputs into the literals of every patch, shape (N, n_patches, n_literals), bool."""
        H, W, C = self.dim
        h, w = self.patch_dim
        X = np.asarray(X).astype(bool).reshape(-1, H, W, C)

        # (N, ny, nx, C, h, w) -> (N, n_patches, h * w * C), each patch in row-major (h, w, C) order
        windows = np.lib.stride_tricks.sliding_window_view(X, (h, w), axis=(1, 2))
        patches = windows.transpose(0, 1, 2, 4, 5, 3).reshape(X.shape[0], self.n_patches, h * w * C)

        if self.n_position_feats > 0:
            # Thermometer code of the patch position: bit k is 1 when the coordinate is > k.
            pos_y = np.arange(self.n_patches_y)[:, None] > np.arange(self.n_patches_y - 1)[None, :]
            pos_x = np.arange(self.n_patches_x)[:, None] > np.arange(self.n_patches_x - 1)[None, :]
            py, px = np.divmod(np.arange(self.n_patches), self.n_patches_x)
            pos = np.hstack((pos_y[py], pos_x[px]))
            patches = np.concatenate((np.broadcast_to(pos, (X.shape[0], *pos.shape)), patches), axis=2)

        return np.concatenate((patches, ~patches), axis=2) if self.negated_literals else patches

    def _eval_clauses(self, literals: np.ndarray, bank: int) -> np.ndarray:
        """Match each clause of ``bank`` with each patch. Returns shape (..., n_patches, n_clauses), bool.

        A clause matches a patch when none of its included literals is false. Empty clauses match
        every patch.
        """
        included = (self.states[bank] >= self.include_state).astype(np.float32)
        violations = (~literals).astype(np.float32) @ included.T
        return violations == 0

    def calc_class_sums(self, clause_outputs: np.ndarray, clip: bool) -> np.ndarray:
        """Weighted vote of the active clauses per class. ``clause_outputs`` has shape (N, n_banks, n_clauses)."""
        cs = np.empty((clause_outputs.shape[0], self.n_classes), dtype=np.float32)
        for c in range(self.n_classes):
            cs[:, c] = clause_outputs[:, self._bank(c)].astype(np.float32) @ self.weights[c]
        return np.clip(cs, self.T_min, self.T_max) if clip else cs

    def t1a(self, bank: int, clause_filter: np.ndarray, X: np.ndarray) -> None:
        """Type Ia feedback: reinforce include for true literals, exclude for false literals."""
        states = self.states[bank, clause_filter]
        X = X[clause_filter]

        inc = X & (states < self.max_state)
        if not self.boost_tp_inc:
            inc &= self.rng.random(states.shape) <= (self.s - 1) / self.s

        dec = ~X & (states > 0)
        if not self.boost_tp_dec:
            dec &= self.rng.random(states.shape) <= 1 / self.s

        self.states[bank, clause_filter] = states + inc - dec

    def t1b(self, bank: int, clause_filter: np.ndarray) -> None:
        """Type Ib feedback: reinforce exclude for every literal with probability 1/s."""
        states = self.states[bank, clause_filter]
        dec = (states > 0) & (self.rng.random(states.shape) <= 1 / self.s)
        self.states[bank, clause_filter] = states - dec

    def t2(self, bank: int, clause_filter: np.ndarray, X: np.ndarray) -> None:
        """Type II feedback: penalize the exclusion of false literals, making the clause output 0 for the input."""
        states = self.states[bank, clause_filter]
        inc = ~X[clause_filter] & (states < self.include_state)
        self.states[bank, clause_filter] = states + inc

    def feedback(
        self,
        class_id: int,
        X: np.ndarray,
        co: np.ndarray,
        dropped: np.ndarray,
        target: int,
        prob: float,
    ) -> None:
        """Give feedback for one class.

        Parameters
        ----------
        class_id : int
            Class to give feedback for.

        X : ndarray of shape (n_clauses, n_literals), bool
            Literals of the patch each clause is contrasted against.

        co : ndarray of shape (n_clauses,), bool
            Clause outputs.

        dropped : ndarray of shape (n_clauses,), bool
            Clauses dropped for this epoch.

        target : int
            +1 to push the class sum up to ``T_max``, -1 to push it down to ``T_min``.

        prob : float
            Probability of selecting each clause for feedback.
        """
        bank = self._bank(class_id)
        weights = self.weights[class_id]
        sign = np.where(weights >= 0, 1.0, -1.0).astype(np.float32)

        selected = (self.rng.random(self.n_clauses) < prob) & ~dropped
        agrees = sign == target  # Clause polarity agrees with the target: Type I, else Type II.

        # Clauses at the literal budget cannot include more, so they get Type Ib instead of Type Ia.
        has_space = (self.states[bank] >= self.include_state).sum(axis=1) < self.max_includes

        filter1a = selected & agrees & co & has_space
        filter1b = selected & agrees & (~co | ~has_space)
        filter2 = selected & ~agrees & co

        if np.any(filter1a):
            self.t1a(bank, filter1a, X)
            weights[filter1a] += sign[filter1a]

        if np.any(filter1b):
            self.t1b(bank, filter1b)

        if np.any(filter2):
            self.t2(bank, filter2, X)
            new = weights[filter2] - sign[filter2]
            if not self.negative_clauses:
                new = np.maximum(new, 1.0)
            else:
                flip = np.abs(new) < 1.0
                new[flip] = -sign[filter2][flip] if self.allow_polarity_change else sign[filter2][flip]
            weights[filter2] = new

    def _fit_sample(self, literals: np.ndarray, enc_y: np.ndarray, drop_mask: np.ndarray) -> None:
        co = np.empty((self.n_banks, self.n_clauses), dtype=bool)
        X = np.empty((self.n_banks, self.n_clauses, self.n_literals), dtype=bool)
        for b in range(self.n_banks):
            match = self._eval_clauses(literals, b).T  # (n_clauses, n_patches)
            match[drop_mask[b]] = False
            co[b] = match.any(axis=1)

            # Each clause is contrasted against one patch, picked at random among the patches it matches.
            r = self.rng.random(match.shape)
            r[~match] = -1.0
            X[b] = literals[np.argmax(r, axis=1)]

        cs = self.calc_class_sums(co[None], clip=True)[0]
        for c in range(self.n_classes):
            # Classes not targeted at T_max are negative classes, each sampled with probability q / (n_classes - 1).
            if enc_y[c] != self.T_max and self.n_classes > 1 and self.rng.random() > self.q / (self.n_classes - 1):
                continue
            # Signed distance of the class sum from its target: the sign gives the direction, the
            # magnitude the probability of selecting a clause for feedback.
            prob = (enc_y[c] - cs[c]) / (self.T_max - self.T_min)
            if prob != 0:
                b = self._bank(c)
                self.feedback(c, X[b], co[b], drop_mask[b], 1 if prob > 0 else -1, abs(prob))

    def _to_targets(self, labels: np.ndarray) -> np.ndarray:
        """Labels 0 or 1 per class to class sum targets: ``T_max`` for 1, ``T_min`` for 0."""
        return np.where(labels == 1, self.T_max, self.T_min).astype(np.float32)

    def _encode_Y(self, Y: np.ndarray) -> np.ndarray:
        """Class sum target of each class, within ``[T_min, T_max]``, shape (N, n_classes)."""
        raise NotImplementedError

    def fit(self, X: np.ndarray, Y: np.ndarray, shuffle: bool = True, clause_drop_p: float = 0.0) -> None:
        """Train for one epoch.

        Parameters
        ----------
        X : array-like of bool, shape ``(N, *dim)`` or ``(N, prod(dim))``
            Binary inputs.

        Y : array-like
            Targets, see the subclass.

        shuffle : bool, default=True
            Visit the samples in a random order.

        clause_drop_p : float, default=0.0
            Probability of dropping each clause for this epoch (Drop Clause). Dropped clauses output 0
            and get no feedback.
        """
        assert X.shape[0] == Y.shape[0], "X and Y must have the same number of samples."
        Y = self._encode_Y(Y)
        drop_mask = self.rng.random((self.n_banks, self.n_clauses)) < clause_drop_p
        iota = self.rng.permutation(X.shape[0]) if shuffle else np.arange(X.shape[0])
        for i in iota:
            self._fit_sample(self._encode_X(X[i : i + 1])[0], Y[i], drop_mask)

    def clause_outputs(self, X: np.ndarray) -> np.ndarray:
        """Clause outputs, the OR over patches. Returns shape (N, n_banks, n_clauses), bool."""
        literals = self._encode_X(X)
        return np.stack([self._eval_clauses(literals, b).any(axis=1) for b in range(self.n_banks)], axis=1)

    def score(self, X: np.ndarray) -> np.ndarray:
        """Class sums for X, not clipped. Returns shape (N, n_classes), float32."""
        return self.calc_class_sums(self.clause_outputs(X), clip=False)
