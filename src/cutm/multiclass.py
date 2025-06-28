import numpy as np
from .base import BaseTM


class MultiClassTM(BaseTM):
    def __init__(
        self,
        number_of_clauses: int,
        T: int,
        s: float,
        dim: tuple[int, int, int],
        n_classes: int,
        q: float = 1.0,
        patch_dim: tuple[int, int] | None = None,
        max_included_literals: int | None = None,
        number_of_ta_states: int = 256,
        append_negated: bool = True,
        init_neg_weights: bool = True,
        negative_polarity: bool = True,
        encode_loc: bool = True,
        seed: int | None = None,
        block_size: int = 128,
    ):
        super().__init__(
            number_of_clauses,
            T,
            s,
            dim=dim,
            n_classes=n_classes,
            q=q,
            patch_dim=patch_dim,
            max_included_literals=max_included_literals,
            number_of_ta_states=number_of_ta_states,
            append_negated=append_negated,
            init_neg_weights=init_neg_weights,
            negative_polarity=negative_polarity,
            encode_loc=encode_loc,
            seed=seed,
            block_size=block_size,
        )

    def fit(
        self,
        X: np.ndarray[tuple[int, int], np.dtype[np.uint32]],
        Y: np.ndarray[tuple[int], np.dtype[np.uint32]],
        is_X_encoded: bool = False,
    ):
        assert Y.ndim == 1, "Y must be 1D array (samples,)"
        assert X.shape[0] == Y.shape[0], "X and Y must have the same number of samples."

        self.max_y = None
        self.min_y = None

        encoded_Y = np.empty((Y.shape[0], self.number_of_outputs), dtype=np.int32)
        for i in range(self.number_of_outputs):
            encoded_Y[:, i] = np.where(Y == i, self.T, -self.T)

        encoded_X = self.encode(X) if not is_X_encoded else X
        self._fit_batch(encoded_X, encoded_Y)

    def score(
        self,
        X: np.ndarray[tuple[int, int], np.dtype[np.uint32]],
        is_X_encoded,
    ):
        encoded_X = self.encode(X) if not is_X_encoded else X
        return self._score_batch(encoded_X)

    def predict(self, X: np.ndarray[tuple[int, int], np.dtype[np.uint32]], is_X_encoded: bool = False):
        class_sums = self.score(X, is_X_encoded)
        preds = np.argmax(class_sums, axis=1)
        return preds, class_sums
