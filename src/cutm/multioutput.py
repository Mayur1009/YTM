import numpy as np
from .base import BaseTM


class MultiOutputTM(BaseTM):
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
        coalesced: bool = True,
        seed: int | None = None,
        block_size: int = 128,
    ) -> None:
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
            coalesced=coalesced,
            seed=seed,
            block_size=block_size,
        )

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray[tuple[int, int], np.dtype[np.uint32]],
        is_X_encoded: bool = False,
        block_size: int | None = None,
    ) -> None:
        # Input validation
        assert Y.ndim == 2, f"Y must be 2D array (samples, outputs), got {Y.ndim}D"
        assert X.shape[0] == Y.shape[0], "X and Y must have the same number of samples"

        self.max_y = None
        self.min_y = None

        encoded_Y = np.where(Y == 1, self.T, -self.T).astype(np.int32)
        encoded_X = self.encode(X) if not is_X_encoded else X
        self._fit(encoded_X, encoded_Y, block_size=block_size)

    def score(
        self,
        X: np.ndarray,
        is_X_encoded: bool,
        block_size: int | None = None,
    ):
        encoded_X = X if is_X_encoded else self.encode(X)
        return self._score_batch(encoded_X, block_size=block_size)

    def predict(
        self,
        X: np.ndarray,
        is_X_encoded: bool = False,
        block_size: int | None = None,
    ):
        class_sums = self.score(X, is_X_encoded, block_size=block_size)
        preds = (class_sums >= 0).astype(np.uint32)
        return preds, class_sums
