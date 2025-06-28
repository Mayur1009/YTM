import numpy as np
from .base import BaseTM


class RegressionTM(BaseTM):
    def __init__(
        self,
        number_of_clauses,
        T,
        s,
        dim: tuple[int, int, int],
        patch_dim: tuple[int, int] | None = None,
        max_included_literals=None,
        number_of_ta_states=256,
        append_negated=True,
        init_neg_weights=False,  # Regression does not have negative polarity
        negative_polarity=False,  # Regression does not have negative polarity
        seed: int | None = None,
        block_size: int = 128,
    ):
        super().__init__(
            number_of_clauses,
            T,
            s,
            dim=dim,
            n_classes=1,
            patch_dim=patch_dim,
            max_included_literals=max_included_literals,
            number_of_ta_states=number_of_ta_states,
            append_negated=append_negated,
            init_neg_weights=init_neg_weights,
            negative_polarity=negative_polarity,
            seed=seed,
            block_size=block_size,
        )

    def fit(self, X, Y, is_X_encoded=False):
        X = X.reshape(X.shape[0], X.shape[1], 1)

        self.max_y = np.max(Y)
        self.min_y = np.min(Y)

        encoded_Y = ((Y - self.min_y) / (self.max_y - self.min_y) * self.T).astype(np.int32)
        encoded_X = self.encode(X) if not is_X_encoded else X
        self._fit_batch(encoded_X, encoded_Y)
        return

    def predict(self, X, is_X_encoded=False):
        encoded_X = self.encode(X) if not is_X_encoded else X
        class_sums = self._score_batch(encoded_X)
        preds = 1.0 * (class_sums[0, :]) * (self.max_y - self.min_y) / (self.T) + self.min_y
        return preds
