import numpy as np
from scipy.sparse import csr_matrix
from .base import CommonTsetlinMachine


class MultiClassConvolutionalTsetlinMachine2D(CommonTsetlinMachine):
    """
    This class ...
    """

    def __init__(
        self,
        number_of_clauses,
        T,
        s,
        dim,
        patch_dim,
        q: float = 1.0,
        max_included_literals=None,
        number_of_ta_states=256,
        append_negated=True,
        encode_loc: bool = True,
        seed: int | None = None,
        block_size: int = 128,
    ):
        super().__init__(
            number_of_clauses,
            T,
            s,
            q=q,
            max_included_literals=max_included_literals,
            number_of_ta_states=number_of_ta_states,
            append_negated=append_negated,
            encode_loc=encode_loc,
            seed=seed,
            block_size=block_size,
        )
        self.dim = dim
        self.patch_dim = patch_dim
        self.negative_clauses = 1

    def fit(self, X, Y, epochs=100, incremental=False):
        if len(X.shape) == 3:
            print(f"Expecting X with 2D shape, got {X.shape}. Flattening the array...")
            X = X.reshape((X.shape[0], -1))
            print(f"New X.shape => {X.shape}")
        X = csr_matrix(X)

        self.number_of_outputs = int(np.max(Y) + 1)

        self.max_y = None
        self.min_y = None

        encoded_Y = np.empty((Y.shape[0], self.number_of_outputs), dtype=np.int32)
        for i in range(self.number_of_outputs):
            # encoded_Y[:, i] = np.where(Y == i, 1, 0)
            encoded_Y[:, i] = np.where(Y == i, self.T, -self.T)

        self._fit(X, encoded_Y, epochs=epochs, incremental=incremental)

    def score(self, X):
        X = csr_matrix(X)
        return self._score(X)

    def predict(self, X, return_class_sums=False):
        class_sums = self.score(X)
        preds = np.argmax(class_sums, axis=1)
        if return_class_sums:
            return preds, class_sums
        else:
            return preds


class MultiOutputConvolutionalTsetlinMachine2D(CommonTsetlinMachine):
    """
    This class ...
    """

    def __init__(
        self,
        number_of_clauses,
        T,
        s,
        dim,
        patch_dim,
        q: float = 1.0,
        max_included_literals=None,
        number_of_ta_states=256,
        append_negated=True,
        encode_loc: bool = True,
        seed: int | None = None,
        block_size: int = 128,
    ):
        super().__init__(
            number_of_clauses,
            T,
            s,
            q=q,
            max_included_literals=max_included_literals,
            number_of_ta_states=number_of_ta_states,
            append_negated=append_negated,
            encode_loc=encode_loc,
            seed=seed,
            block_size=block_size,
        )
        self.dim = dim
        self.patch_dim = patch_dim
        self.negative_clauses = 1

    def fit(self, X, Y, epochs=100, incremental=False):
        if len(X.shape) == 3:
            print(f"Expecting X with 2D shape, got {X.shape}. Flattening the array...")
            X = X.reshape((X.shape[0], -1))
            print(f"New X.shape => {X.shape}")
        X = csr_matrix(X)

        self.number_of_outputs = Y.shape[1]

        self.max_y = None
        self.min_y = None

        encoded_Y = np.where(Y == 1, self.T, -self.T).astype(np.int32)

        self._fit(X, encoded_Y, epochs=epochs, incremental=incremental)

    def score(self, X):
        X = csr_matrix(X)

        return self._score(X)

    def predict(self, X, return_class_sums=False):
        if len(X.shape) == 3:
            print(f"Expecting X with 2D shape, got {X.shape}. Flattening samples...")
            X = X.reshape((X.shape[0], -1))
            print(f"New X.shape => {X.shape}")
        class_sums = self.score(X)
        preds = (class_sums >= 0).astype(np.uint32)
        if return_class_sums:
            return preds, class_sums
        else:
            return preds


class MultiOutputTsetlinMachine(CommonTsetlinMachine):
    def __init__(
        self,
        number_of_clauses,
        T,
        s,
        q: float = 1.0,
        max_included_literals=None,
        number_of_ta_states=256,
        append_negated=True,
        seed: int | None = None,
        block_size: int = 128,
    ):
        super().__init__(
            number_of_clauses,
            T,
            s,
            q=q,
            max_included_literals=max_included_literals,
            number_of_ta_states=number_of_ta_states,
            append_negated=append_negated,
            seed=seed,
            block_size=block_size,
        )
        self.negative_clauses = 1

    def fit(self, X, Y, epochs=100, incremental=False):
        X = csr_matrix(X)

        self.number_of_outputs = Y.shape[1]

        self.dim = (X.shape[1], 1, 1)
        self.patch_dim = (X.shape[1], 1)

        self.max_y = None
        self.min_y = None

        encoded_Y = np.where(Y == 0, self.T, -self.T).astype(np.int32)
        self._fit(X, encoded_Y, epochs=epochs, incremental=incremental)

        return

    def score(self, X):
        X = csr_matrix(X)
        return self._score(X)

    def predict(self, X, return_class_sums=True):
        if len(X.shape) == 3:
            print(f"Expecting X with 2D shape, got {X.shape}. Flattening samples...")
            X = X.reshape((X.shape[0], -1))
            print(f"New X.shape => {X.shape}")
        class_sums = self.score(X)
        preds = (class_sums >= 0).astype(np.uint32)
        if return_class_sums:
            return preds, class_sums
        else:
            return preds


class MultiClassTsetlinMachine(CommonTsetlinMachine):
    def __init__(
        self,
        number_of_clauses,
        T,
        s,
        q: float = 1.0,
        max_included_literals=None,
        number_of_ta_states=256,
        append_negated=True,
        seed: int | None = None,
        block_size: int = 128,
    ):
        super().__init__(
            number_of_clauses,
            T,
            s,
            q=q,
            max_included_literals=max_included_literals,
            number_of_ta_states=number_of_ta_states,
            append_negated=append_negated,
            seed=seed,
            block_size=block_size,
        )
        self.negative_clauses = 1

    def fit(self, X, Y, epochs=100, incremental=False):
        X = csr_matrix(X)

        self.number_of_outputs = int(np.max(Y) + 1)

        self.dim = (X.shape[1], 1, 1)
        self.patch_dim = (X.shape[1], 1)

        self.max_y = None
        self.min_y = None

        encoded_Y = np.empty((Y.shape[0], self.number_of_outputs), dtype=np.int32)
        for i in range(self.number_of_outputs):
            encoded_Y[:, i] = np.where(Y == i, self.T, -self.T)

        self._fit(X, encoded_Y, epochs=epochs, incremental=incremental)

        return

    def score(self, X):
        X = csr_matrix(X)
        return self._score(X)

    def predict(self, X, return_class_sums=False):
        class_sums = self.score(X)
        preds = np.argmax(class_sums, axis=1)
        if return_class_sums:
            return preds, class_sums
        else:
            return preds


class TsetlinMachine(CommonTsetlinMachine):
    def __init__(
        self,
        number_of_clauses,
        T,
        s,
        q: float = 1.0,
        max_included_literals=None,
        number_of_ta_states=256,
        append_negated=True,
        seed: int | None = None,
        block_size: int = 128,
    ):
        super().__init__(
            number_of_clauses,
            T,
            s,
            q=q,
            max_included_literals=max_included_literals,
            number_of_ta_states=number_of_ta_states,
            append_negated=append_negated,
            seed=seed,
            block_size=block_size,
        )
        self.negative_clauses = 1

    def fit(self, X, Y, epochs=100, incremental=False):
        X = X.reshape(X.shape[0], X.shape[1], 1)

        self.number_of_outputs = 1
        self.patch_dim = (X.shape[1], 1, 1)

        self.max_y = None
        self.min_y = None

        encoded_Y = np.where(Y == 0, self.T, -self.T).astype(np.int32)

        self._fit(X, encoded_Y, epochs=epochs, incremental=incremental)

        return

    def score(self, X):
        X = X.reshape(X.shape[0], X.shape[1], 1)
        return self._score(X)[0, :]

    def predict(self, X, return_class_sums=False):
        class_sums = self.score(X)
        preds = int(class_sums >= 0)

        if return_class_sums:
            return preds, class_sums
        else:
            return preds


class RegressionTsetlinMachine(CommonTsetlinMachine):
    def __init__(
        self,
        number_of_clauses,
        T,
        s,
        max_included_literals=None,
        number_of_ta_states=256,
        append_negated=True,
        seed: int | None = None,
        block_size: int = 128,
    ):
        super().__init__(
            number_of_clauses,
            T,
            s,
            max_included_literals=max_included_literals,
            number_of_ta_states=number_of_ta_states,
            append_negated=append_negated,
            seed=seed,
            block_size=block_size,
        )
        self.negative_clauses = 0

    def fit(self, X, Y, epochs=100, incremental=False):
        X = X.reshape(X.shape[0], X.shape[1], 1)

        self.number_of_outputs = 1
        self.patch_dim = (X.shape[1], 1, 1)

        self.max_y = np.max(Y)
        self.min_y = np.min(Y)

        encoded_Y = ((Y - self.min_y) / (self.max_y - self.min_y) * self.T).astype(np.int32)

        self._fit(X, encoded_Y, epochs=epochs, incremental=incremental)

        return

    def predict(self, X, return_class_sums=False):
        X = X.reshape(X.shape[0], X.shape[1], 1)
        class_sums = self._score(X)
        preds = 1.0 * (class_sums[0, :]) * (self.max_y - self.min_y) / (self.T) + self.min_y

        if return_class_sums:
            return preds, class_sums
        else:
            return preds
