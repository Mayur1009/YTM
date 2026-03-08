import numpy as np
from typing import Literal
from dataclasses import asdict
from .args import TMArgs


class BaseTM:
    def __init__(
        self,
        n_clauses: int,
        T: float | int,
        s: float,
        dim: tuple[int, int, int],
        n_classes: int,
        **kwargs,
    ):
        self.args = TMArgs(n_clauses, T, s, dim, n_classes, **kwargs)
        self.n_clauses = self.args.n_clauses
        self.T = self.args.T
        self.s = self.args.s
        self.dim = self.args.dim
        self.rng = np.random.default_rng(self.args.seed)

        if self.args.device == "cpu":
            from .backends.cpu.cpu_backend import CPUDevice

            self.dev = CPUDevice(self.args)
        elif self.args.device == "cuda":
            from .backends.cuda.cuda_backend import CUDADevice

            self.dev = CUDADevice(self.args)
        else:
            raise ValueError(f"Unsupported device: {self.args.device}")

    def encode(self, X: np.ndarray, batch_size: int = -1) -> np.ndarray[tuple[int, int, int], np.dtype[np.uint32]]:
        if batch_size == -1:
            batch_size = X.shape[0]

        encoded_X = np.empty((X.shape[0], self.dev.n_patches, self.dev.n_literal_chunks), dtype=np.uint32)
        for i in range(0, X.shape[0], batch_size):
            X_batch = X[i : i + batch_size]
            encoded_X[i : i + batch_size] = self.dev.encode(X_batch)
        return encoded_X

    def _fit(
        self,
        encoded_X: np.ndarray[tuple[int, int, int], np.dtype[np.uint32]],
        one_hot_Y: np.ndarray[tuple[int, int], np.dtype[np.int8]],
        shuffle: bool = True,
        clause_drop_p: float = 0.0,
    ) -> None:
        N = encoded_X.shape[0]
        iota = np.arange(N)
        if shuffle:
            self.rng.shuffle(iota)
        encoded_X = encoded_X[iota]
        one_hot_Y = one_hot_Y[iota]

        # Precompute targets for each sample. 1 means the sample belongs to the class. -1 means, selected not classes. 0 means ignore.
        targets = self._target_sampling(one_hot_Y)

        self.dev.fit_epoch(encoded_X, targets, clause_drop_p)

    def score(self, encoded_X: np.ndarray):
        class_sums = self.dev.infer(encoded_X)
        return class_sums

    def _target_sampling(
        self, one_hot_Y: np.ndarray[tuple[int, int], np.dtype[np.int8]]
    ) -> np.ndarray[tuple[int, int], np.dtype[np.int8]]:
        N = one_hot_Y.shape[0]
        targets = np.copy(one_hot_Y).astype(np.int8)
        p = self.args.q / max(1, self.args.n_classes - 1)
        for i in range(N):
            # Each negative class can get feedback with prob (q / number_of_outputs - 1)
            false_classes = np.where(one_hot_Y[i, :] == 0)[0]
            if len(false_classes) > 0:
                not_skip = self.rng.random(size=len(false_classes)) <= p
                targets[i, false_classes[not_skip]] = -1

        return targets

    def get_weights(self) -> np.ndarray[tuple[int, int], np.dtype[np.float32]]:
        return self.dev.get_weights()

    def get_ta_states(self) -> np.ndarray[tuple[int, int, int], np.dtype[np.uint32]]:
        return self.dev.get_ta_states()

    def transform(
        self, X: np.ndarray, is_X_encoded: bool = False
    ) -> np.ndarray[tuple[int, int, int], np.dtype[np.bool]]:
        encoded_X = self.encode(X) if not is_X_encoded else X

        clause_output_patchwise = self.dev.transform_patchwise(encoded_X)
        clause_outputs = np.any(clause_output_patchwise, axis=-1).astype(np.uint8)
        return clause_outputs

    def transform_patchwise(
        self, X: np.ndarray, is_X_encoded: bool = False
    ) -> np.ndarray[tuple[int, int, int, int], np.dtype[np.bool]]:
        encoded_X = self.encode(X) if not is_X_encoded else X

        clause_output_patchwise = self.dev.transform_patchwise(encoded_X)
        return clause_output_patchwise

    def to(self, device: Literal["cpu", "cuda"]):
        if device == self.args.device:
            return

        orig_dev_state = self.dev.get_state_dict()
        if device == "cpu":
            from .backends.cpu.cpu_backend import CPUDevice

            self.dev = CPUDevice(self.args)
        elif device == "cuda":
            from .backends.cuda.cuda_backend import CUDADevice

            self.dev = CUDADevice(self.args)
        else:
            raise ValueError(f"Unsupported device: {device}")

        self.dev.load_state_dict(orig_dev_state)
        self.args.device = device

    def __getstate__(self):
        state = {
            "args": asdict(self.args),
            "params": self.dev.get_state_dict(),
        }
        return state

    def __setstate__(self, state):
        state["args"]["device"] = "cpu"
        self.__init__(**state["args"])
        self.dev.load_state_dict(state["params"])
