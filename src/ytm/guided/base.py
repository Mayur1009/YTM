from dataclasses import asdict
from typing import Literal, NamedTuple, Unpack

import numpy as np

from .args import T_args, TMArgs


class ClauseInfo(NamedTuple):
    feature_bounds: np.ndarray  # (total_clauses, n_raw_patch_feats, 2) closed [lower, upper]
    position_bounds: np.ndarray | None  # (total_clauses, 4) closed [min_y, max_y, min_x, max_x] or None
    is_valid: np.ndarray  # (total_clauses,) bool


class BaseTM:
    def __init__(
        self,
        n_clauses: int,
        s: float,
        dim: tuple[int, int, int],
        n_classes: int,
        **opt_args: Unpack[T_args],
    ):
        self.args = TMArgs(n_clauses, s, dim, n_classes, **opt_args)
        self.np_rng = np.random.default_rng(self.args.seed)
        self.rng_state = self.np_rng.integers(1, 1 << 63, dtype=np.uint64)

        if self.args.device == "cpu":
            from .backends.cpu.cpu_backend import CPUDevice

            self.dev = CPUDevice(self.args)
        elif self.args.device == "cuda":
            from .backends.cuda.cuda_backend import CUDADevice

            self.dev = CUDADevice(self.args)
        else:
            raise ValueError(f"Unsupported device: {self.args.device}")

    def _fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        shuffle: bool = True,
        clause_drop_p: float = 0.0,
        batch_size: int = -1,
        lr: float | None = None,
    ) -> np.ndarray:
        assert np.prod(X.shape[1:]) == np.prod(self.args.dim), f"Expected input features to match dim {self.args.dim}, but got {X.shape[1:]}"
        assert Y.ndim == 2, f"Y must be 2D array (samples, outputs), got {Y.ndim}D"

        N = X.shape[0]
        iota = np.arange(N)
        if shuffle:
            self.np_rng.shuffle(iota)
        X = X[iota]
        Y = Y[iota]

        Y = Y.astype(np.float32)

        loss = self.dev.fit_epoch(X, Y, clause_drop_p, batch_size, lr=lr)
        self.rng_state = self.np_rng.integers(1, 1 << 63, dtype=np.uint64)
        return loss

    def score(self, X: np.ndarray, batch_size: int = -1):
        return self.dev.infer(X, batch_size)

    def transform_patchwise(self, X: np.ndarray, batch_size: int = -1) -> np.ndarray:
        patch_outputs = self.dev.transform_patchwise(X, batch_size)
        return patch_outputs

    def freeze_clauses(self, class_id: int, clause_ids: list[int] | np.ndarray):
        if self.args.coalesced:
            if class_id != 0:
                print(f"Warning: coalesced is true, ignoring class_id {class_id} and freezing clauses for all classes")
            self.dev.freeze_clauses(0, clause_ids)
        else:
            assert class_id < self.args.n_classes, f"Invalid class_id {class_id} for n_classes {self.args.n_classes}"
            self.dev.freeze_clauses(class_id, clause_ids)

    def unfreeze_clauses(self):
        self.dev.unfreeze_clauses()

    def get_weights(self) -> np.ndarray:
        return self.dev.get_weights()

    def get_ta_states(self) -> np.ndarray:
        return self.dev.get_ta_states()

    def get_literals(self) -> np.ndarray:
        return np.asarray(self.get_ta_states() >= self.args.include_state, dtype=np.uint8)

    def get_patch_weights(self) -> np.ndarray:
        return self.dev.get_patch_weights()

    def get_clauses(self, force_repack=False) -> ClauseInfo:
        self.dev.pack_clauses(force_repack)
        buf = self.dev.packed_clauses.get()

        clause_feat_bounds = buf.clause_feat_bounds.reshape(
            (self.dev.n_clause_banks, self.args.n_clauses, self.dev.n_raw_patch_feats * 2)
        )

        position_bounds = None
        if self.args.position_literals or self.dev.n_patches > 1:
            position_bounds = buf.clause_position_bounds.reshape((self.dev.n_clause_banks, self.args.n_clauses, 4))

        is_valid = buf.is_clause_synced.reshape((self.dev.n_clause_banks, self.args.n_clauses)).astype(bool)

        return ClauseInfo(
            feature_bounds=clause_feat_bounds,
            position_bounds=position_bounds,
            is_valid=is_valid,
        )

    def to(self, device: Literal["cpu", "cuda"]):
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

    def set_threads(self, n: int) -> None:
        self.args.n_threads = max(1, n)
        self.dev.set_threads(self.args.n_threads)

    def get_state_dict(self) -> dict:
        return {
            "args": asdict(self.args),
            "params": self.dev.get_state_dict(),
            "rng_state": self.rng_state,
            "np_rng_state": self.np_rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict) -> None:
        state["args"]["device"] = "cpu"
        BaseTM.__init__(self, **state["args"])
        self.dev.load_state_dict(state["params"])

        self.rng_state = state.get("rng_state", self.rng_state)
        self.np_rng.bit_generator.state = state.get("np_rng_state", self.np_rng.bit_generator.state)

    @classmethod
    def from_state_dict(cls, state: dict) -> "BaseTM":
        instance = cls.__new__(cls)
        instance.load_state_dict(state)
        return instance

    def __getstate__(self):
        return self.get_state_dict()

    def __setstate__(self, state):
        self.load_state_dict(state)
