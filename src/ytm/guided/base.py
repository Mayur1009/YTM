from dataclasses import asdict
from typing import Literal, NamedTuple, Unpack

import numpy as np

from .args import T_args, TMArgs


class ClauseInfo(NamedTuple):
    feature_bounds: np.ndarray  # (total_clauses, n_raw_patch_feats, 2) closed [lower, upper]
    position_bounds: np.ndarray | None  # (total_clauses, 4) closed [min_y, max_y, min_x, max_x] or None
    clause_density: np.ndarray  # (total_clauses,) int, -1 marks an invalid clause (contains a contradiction)


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
        self._rng = np.random.default_rng(self.args.seed)

        if self.args.device.startswith("cpu"):
            from .backends.cpu.cpu_backend import CPUDevice

            self.dev = CPUDevice(self.args)
        elif self.args.device.startswith("cuda"):
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
        lambda_: float | None = None,
    ):
        assert np.prod(X.shape[1:]) == np.prod(self.args.dim), (
            f"Expected input features to match dim {self.args.dim}, but got {X.shape[1:]}"
        )
        assert Y.ndim == 2, f"Y must be 2D array (samples, outputs), got {Y.ndim}D"
        N = X.shape[0]
        iota = self._rng.permutation(np.arange(N))
        X = np.ascontiguousarray(X)[iota]
        Y = Y[iota].astype(np.float32)
        epoch_loss = self.dev.fit_epoch(X, Y, clause_drop_p, batch_size, lr=lr, lambda_=lambda_)
        return epoch_loss

    def score(self, X: np.ndarray, batch_size: int = -1):
        return self.dev.infer(np.ascontiguousarray(X), batch_size)

    def transform(self, X: np.ndarray, batch_size: int = -1, force_repack: bool = False) -> np.ndarray:
        clause_outputs = self.dev.transform(np.ascontiguousarray(X), batch_size, force_repack)
        return clause_outputs

    def transform_patchwise(self, X: np.ndarray, batch_size: int = -1, force_repack: bool = False) -> np.ndarray:
        patch_outputs = self.dev.transform_patchwise(np.ascontiguousarray(X), batch_size, force_repack)
        return patch_outputs

    def wic(self, class_id: int, polarity: int, pw_th: float = 0.0, force_repack: bool = False) -> np.ndarray:
        if self.dev.n_patches > 1 and not self.args.track_patch_weights:
            raise ValueError("track_patch_weights=True is required for wic() on a convolutional model.")
        return self.dev.wic(class_id, polarity, pw_th, force_repack)

    def wac(self, X: np.ndarray, target_classes: np.ndarray, polarity: int, force_repack: bool = False) -> np.ndarray:
        return self.dev.wac(np.ascontiguousarray(X), target_classes, polarity, force_repack)

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

    def get_bias(self) -> np.ndarray:
        return self.dev.get_bias()

    def get_ta_states(self) -> np.ndarray:
        return self.dev.get_ta_states()

    def get_literals(self) -> np.ndarray:
        return np.asarray(self.get_ta_states() >= self.args.include_state, dtype=np.uint8)

    def get_patch_weights(self) -> np.ndarray:
        return self.dev.get_patch_weights()

    def get_clauses(self, force_repack=False) -> ClauseInfo:
        self.dev.pack_clauses(force_repack)
        buf = self.dev.get_packed_clauses()

        clause_feat_bounds = buf.clause_feat_bounds.reshape((self.dev.n_clause_banks, self.args.n_clauses, self.dev.n_raw_patch_feats * 2))

        position_bounds = None
        if self.args.position_literals or self.dev.n_patches > 1:
            position_bounds = buf.clause_position_bounds.reshape((self.dev.n_clause_banks, self.args.n_clauses, 4))

        clause_density = buf.clause_density.reshape((self.dev.n_clause_banks, self.args.n_clauses))

        return ClauseInfo(
            feature_bounds=clause_feat_bounds,
            position_bounds=position_bounds,
            clause_density=clause_density,
        )

    def to(self, device: str):
        orig_dev_state = self.dev.get_state_dict()
        self.args.device = device
        if device.startswith("cpu"):
            from .backends.cpu.cpu_backend import CPUDevice

            self.dev = CPUDevice(self.args)
        elif device.startswith("cuda"):
            from .backends.cuda.cuda_backend import CUDADevice

            self.dev = CUDADevice(self.args)
        else:
            raise ValueError(f"Unsupported device: {device}")

        self.dev.load_state_dict(orig_dev_state)

    def set_threads(self, n: int) -> None:
        self.dev.set_threads(max(1, n))

    def get_state_dict(self) -> dict:
        from dataclasses import replace

        return {
            "args": asdict(replace(self.args, device="cpu")),
            "dev": self.dev.get_state_dict(),
            "rng": self._rng,
        }

    def load_state_dict(self, state: dict) -> None:
        from .backends.cpu.cpu_backend import CPUDevice

        self.args = TMArgs(**state["args"])
        self.dev = CPUDevice(self.args)
        self.dev.load_state_dict(state["dev"])
        self._rng = state["rng"]

    @classmethod
    def from_state_dict(cls, state: dict) -> "BaseTM":
        instance = cls.__new__(cls)
        instance.load_state_dict(state)
        return instance

    def __getstate__(self):
        return self.get_state_dict()

    def __setstate__(self, state):
        self.load_state_dict(state)
