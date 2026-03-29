import numpy as np
from typing import Literal, NamedTuple, Unpack
from dataclasses import asdict
from .args import TMArgs, T_args
from .backends.base import PackedClauses


class ClauseInfo(NamedTuple):
    feature_bounds: np.ndarray  # (total_clauses, n_raw_patch_feats, 2) closed [lower, upper]
    position_bounds: np.ndarray | None  # (total_clauses, 4) closed [min_y, max_y, min_x, max_x] or None
    is_valid: np.ndarray  # (total_clauses,) bool


class BaseTM:
    def __init__(
        self,
        n_clauses: int,
        T: float | int,
        s: float,
        dim: tuple[int, int, int],
        n_classes: int,
        **opt_args: Unpack[T_args],
    ):
        self.args = TMArgs(n_clauses, T, s, dim, n_classes, **opt_args)
        self.rng = np.random.default_rng(self.args.seed)

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
        one_hot_Y: np.ndarray,
        shuffle: bool = True,
        clause_drop_p: float = 0.0,
        batch_size: int = -1,
    ) -> None:
        N = X.shape[0]
        iota = np.arange(N)
        if shuffle:
            self.rng.shuffle(iota)
        X = X[iota]
        one_hot_Y = one_hot_Y[iota]

        targets = self._target_sampling(one_hot_Y)

        self.dev.fit_epoch(X, targets, clause_drop_p, batch_size)

    def score(self, X: np.ndarray, batch_size: int = -1):
        class_sums = self.dev.infer(X, batch_size)
        return class_sums

    def transform_patchwise(self, X: np.ndarray, batch_size: int = -1) -> np.ndarray:
        patch_outputs = self.dev.transform_patchwise(X, batch_size)
        return patch_outputs

    def _target_sampling(self, one_hot_Y: np.ndarray) -> np.ndarray:
        N = one_hot_Y.shape[0]
        targets = np.copy(one_hot_Y).astype(np.float32)
        for i in range(N):
            false_classes = np.where(one_hot_Y[i, :] == 0)[0]
            if len(false_classes) > 0:
                targets[i, false_classes] = -self.args.q / max(1, self.args.n_classes - 1)

        return targets

    def get_weights(self) -> np.ndarray:
        return self.dev.get_weights()

    def get_ta_states(self) -> np.ndarray:
        return self.dev.get_ta_states()

    def get_patch_weights(self) -> np.ndarray:
        return self.dev.get_patch_weights()

    def get_clauses(self) -> ClauseInfo:
        buf: PackedClauses = self.dev.pack_clauses()
        buf.to_cpu()

        position_bounds = buf.clause_position_bounds if self.args.position_literals else None

        return ClauseInfo(
            feature_bounds=buf.clause_feat_bounds,
            position_bounds=position_bounds,
            is_valid=buf.is_clause_valid.astype(bool),
        )

    def wac(self):
        pass

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
