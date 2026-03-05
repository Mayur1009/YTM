import numpy as np
from ..args import TMArgs
from dataclasses import dataclass
from typing import Any
import abc


@dataclass
class FitBuffers:
    encoded_X: Any
    targets: Any
    packed_clauses: Any
    n_includes: Any
    clause_outputs: Any
    selected_patch_ids: Any
    pos_votes: Any
    neg_votes: Any
    update_probs: Any
    clause_drop_mask: Any


class BaseDevice(abc.ABC):
    def __init__(self, args: TMArgs):
        self.args = args

        self.n_clause_banks = 1 if self.args.coalesced else self.args.n_classes
        self.total_clauses = self.n_clause_banks * self.args.n_clauses

        assert (
            self.args.patch_dim is not None
            and self.args.patch_dim[0] <= self.args.dim[0]
            and self.args.patch_dim[1] <= self.args.dim[1]
        ), "Patch dimensions must be less than or equal to input dimensions"

        self.n_literals = (
            self.args.patch_dim[0] * self.args.patch_dim[1] * self.args.dim[2]
            + (self.args.dim[0] - self.args.patch_dim[0])
            + (self.args.dim[1] - self.args.patch_dim[1])
        )

        if self.args.negated_literals:
            self.n_literals *= 2

        self.n_patches = (self.args.dim[0] - self.args.patch_dim[0] + 1) * (
            self.args.dim[1] - self.args.patch_dim[1] + 1
        )
        self.n_literal_chunks = (self.n_literals + 31) // 32

        if self.args.max_included_literals <= 0 or self.args.max_included_literals > self.n_literals:
            self.args.max_included_literals = self.n_literals

        self.dev_init()

    @abc.abstractmethod
    def dev_init(self):
        pass

    def fit_epoch(self, encoded_X: np.ndarray, targets: np.ndarray, clause_drop_p: float) -> None:
        raise NotImplementedError("fit_epoch() not implemented for this device")

    def prepare_fit_buffers(
        self, encoded_X: np.ndarray, targets: np.ndarray, clause_drop_mask: np.ndarray
    ) -> FitBuffers:
        raise NotImplementedError("prepare_fit_buffers() not implemented for this device")

    def encode(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("encode() not implemented for this device")

    def decode(self, encoded_X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("decode() not implemented for this device")

    def pack_clauses(self, packed_clauses, n_includes):
        raise NotImplementedError("pack_clauses() not implemented for this device")

    def eval_clauses(
        self,
        packed_clauses,
        n_includes,
        clause_drop_mask,
        clause_outputs,
        encoded_X,
        e: int,
    ):
        raise NotImplementedError("eval_clauses() not implemented for this device")

    def select_patch_and_count_votes(self, clause_outputs, selected_patch_ids, pos_votes, neg_votes):
        raise NotImplementedError("select_patch_and_count_votes() not implemented for this device")

    def calc_update_prob(self, pos_votes, neg_votes, targets, update_probs, e: int):
        raise NotImplementedError("calc_update_prob() not implemented for this device")

    def update_clauses(
        self,
        n_includes,
        selected_patch_ids,
        clause_drop_mask,
        update_probs,
        encoded_X,
        targets,
        e: int,
    ):
        raise NotImplementedError("update_clauses() not implemented for this device")

    def infer(self, encoded_X: np.ndarray, batch_size: int = -1) -> np.ndarray:
        raise NotImplementedError("infer() not implemented for this device")

    def get_weights(self) -> np.ndarray:
        raise NotImplementedError("get_weights() not implemented for this device")

    def get_ta_states(self) -> np.ndarray:
        raise NotImplementedError("get_ta_states() not implemented for this device")

    def transform_patchwise(self, encoded_X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("transform_patchwise() not implemented for this device")

    def get_state_dict(self) -> dict:
        raise NotImplementedError("get_state_dict() not implemented for this device")

    def load_state_dict(self, state_dict: dict) -> None:
        raise NotImplementedError("load_state_dict() not implemented for this device")
