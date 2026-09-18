import pathlib
from ctypes import POINTER, c_int, c_uint8, c_uint64
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..._core.backends.cpu import CPUDevice as CoreCPUDevice
from ..._core.backends.cpu import CPUFitBuffers, float_p, read_file
from ..._core.utils import tqdm_bar
from .base import BaseDevice as DiscreteBaseDevice

uint8_p = POINTER(c_uint8)


@dataclass(kw_only=True)
class DiscreteFitBuffers(CPUFitBuffers):
    feedback_type: Any  # (n_clauses, n_classes) one of FB_NONE/T1A/T1B/T2, indexed [rel_clause, class]
    prob: Any  # (n_classes,) signed update probability for the current sample
    label_probs: Any  # (n_samples, n_classes) chance a class gives feedback at all

    p_feedback_type: Any = field(init=False)
    p_prob: Any = field(init=False)
    p_label_probs: Any = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.p_feedback_type = self.feedback_type.ctypes.data_as(uint8_p)
        self.p_prob = self.prob.ctypes.data_as(float_p)
        self.p_label_probs = self.label_probs.ctypes.data_as(float_p)


class CPUDevice(DiscreteBaseDevice, CoreCPUDevice):
    def _code_sections(self) -> dict[str, str]:
        sections = super()._code_sections()
        sections["update.c"] = read_file(pathlib.Path(__file__).parent / "update.c")
        return sections

    def _fit_allocs(self, X: np.ndarray, Y: np.ndarray, clause_drop_mask: np.ndarray, label_probs: np.ndarray) -> DiscreteFitBuffers:
        cfg = self.config
        for name, arr, dtype in (("X", X, cfg._fbound_dtype), ("Y", Y, np.float32), ("label_probs", label_probs, np.float32)):
            assert arr.dtype == dtype and arr.flags.c_contiguous, (
                f"`{name}` must be C contiguous {np.dtype(dtype)}, got {arr.dtype} contiguous={arr.flags.c_contiguous}"
            )

        return DiscreteFitBuffers(
            X=X,
            Y=Y,
            clause_drop_mask=clause_drop_mask,
            clause_output=np.empty(cfg._total_clauses, dtype=np.int8),
            selected_pids=np.empty(cfg._total_clauses if cfg._n_patches > 1 else 1, dtype=cfg._npatches_dtype),
            votes=np.empty(cfg.n_classes, dtype=np.float32),
            feedback_type=np.zeros((cfg._n_clauses, cfg.n_classes), dtype=np.uint8),
            prob=np.zeros(cfg.n_classes, dtype=np.float32),
            label_probs=label_probs,
        )

    def fit_epoch(self, X: np.ndarray, Y: np.ndarray, clause_drop_p: float, batch_size: int, label_probs: np.ndarray) -> None:
        buf = self._fit_allocs(X, Y, self._fit_drop_mask(clause_drop_p), label_probs)

        for e, rng_key in self._fit_samples(tqdm_bar(range(X.shape[0]), desc="Fit")):
            self.fit_sample(rng_key, buf, e)

    def fit_sample(self, rng_key: int, buf: DiscreteFitBuffers, e: int) -> None:
        self.pack_clauses()
        self._fit_eval(buf, e, rng_key)
        self._fit_voting(buf)
        self._fit_decide_fb(buf, e, rng_key)
        self._fit_apply_fb(buf, e, rng_key)
        self._fit_update_weights(buf)

    def _fit_decide_fb(self, buf: DiscreteFitBuffers, e: int, rng_key: int) -> None:
        """Turn the votes into a per class update probability, then pick a feedback type per clause."""
        self.lib.calc_update_prob(buf.p_votes, buf.p_Y, c_int(e), buf.p_prob)
        self.lib.decide_feedback(
            c_uint64(rng_key),
            buf.p_clause_output,
            self.p_clause_len,
            buf.p_clause_drop_mask,
            buf.p_prob,
            buf.p_label_probs,
            c_int(e),
            self.p_clause_weights,
            buf.p_feedback_type,
        )

    def _fit_update_weights(self, buf: DiscreteFitBuffers) -> None:
        self.lib.update_weights(buf.p_feedback_type, self.p_clause_weights)

    def _fit_apply_fb(self, buf: DiscreteFitBuffers, e: int, rng_key: int) -> None:
        self.lib.update_clauses(
            c_uint64(rng_key),
            buf.p_clause_output,
            buf.p_selected_pids,
            buf.p_X,
            c_int(e),
            self.p_literal_offsets,
            buf.p_feedback_type,
            self.p_ta_states,
            self.p_is_clause_synced,
        )
