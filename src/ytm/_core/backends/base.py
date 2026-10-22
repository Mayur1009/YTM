import abc
import contextlib
import copy
import types
from collections.abc import Iterator
from typing import Any, Generic, TypeVar

import numpy as np

from ..config import BaseTMConfig
from ..device_config import DeviceConfig
from ..utils import tqdm_bar
from .buffers import FitBuffers, PackedClauses

BuffersT = TypeVar("BuffersT", bound=FitBuffers)
ConfigT = TypeVar("ConfigT", bound=BaseTMConfig)
ArrayT = TypeVar("ArrayT")


class BaseDevice(abc.ABC, Generic[BuffersT, ConfigT, ArrayT]):
    xp: types.ModuleType

    def __init__(self, config: ConfigT, device_config: DeviceConfig, state: dict | None = None):
        self.config = config
        self.device_config = device_config
        self._rng = np.random.default_rng(self.config.seed)
        self._setup()
        with self._ctx():
            self._init_consts()
            if state is None:
                self._init_params()
            else:
                self._restore_params(state)
            self._bind()
            self.buf = self._fit_buffers_cls()(make_ptrs=self.xp is np, xp=self.xp, cfg=self.config)

    def _ctx(self):
        return contextlib.nullcontext()

    @abc.abstractmethod
    def _setup(self): ...

    @abc.abstractmethod
    def _bind(self): ...

    @abc.abstractmethod
    def _to_host(self, arr: ArrayT) -> np.ndarray: ...

    @abc.abstractmethod
    def _to_dev(self, arr: np.ndarray) -> ArrayT: ...

    @abc.abstractmethod
    def _code_sections(self) -> dict[str, str]: ...

    @abc.abstractmethod
    def pack_clauses(self, force_repack: bool = False, full: bool = False): ...

    @staticmethod
    @abc.abstractmethod
    def _fit_buffers_cls() -> type[BuffersT]: ...

    def _chunk_size(self, N: int, bytes_per_smp: int) -> int:
        return N

    def fit_epoch(self, X: np.ndarray, Y: np.ndarray, clause_drop_p: float, **kwargs):
        with self._ctx():
            self.pre_epoch(X, Y, clause_drop_p, **kwargs)

            N = X.shape[0]
            chunk_N = self._chunk_size(N, self.buf.per_sample_bytes())

            total = 0.0
            for e in tqdm_bar(range(N), desc="Fit"):
                if e == self.buf.chunk_end:
                    self.buf.load_chunk(e, min(e + chunk_N, N))

                total += self.fit_sample(e)

            out = self.post_epoch(total)
            self.buf.end_epoch()
            return out

    def fit_sample(self, e: int) -> float:
        i = e - self.buf.chunk_start

        self.pre_sample(e)

        self.pack_clauses()
        self._fit_eval(i)
        self._fit_voting()
        self._fit_decide_fb(i)
        self._fit_apply_fb(i)
        if self.config.weighted:
            self._fit_update_weights()

        return self.post_sample(e)

    def pre_epoch(self, X: np.ndarray, Y: np.ndarray, clause_drop_p: float, **kwargs) -> None:
        assert X.dtype == self.config._fbound_dtype, f"X must be {np.dtype(self.config._fbound_dtype)}, got {X.dtype}"

        if clause_drop_p <= 0:
            clause_drop_mask = self.xp.asarray(np.zeros(self.config._total_clauses, dtype=np.uint8))
        else:
            clause_drop_mask = self.xp.asarray((self._rng.random(self.config._total_clauses) <= clause_drop_p), dtype=np.uint8)

        self.buf.start_epoch(
            X=X,
            Y=Y,
            clause_drop_mask=clause_drop_mask,
            rng_keys=self._rng.integers(1, 1 << 62, size=X.shape[0], dtype=np.uint64),
            **kwargs,
        )

    def post_epoch(self, total: float) -> Any:
        return None

    def pre_sample(self, e: int) -> None:
        pass

    def post_sample(self, e: int) -> float:
        return 0.0

    @abc.abstractmethod
    def _fit_eval(self, e: int): ...

    @abc.abstractmethod
    def _fit_voting(self): ...

    @abc.abstractmethod
    def _fit_decide_fb(self, e: int): ...

    @abc.abstractmethod
    def _fit_apply_fb(self, e: int): ...

    @abc.abstractmethod
    def _fit_update_weights(self): ...

    def _init_weights(self):
        cfg = self.config
        shape = (cfg.n_classes, cfg._n_clauses)
        sign = np.ones(shape, dtype=np.float32)

        if cfg.negative_clauses:
            n_neg = cfg._n_clauses // 2
            if cfg.coalesced:
                pol = np.ones(cfg._n_clauses, dtype=np.float32)
                pol[n_neg:] = -1.0
                for i in range(cfg.n_classes):
                    sign[i, :] = self._rng.permutation(pol)
            else:
                sign[:, n_neg:] = -1.0

        if cfg.weight_init == "random":
            mag = self._rng.uniform(0.0, 1.0, size=shape)
        elif isinstance(cfg.weight_init, str):
            mag = self._rng.uniform(0.0, float(cfg.weight_init[len("random:") :]), size=shape)
        else:
            mag = float(cfg.weight_init)

        self.clause_weights = self.xp.asarray(sign * mag, dtype=np.float32)

    def infer(self, X: np.ndarray, batch_size: int = -1, force_repack: bool = False) -> np.ndarray:
        cfg = self.config
        with self._ctx():
            self.pack_clauses(force_repack)
            out = self.xp.empty((X.shape[0], cfg.n_classes), dtype=np.float32)
            for start, end in self._batches(X.shape[0], batch_size, "Infer"):
                out[start:end] = self._infer_batch(self.xp.asarray(X[start:end], dtype=cfg._fbound_dtype), end - start)
            return self._to_host(out)

    @abc.abstractmethod
    def _infer_batch(self, Xb: ArrayT, bs: int) -> ArrayT: ...

    def transform(self, X: np.ndarray, batch_size: int = -1, force_repack: bool = False) -> np.ndarray:
        cfg = self.config
        N = X.shape[0]
        out = np.empty((N, cfg._total_clauses), dtype=np.int8)
        with self._ctx():
            self.pack_clauses(force_repack)
            for start, end in self._batches(N, batch_size, "Transform"):
                Xb = self.xp.asarray(X[start:end], dtype=cfg._fbound_dtype)
                out[start:end] = self._to_host(self._clause_outputs(Xb, end - start))
        return out.reshape(N, cfg._n_clause_banks, cfg._n_clauses)

    @abc.abstractmethod
    def _clause_outputs(self, Xb: ArrayT, bs: int) -> ArrayT: ...

    @abc.abstractmethod
    def _patch_outputs(self, Xb: ArrayT, bs: int) -> ArrayT: ...

    def transform_patchwise(self, X: np.ndarray, batch_size: int = -1, force_repack: bool = False) -> np.ndarray:
        cfg = self.config
        N = X.shape[0]
        out = np.empty((N, cfg._total_clauses, cfg._n_patches), dtype=np.int8)
        with self._ctx():
            self.pack_clauses(force_repack)
            for start, end in self._batches(N, batch_size, "Transform"):
                Xb = self.xp.asarray(X[start:end], dtype=cfg._fbound_dtype)
                out[start:end] = self._to_host(self._patch_outputs(Xb, end - start))
        return out.reshape(N, cfg._n_clause_banks, cfg._n_clauses, cfg._n_patches_y, cfg._n_patches_x)

    def _batches(self, N: int, batch_size: int, desc: str) -> Iterator[tuple[int, int]]:
        bs = N if batch_size == -1 else batch_size
        with tqdm_bar(None, total=N, desc=desc) as bar:
            for start in range(0, N, bs):
                end = min(start + bs, N)
                yield start, end
                bar.update(end - start)

    @abc.abstractmethod
    def wic(self, class_id: int, polarity: int, pw_th: float = 0.0, force_repack: bool = False) -> np.ndarray: ...

    def wac(self, X: np.ndarray, target_classes: np.ndarray, polarity: int, batch_size: int = -1, force_repack: bool = False) -> np.ndarray:
        cfg = self.config
        N = X.shape[0]
        output = np.empty((N, *cfg._dim), dtype=np.float32)
        with self._ctx():
            self.pack_clauses(force_repack)
            for start, end in self._batches(N, batch_size, "WAC"):
                Xb = self.xp.asarray(X[start:end], dtype=cfg._fbound_dtype)
                target_b = self.xp.asarray(target_classes[start:end], dtype=np.int32)
                output[start:end] = self._to_host(self._wac_batch(Xb, target_b, polarity, end - start))
        return output

    @abc.abstractmethod
    def _wac_batch(self, Xb: ArrayT, target_b: ArrayT, polarity: int, bs: int) -> ArrayT: ...

    def set_threads(self, n: int):
        raise NotImplementedError(f"set_threads is not supported on {self.device_config.device!r}.")

    def _build_code(self) -> str:
        return self.config._header + "\n".join(self._code_sections().values())

    def set_feat_drop(self, fids: int | list[int], value: int) -> None:
        cfg = self.config
        half = cfg._n_literals // 2
        for fid in [fids] if isinstance(fids, int) else fids:
            start = cfg._n_position_feats + int(cfg._literal_offsets[fid])
            end = cfg._n_position_feats + int(cfg._literal_offsets[fid + 1])
            self.buf.lit_drop_mask[start:end] = value
            if cfg.negated_literals:
                self.buf.lit_drop_mask[half + start : half + end] = value

    # == initializations ==
    def _init_params(self):
        self._init_weights()
        self._init_clauses()
        self._init_packed_clauses()
        self._init_patch_weights()

    def _restore_params(self, state: dict):
        cfg = self.config
        self.clause_weights = self.xp.array(state["clause_weights"], dtype=np.float32, order="C")
        self.ta_states = self.xp.array(state["ta_states"], dtype=cfg._ta_dtype, order="C")
        self.patch_weights = self.xp.array(state["patch_weights"], dtype=np.int32, order="C")
        self._rng = copy.deepcopy(state["rng"])
        self._init_packed_clauses()

    def _init_consts(self):
        cfg = self.config
        self.therm_bits = self.xp.asarray(cfg._therm_bits, dtype=cfg._fbound_dtype)
        self.literal_offsets = self.xp.asarray(cfg._literal_offsets, dtype=cfg._nlits_dtype)

    def _init_clauses(self):
        cfg = self.config
        shape = (cfg._total_clauses, cfg._n_literals)
        states = cfg._ta_init(self._rng, shape, cfg.n_states, cfg=cfg, weights=self._to_host(self.clause_weights))
        self.ta_states = self.xp.asarray(states, dtype=cfg._ta_dtype)

    def _init_patch_weights(self):
        cfg = self.config
        shape = (cfg._total_clauses, cfg._n_patches) if cfg.track_patch_weights else (1, 1)
        self.patch_weights = self.xp.zeros(shape, dtype=np.int32)

    def _init_packed_clauses(self):
        cfg = self.config
        self.packed_clauses = PackedClauses(
            clause_feat_ids=self.xp.empty((cfg._total_clauses, cfg._n_raw_patch_feats), dtype=cfg._nfeat_dtype),
            clause_feat_bounds=self.xp.empty((cfg._total_clauses, cfg._n_raw_patch_feats, 2), dtype=cfg._fbound_dtype),
            clause_n_feats=self.xp.empty(cfg._total_clauses, dtype=cfg._nfeat_dtype),
            clause_position_bounds=self.xp.empty((cfg._total_clauses, 4) if cfg._n_patches > 1 else (1, 1), dtype=cfg._pbound_dtype),
            has_contra=self.xp.empty(cfg._total_clauses, dtype=np.int8),
            clause_len=self.xp.empty(cfg._total_clauses, dtype=cfg._nlits_dtype),
            is_clause_synced=self.xp.zeros(cfg._total_clauses, dtype=np.int8),
        )

    # == Getters ==
    def get_ta_states(self) -> np.ndarray:
        cfg = self.config
        return self._to_host(self.ta_states).reshape(cfg._n_clause_banks, cfg._n_clauses, cfg._n_literals)

    def get_weights(self) -> np.ndarray:
        return self._to_host(self.clause_weights)

    def get_patch_weights(self) -> np.ndarray:
        cfg = self.config
        if not cfg.track_patch_weights:
            raise RuntimeError("`track_patch_weights` must be True to get the patch weights.")
        return self._to_host(self.patch_weights).reshape(cfg._n_clause_banks, cfg._n_clauses, cfg._n_patches_y, cfg._n_patches_x)

    def get_packed_clauses(self) -> PackedClauses:
        pc = self.packed_clauses
        return PackedClauses(
            clause_feat_bounds=self._to_host(pc.clause_feat_bounds),
            clause_position_bounds=self._to_host(pc.clause_position_bounds),
            clause_feat_ids=self._to_host(pc.clause_feat_ids),
            clause_n_feats=self._to_host(pc.clause_n_feats),
            has_contra=self._to_host(pc.has_contra),
            clause_len=self._to_host(pc.clause_len),
            is_clause_synced=self._to_host(pc.is_clause_synced),
        )

    # == serialization ==
    def get_state_dict(self) -> dict:
        return {
            "ta_states": self._to_host(self.ta_states),
            "clause_weights": self._to_host(self.clause_weights),
            "patch_weights": self._to_host(self.patch_weights),
            "rng": copy.deepcopy(self._rng),
        }

    def load_state_dict(self, state: dict) -> None:
        """Written in place, so anything already pointing at these arrays stays valid."""
        with self._ctx():
            self.ta_states[:] = self.xp.asarray(state["ta_states"])
            self.clause_weights[:] = self.xp.asarray(state["clause_weights"])
            self.patch_weights[:] = self.xp.asarray(state["patch_weights"])
            self._rng = copy.deepcopy(state["rng"])
            self.packed_clauses.is_clause_synced.fill(0)
