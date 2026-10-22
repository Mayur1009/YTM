import abc
import contextlib
import copy
import types
from collections.abc import Iterator

import numpy as np

from ..config import BaseTMConfig
from ..device_config import DeviceConfig
from ..utils import tqdm_bar
from .buffers import FitBuffers, PackedClauses


class BaseDevice(abc.ABC):
    xp: types.ModuleType
    clause_weights: np.ndarray

    def __init__(self, config: BaseTMConfig, device_config: DeviceConfig, state: dict | None = None):
        self.config = config
        self.device_config = device_config
        self._rng = np.random.default_rng(self.config.seed + 1)
        self._setup()
        with self._ctx():
            self._init_consts()
            if state is None:
                self._init_params()
            else:
                self._restore_params(state)
            self._bind()

    def _ctx(self):
        return contextlib.nullcontext()

    @abc.abstractmethod
    def _setup(self): ...

    @abc.abstractmethod
    def _bind(self): ...

    @abc.abstractmethod
    def _to_host(self, arr) -> np.ndarray: ...

    @abc.abstractmethod
    def _to_dev(self, arr: np.ndarray): ...

    @abc.abstractmethod
    def _code_sections(self) -> dict[str, str]: ...

    @abc.abstractmethod
    def pack_clauses(self, force_repack: bool = False, full: bool = False): ...

    @abc.abstractmethod
    def fit_epoch(self, X: np.ndarray, Y: np.ndarray, clause_drop_p: float, *args, **kwargs): ...

    def fit_sample(self, rng_key: int, buf, e: int):
        self.pack_clauses()
        self._fit_eval(buf, e, rng_key)
        self._fit_voting(buf)
        self._fit_decide_fb(buf, e, rng_key)
        self._fit_apply_fb(buf, e, rng_key)
        if self.config.weighted:
            self._fit_update_weights(buf)

    @abc.abstractmethod
    def _fit_eval(self, buf, e: int, rng_key): ...

    @abc.abstractmethod
    def _fit_voting(self, buf): ...

    @abc.abstractmethod
    def _fit_decide_fb(self, buf, e: int, rng_key: int): ...

    @abc.abstractmethod
    def _fit_apply_fb(self, buf, e: int, rng_key): ...

    @abc.abstractmethod
    def _fit_update_weights(self, buf): ...

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
    def _infer_batch(self, Xb, bs: int): ...

    @abc.abstractmethod
    def transform(self, X: np.ndarray, batch_size: int = -1, force_repack: bool = False): ...

    @abc.abstractmethod
    def _patch_outputs(self, X: np.ndarray, batch_size: int = -1, force_repack: bool = False, desc: str = "Transform") -> np.ndarray: ...

    def transform_patchwise(self, X: np.ndarray, batch_size: int = -1, force_repack: bool = False) -> np.ndarray:
        cfg = self.config
        out = self._patch_outputs(X, batch_size, force_repack)
        return out.reshape(out.shape[0], cfg._n_clause_banks, cfg._n_clauses, cfg._n_patches_y, cfg._n_patches_x)

    def _batches(self, N: int, batch_size: int, desc: str) -> Iterator[tuple[int, int]]:
        bs = N if batch_size == -1 else batch_size
        with tqdm_bar(None, total=N, desc=desc) as bar:
            for start in range(0, N, bs):
                end = min(start + bs, N)
                yield start, end
                bar.update(end - start)

    @abc.abstractmethod
    def wic(self, class_id: int, polarity: int, pw_th: float = 0.0, force_repack: bool = False): ...

    @abc.abstractmethod
    def wac(self, X: np.ndarray, target_classes: np.ndarray, polarity: int, batch_size: int = -1, force_repack: bool = False): ...

    def set_threads(self, n: int):
        raise NotImplementedError(f"set_threads is not supported on {self.device_config.device!r}.")

    def _build_code(self) -> str:
        return self.config._header + "\n".join(self._code_sections().values())

    def _fit_iter(self, pbar) -> Iterator[tuple[int, int]]:
        keys = self._rng.integers(1, 1 << 62, size=len(pbar), dtype=np.uint64).tolist()
        for e in pbar:
            yield e, keys[e]

    def _fit_drop_mask(self, clause_drop_p: float):
        cfg = self.config
        if clause_drop_p <= 0.0:
            return self.xp.zeros(cfg._total_clauses, dtype=np.int8)
        return self.xp.asarray((self._rng.random(cfg._total_clauses) <= clause_drop_p).astype(np.int8))

    def set_feat_drop(self, buf: FitBuffers, fids: int | list[int], value: int) -> None:
        cfg = self.config
        half = cfg._n_literals // 2
        for fid in [fids] if isinstance(fids, int) else fids:
            start = cfg._n_position_feats + int(cfg._literal_offsets[fid])
            end = cfg._n_position_feats + int(cfg._literal_offsets[fid + 1])
            buf.lit_drop_mask[start:end] = value
            if cfg.negated_literals:
                buf.lit_drop_mask[half + start : half + end] = value

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
