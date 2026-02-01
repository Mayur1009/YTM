from abc import ABC, abstractmethod
from typing import Any, Optional
import numpy as np


class Backend(ABC):
    def __init__(self, header: str = "", seed: Optional[int] = None, **kwargs):
        self.header = header
        self.seed = seed
        self._set_rng(seed)

    @abstractmethod
    def _set_rng(self, seed: Optional[int]) -> None:
        raise NotImplementedError

    def _get_rng(self) -> Any:
        return getattr(self, "rng_dev", None)

    @abstractmethod
    def allocate(self, size: int, nbytes: int) -> Any:
        raise NotImplementedError

    @abstractmethod
    def to_device(self, dev: Any, host: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def to_host(self, host: np.ndarray, dev: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def memset(self, dev: Any, value: int, size: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def encode_batch(self, X: Any, encoded_X: Any, N: int, n_patches: int) -> None:
        raise NotImplementedError

    # @abstractmethod
    # def pack_clauses(self, *args, **kwargs) -> None:
    #     raise NotImplementedError
    #
    # @abstractmethod
    # def eval_clauses(self, *args, **kwargs) -> None:
    #     raise NotImplementedError
    #
    # @abstractmethod
    # def select_active_patch(self, *args, **kwargs) -> None:
    #     raise NotImplementedError
    #
    # @abstractmethod
    # def inference_eval(self, *args, **kwargs) -> None:
    #     raise NotImplementedError
    #
    # @abstractmethod
    # def evidence_to_update_prob(self, *args, **kwargs) -> None:
    #     raise NotImplementedError
    #
    # @abstractmethod
    # def update_clauses(self, *args, **kwargs) -> None:
    #     raise NotImplementedError
    #
    # @abstractmethod
    # def transform(self, *args, **kwargs) -> None:
    #     raise NotImplementedError
    #
    # @abstractmethod
    # def transform_patchwise(self, *args, **kwargs) -> None:
    #     raise NotImplementedError



