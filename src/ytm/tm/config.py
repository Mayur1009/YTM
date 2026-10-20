from dataclasses import dataclass

import numpy as np

from .._core.config import BaseTMConfig, T_BaseTMConfig
from .._core.device_config import T_DeviceConfig

_FLT_MAX = float(np.finfo(np.float32).max)


def _clamp_f32(x: float) -> float:
    v = np.float32(x)
    if np.isinf(v):
        return -_FLT_MAX if v < 0 else _FLT_MAX
    return float(v)


@dataclass(kw_only=True)
class TMConfig(BaseTMConfig):
    T: float | tuple[float, float]
    q: float = 1.0
    weight_init: int = 1

    def _derive_vars(self):
        super()._derive_vars()

        assert isinstance(self.weight_init, int) and not isinstance(self.weight_init, bool) and self.weight_init > 0, (
            f"`weight_init` must be a positive int, got {self.weight_init!r}"
        )

        if isinstance(self.T, (tuple, list)):
            self._T_min, self._T_max = (_clamp_f32(v) for v in self.T)
        else:
            self._T_min, self._T_max = -_clamp_f32(self.T), _clamp_f32(self.T)

        assert self._T_min < self._T_max, f"`T` must give T_min < T_max, got {self._T_min} and {self._T_max}"

    def _build_header(self):
        super()._build_header()
        self._header += f"""
#define T_MIN {self._T_min}f
#define T_MAX {self._T_max}f
"""


@dataclass(kw_only=True)
class RegressionConfig(TMConfig):
    y_range: tuple[float, float] = (0.0, 1.0)

    def _derive_vars(self):
        super()._derive_vars()
        assert self.y_range[0] < self.y_range[1], f"`y_range[0]` must be < `y_range[1]`, got {self.y_range}"


class T_Config(T_BaseTMConfig, T_DeviceConfig, total=False):
    q: float
    weight_init: int
