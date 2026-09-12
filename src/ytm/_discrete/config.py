from dataclasses import dataclass

from .._core.config import BaseTMConfig, _BaseTMConfig_T
from .._core.device_config import _DeviceConfig_T


@dataclass(kw_only=True)
class TMConfig(BaseTMConfig):
    T: float | tuple[float, float]
    q: float = 1.0

    def _derive_vars(self):
        super()._derive_vars()

        assert not self.bias, "`bias` is not supported by the discrete update, it must stay False."

        if isinstance(self.T, (tuple, list)):
            self._T_min, self._T_max = (float(v) for v in self.T)
        else:
            self._T_min, self._T_max = -float(self.T), float(self.T)

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


class T_args(_BaseTMConfig_T, _DeviceConfig_T, total=False):
    q: float
