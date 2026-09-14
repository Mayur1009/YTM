from dataclasses import dataclass

from .._core.config import BaseTMConfig, T_BaseTMConfig
from .._core.device_config import T_DeviceConfig


@dataclass(kw_only=True)
class TMConfig(BaseTMConfig):
    T: float | tuple[float, float]
    q: float = 1.0
    weight_init: int = 1
    allow_polarity_change: bool = True

    def _derive_vars(self):
        super()._derive_vars()

        assert isinstance(self.weight_init, int) and not isinstance(self.weight_init, bool) and self.weight_init > 0, (
            f"`weight_init` must be a positive int, got {self.weight_init!r}"
        )

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
#define ALLOW_POLARITY_CHANGE {int(self.allow_polarity_change)}
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
    allow_polarity_change: bool
