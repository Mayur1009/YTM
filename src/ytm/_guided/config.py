from dataclasses import dataclass, field
from enum import IntEnum
from typing import Literal

from .._core.config import BaseTMConfig, T_BaseTMConfig
from .._core.device_config import T_DeviceConfig
from .._core.utils import enum_to_header
from .backends.act_loss import ActLoss, SoftmaxCE


class FbSignal(IntEnum):
    GRAD = 0
    DELTA_L = 1


class ActFn(IntEnum):
    SOFTMAX = 0
    SIGMOID = 1
    IDENTITY = 2


@dataclass(kw_only=True)
class TMConfig(BaseTMConfig):
    lr: float = 1.0
    lambda_: float = 1.0
    weight_init: Literal["random"] | float | str = "random"
    act_loss: ActLoss = field(default_factory=SoftmaxCE)
    fb_signal: Literal["delta_l", "grad"] = "delta_l"

    def _derive_vars(self):
        super()._derive_vars()

        if isinstance(self.weight_init, str) and self.weight_init.startswith("random:"):
            try:
                n = float(self.weight_init[len("random:") :])
                if n <= 0:
                    raise ValueError
            except ValueError:
                raise ValueError(f"weight_init 'random:N' needs a positive float N, got {self.weight_init!r}") from None
        else:
            assert self.weight_init == "random" or (isinstance(self.weight_init, (int, float)) and self.weight_init >= 0), (
                f"weight_init must be 'random', 'random:N', or a positive number, got {self.weight_init!r}"
            )

        assert self.act_loss.act.upper() in ActFn.__members__, (
            f"act_loss.act must be one of {[m.lower() for m in ActFn.__members__]}, got {self.act_loss.act!r}"
        )
        assert self.fb_signal.upper() in FbSignal.__members__, (
            f"fb_signal must be one of {[m.lower() for m in FbSignal.__members__]}, got {self.fb_signal!r}"
        )

        self._act_fn = ActFn[self.act_loss.act.upper()]
        self._fb_signal = FbSignal[self.fb_signal.upper()]

    def _build_header(self):
        super()._build_header()
        self._header += f"""
{enum_to_header("FB_SIGNAL", FbSignal)}
{enum_to_header("ACT", ActFn)}

#define FB_SIGNAL {self._fb_signal.value}
#define ACT_FN {self._act_fn.value}
"""


class T_Config(T_BaseTMConfig, T_DeviceConfig, total=False):
    lr: float
    lambda_: float
    weight_init: Literal["random"] | float | str
    act_loss: ActLoss
    fb_signal: Literal["delta_l", "grad"]
