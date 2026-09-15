from dataclasses import dataclass, field
from enum import IntEnum
from typing import Literal

from .._core.config import BaseTMConfig, T_BaseTMConfig
from .._core.device_config import T_DeviceConfig
from .._core.utils import enum_to_header


class FbSignal(IntEnum):
    GRAD = 0
    DELTA_L = 1


class ActFn(IntEnum):
    SOFTMAX = 0
    SIGMOID = 1
    IDENTITY = 2


class LossFn(IntEnum):
    CE = 0
    MSE = 1
    MAE = 2
    SCE = 3
    ASL = 4
    TVERSKY = 5
    HUBER = 6


@dataclass(kw_only=True)
class TMConfig(BaseTMConfig):
    lr: float = 1.0
    lambda_: float = 1.0
    weight_init: Literal["random"] | float | str = "random"
    act_fn: Literal["softmax", "sigmoid", "identity"] = "softmax"
    loss_fn: Literal["ce", "sce", "mse", "mae", "huber", "tversky", "asl"] = "ce"
    loss_fn_kwargs: dict = field(default_factory=dict)
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

        assert self.act_fn.upper() in ActFn.__members__, (
            f"act_fn must be one of {[m.lower() for m in ActFn.__members__]}, got {self.act_fn!r}"
        )
        assert self.loss_fn.upper() in LossFn.__members__, (
            f"loss_fn must be one of {[m.lower() for m in LossFn.__members__]}, got {self.loss_fn!r}"
        )
        assert self.fb_signal.upper() in FbSignal.__members__, (
            f"fb_signal must be one of {[m.lower() for m in FbSignal.__members__]}, got {self.fb_signal!r}"
        )

        self._act_fn = ActFn[self.act_fn.upper()]
        self._loss_fn = LossFn[self.loss_fn.upper()]
        self._fb_signal = FbSignal[self.fb_signal.upper()]

        kw = self.loss_fn_kwargs
        self._loss_eps = kw.get("eps", 1e-4 if self.loss_fn == "sce" else (1e-7 if self.loss_fn == "ce" else 1e-6))
        self._loss_alpha = kw.get("alpha", 1.0 if self.loss_fn == "sce" else 0.5)
        self._loss_beta = kw.get("beta", 1.0 if self.loss_fn == "sce" else 0.5)
        self._loss_delta = kw.get("delta", 1.0)
        self._loss_clip = kw.get("clip", 0.05)
        self._loss_gamma_pos = kw.get("gamma_pos", 0.0)
        self._loss_gamma_neg = kw.get("gamma_neg", 4.0)

    def _build_header(self):
        super()._build_header()
        self._header += f"""
{enum_to_header("FB_SIGNAL", FbSignal)}
{enum_to_header("ACT", ActFn)}
{enum_to_header("LOSS", LossFn)}

#define FB_SIGNAL {self._fb_signal.value}
#define ACT_FN {self._act_fn.value}
#define LOSS_FN {self._loss_fn.value}
#define LOSS_EPS {float(self._loss_eps)}f
#define LOSS_ALPHA {float(self._loss_alpha)}f
#define LOSS_BETA {float(self._loss_beta)}f
#define LOSS_DELTA {float(self._loss_delta)}f
#define LOSS_CLIP {float(self._loss_clip)}f
#define LOSS_GAMMA_POS {float(self._loss_gamma_pos)}f
#define LOSS_GAMMA_NEG {float(self._loss_gamma_neg)}f
"""


class T_Config(T_BaseTMConfig, T_DeviceConfig, total=False):
    lr: float
    lambda_: float
    weight_init: Literal["random"] | float | str
    act_fn: Literal["softmax", "sigmoid", "identity"]
    loss_fn: Literal["ce", "sce", "mse", "mae", "huber", "tversky", "asl"]
    loss_fn_kwargs: dict
    fb_signal: Literal["delta_l", "grad"]
