from ..._core.backends.base import BaseDevice
from ..._core.config import BaseTMConfig
from ..._core.device_config import DeviceConfig


def make_device(config: BaseTMConfig, device_config: DeviceConfig) -> BaseDevice:
    if device_config._device_kind == "cpu":
        from .cpu import CPUDevice as cls
    else:
        from .cuda import CUDADevice as cls

    cls.make_device = staticmethod(make_device)
    return cls(config, device_config)
