from ..._core.backends.cuda import CUDADevice as CoreCUDADevice
from .base import BaseDevice as DiscreteBaseDevice


class CUDADevice(DiscreteBaseDevice, CoreCUDADevice): ...
