import pytest

from ytm import pbar

pbar.disable()


def _cuda_usable() -> bool:
    try:
        import cupy as cp
    except ImportError:
        return False

    try:
        cp.cuda.runtime.getDeviceProperties(0)
    except cp.cuda.runtime.CUDARuntimeError:
        return False
    return True


DEVICES = ["cpu:1"] + (["cuda"] if _cuda_usable() else [])


@pytest.fixture(params=DEVICES)
def device(request) -> str:
    return request.param
