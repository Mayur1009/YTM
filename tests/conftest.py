import pytest

from ytm._core.utils import tqdm_disable

tqdm_disable()


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


def pytest_addoption(parser):
    parser.addoption(
        "--seed",
        type=int,
        default=42,
        help="seed for the rng that generates the per-test seed lists in tests/benchmarks",
    )


@pytest.fixture(params=DEVICES)
def device(request) -> str:
    return request.param
