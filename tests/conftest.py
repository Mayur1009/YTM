import pytest

from ytm._core.utils import tqdm_disable

tqdm_disable()


def _cuda_usable() -> bool:
    try:
        from ytm._core._device_checks import check_cuda_available, resolve_cuda_props

        check_cuda_available()
        resolve_cuda_props(0)
        return True
    except Exception:
        return False


DEVICES = ["cpu:1"] + (["cuda"] if _cuda_usable() else [])


def pytest_addoption(parser):
    parser.addoption("--cuda-sanitizer", action="store_true", default=False, help="also run the sanitize tier under compute-sanitizer")
    parser.addoption(
        "--seed",
        type=int,
        default=42,
        help="seed for the rng that generates the per-test seed lists in tests/benchmarks",
    )


@pytest.fixture(params=DEVICES)
def device(request) -> str:
    return request.param
