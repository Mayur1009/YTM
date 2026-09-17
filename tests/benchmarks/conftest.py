import numpy as np
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--seed",
        type=int,
        default=42,
        help="seed for the rng that generates the per-test seed lists in tests/benchmarks",
    )


@pytest.fixture(scope="session")
def seeds(request) -> list[int]:
    seed_rng = np.random.default_rng(request.config.getoption("--seed"))
    return seed_rng.integers(1, 2**31 - 1, size=5).tolist()
