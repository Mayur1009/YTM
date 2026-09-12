import pytest

from ytm._core._device_checks import parse_device


@pytest.mark.parametrize(
    "device, kind, n",
    [
        ("cpu", "cpu", 1),  # bare cpu is one thread
        ("cpu:4", "cpu", 4),
        ("cpu:0", "cpu", 1),  # clamped, zero threads is not a thing
        ("cpu:-3", "cpu", 1),
        ("CPU:2", "cpu", 2),  # case and spacing are tolerated
        (" cpu : 2", "cpu", 2),
        ("cuda", "cuda", 0),  # bare cuda is gpu 0
        ("cuda:1", "cuda", 1),
        ("cuda:-1", "cuda", 0),
    ],
)
def test_parsed_kind_and_count(device, kind, n):
    assert parse_device(device) == (kind, n)


@pytest.mark.parametrize("device", ["gpu", "cpu:abc", "cpu:1.5", "", "cudaa", "cpu:2:3"])
def test_malformed_strings_raise(device):
    with pytest.raises(ValueError):
        parse_device(device)
