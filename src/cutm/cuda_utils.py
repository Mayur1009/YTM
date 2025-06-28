import pathlib
import pycuda.autoinit  # noqa: F401
from pycuda.driver import device_attribute  # pyright: ignore[reportAttributeAccessIssue]


def get_kernel(file, current_dir):
    path = current_dir.joinpath(file)
    with path.open("r") as f:
        ker = f.read()
    return ker


def get_device_properties():
    """Query GPU device properties for optimization"""
    device = pycuda.autoinit.device
    attrs = device.get_attributes()
    properties = {
        "max_threads_per_block": attrs[device_attribute.MAX_THREADS_PER_BLOCK],
        "max_block_dim_x": attrs[device_attribute.MAX_BLOCK_DIM_X],
        "max_grid_dim_x": attrs[device_attribute.MAX_GRID_DIM_X],
        "warp_size": attrs[device_attribute.WARP_SIZE],
        "multiprocessor_count": attrs[device_attribute.MULTIPROCESSOR_COUNT],
        "max_shared_memory_per_block": attrs[device_attribute.MAX_SHARED_MEMORY_PER_BLOCK],
    }
    return properties


def kernel_config(data_size, props, preferred_block_size=128):
    """Get optimal grid and block configuration for 1D kernel"""

    # Ensure hardware compliance
    block_size = min(preferred_block_size, props["max_threads_per_block"])
    # block_size = ((block_size + 31) // 32) * 32

    # Calculate grid size
    grid_size = (data_size + block_size - 1) // block_size

    # Limit grid size to reasonable bounds
    max_blocks = min(65535, props["multiprocessor_count"] * 4)
    grid_size = min(grid_size, max_blocks)

    return (grid_size, 1, 1), (block_size, 1, 1)


device_props = get_device_properties()
