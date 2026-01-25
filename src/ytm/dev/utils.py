import numpy as np

def float_or_list_to_array(x: float | list[float], size: int):
    if isinstance(x, float) or isinstance(x, int):
        return np.array([x] * size, dtype=np.float64)
    elif isinstance(x, list):
        assert len(x) == size, f"List must be of size {size}."
        return np.array(x, dtype=np.float64)
    else:
        raise ValueError("x must be a float or a list of floats.")

def read_file(file, current_dir):
    path = current_dir.joinpath(file)
    with path.open("r") as f:
        ker = f.read()
    return ker

