# Yet another Tsetlin Machine

A Python implementation of the Tsetlin Machine, running on CPU and CUDA.

- Optimized for large datasets, convolution, and thermometer-encoded discretized inputs.
- Binary, multi-class, multi-output, and regression TM classes.
- Binarizer module.
- Pickle support for model saving/loading.
- Local (WAC) and global (WIC) interpretation helpers.
- Timer, Profiler, and other small utils.

Implemented TM variants:
- Vanilla TM
- Weighted TM
- Regression TM
- Coalesced TM
- Convolutional TM
- \*\*\*\*\*\* TM

## Install

Requires Python >= 3.11.

- Using `pip`:
    ```bash
    pip install "ytm @ git+https://gitlab.com/Mayur1009/YTM.git"          # CPU
    pip install "ytm[cuda] @ git+https://gitlab.com/Mayur1009/YTM.git"    # CPU + CUDA
    ```

-  Using [pixi](https://pixi.sh) (*Recommended*):
    ```bash
    pixi add clang llvm-openmp                                                  # C compiler + OpenMP for CPU
    pixi add --pypi ytm --git "https://gitlab.com/Mayur1009/YTM.git"            # CPU only
    pixi add --pypi "ytm[cuda]" --git "https://gitlab.com/Mayur1009/YTM.git"    # CPU+CUDA
    ```

**Note:**
- **CPU**: 
  - C kernels are compiled at runtime, so `clang` or `gcc` must be on `PATH`. **OpenMP** (`libomp`/`libgomp`) is needed for multithreading, without it training runs single-threaded with a warning.

  - `clang` or `gcc` can be installed using:
    ```bash
    # Arch
    sudo pacman -S clang # or gcc

    # Ubuntu
    sudo apt install clang # or gcc

    # macOS should have clang by default, if not:
    xcode-select --install

    # Pixi, if using pixi to manage python environment
    pixi add clang
    ```

  - OpenMP can be installed using:
    ```bash
    # Arch (clang)
    sudo pacman -S openmp

    # Ubuntu (clang)
    sudo apt install libomp-dev

    # macOS (Homebrew, https://brew.sh)
    brew install libomp

    # Pixi
    pixi add llvm-openmp
    ```

- **CUDA**: Optional, **not** installed by default.
  - Use the `cuda` extra, it installs `cupy-cuda12x[ctk]`, which bundles the CUDA runtime and NVRTC (kernels are compiled at runtime), so no system CUDA toolkit is needed.
  - If facing problems with CUDA/cupy, don't use the `cuda` extra, and follow the [CuPy installation instructions](https://docs.cupy.dev/en/stable/install.html) to get it working in the environment first.
  - Requires an NVIDIA GPU with a driver supporting CUDA 12 (check `CUDA Version` in `nvidia-smi`).

## Local Development

The environment needed is defined in `pyproject.toml`.
```bash
git clone https://gitlab.com/Mayur1009/YTM.git && cd YTM
pixi install              # CPU, includes clang and llvm-openmp
pixi install -e cuda      # CUDA
```

### Tests

```bash
pixi run pytest tests                     # CPU only
pixi run -e cuda pytest tests             # CPU + CUDA (CUDA tests skip if no usable GPU)
pixi run pytest tests -m statistical      # RNG and distribution/frequency tests
pixi run pytest tests -m sanitize         # ASan/UBSan, plus compute-sanitizer when CUDA is usable
pixi run pytest tests -m benchmark        # full dataset benchmarks
```

## Inspired by

This implementation is inspired by:
- [TMU](https://github.com/cair/tmu)
- [PySparseCoalescedTsetlinMachineCUDA](https://github.com/cair/PySparseCoalescedTsetlinMachineCUDA)
