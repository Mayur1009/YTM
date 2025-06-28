import pathlib
import numpy as np
from scipy.sparse import csr_array

import pycuda.autoinit  # noqa: F401
from pycuda.compiler import SourceModule
from pycuda.driver import Context as ctx  # pyright: ignore[reportAttributeAccessIssue]
from pycuda.driver import mem_alloc, memcpy_dtoh, memcpy_htod  # pyright: ignore[reportAttributeAccessIssue]
from .cuda_utils import kernel_config, get_kernel, device_props


class InputEncoder:
    def __init__(
        self,
        dim: tuple[int, int, int],
        patch_dim: tuple[int, int] | None = None,
        encode_loc: bool = True,
        append_negated: bool = True,
        block_size: int = 128,
    ):
        self.dim = dim
        if patch_dim is None:
            self.patch_dim = (dim[0], dim[1])
        else:
            self.patch_dim = (patch_dim[0], patch_dim[1])
        self.encode_loc = encode_loc
        self.append_negated = append_negated

        if self.encode_loc:
            self.number_of_literals = int(
                self.patch_dim[0] * self.patch_dim[1] * self.dim[2]
                + (self.dim[0] - self.patch_dim[0])
                + (self.dim[1] - self.patch_dim[1])
            )
        else:
            self.number_of_literals = int(self.patch_dim[0] * self.patch_dim[1] * self.dim[2])

        if self.append_negated:
            self.number_of_literals *= 2

        self.number_of_literal_chunks = ((self.number_of_literals - 1) // 32) + 1
        self.number_of_patches = int((self.dim[0] - self.patch_dim[0] + 1) * (self.dim[1] - self.patch_dim[1] + 1))
        self.block_size = block_size

        self.encoded_X_base = self._init_encoded_X_base()
        if not hasattr(self, "kernel_encode_batch"):
            self._init_kernel()

    def _init_kernel(self):
        self.gpu_macro_string = f"""
        #define DIM0 {self.dim[0]}
        #define DIM1 {self.dim[1]}
        #define DIM2 {self.dim[2]}
        #define PATCH_DIM0 {self.patch_dim[0]}
        #define PATCH_DIM1 {self.patch_dim[1]}
        #define PATCHES {self.number_of_patches}
        #define LITERALS {self.number_of_literals}
        #define APPEND_NEGATED {self.append_negated}
        #define ENCODE_LOC {self.encode_loc}
        """
        current_dir = pathlib.Path(__file__).parent
        kernel_str = get_kernel("cuda/encoder.cu", current_dir)
        mod_new_kernel = SourceModule(self.gpu_macro_string + kernel_str, no_extern_c=True)
        self.kernel_encode_batch = mod_new_kernel.get_function("encode_batch")
        self.kernel_encode_batch.prepare("PPPi")

    def encode(self, input: np.ndarray[tuple[int, int], np.dtype[np.uint32]]):
        assert input.ndim == 2, "X must be 2D array (samples, dim0 * dim1 * dim2)."
        X = csr_array(input.astype(np.uint32))
        N = X.shape[0]
        max_uint32 = np.iinfo(np.uint32).max
        max_safe_N = max_uint32 // self.number_of_patches
        if N > max_safe_N:
            raise OverflowError(
                f"X has too many samples ({N}). Maximum of {max_safe_N} samples can be processed with current number_of_patches. Call this method multiple times with smaller batches of X."
            )

        base = np.stack([self.encoded_X_base] * N, axis=0)

        X_indptr_gpu = mem_alloc(X.indptr.nbytes)
        X_indices_gpu = mem_alloc(X.indices.nbytes)
        encoded_X_gpu = mem_alloc(base.nbytes)

        memcpy_htod(X_indptr_gpu, X.indptr)
        memcpy_htod(X_indices_gpu, X.indices)
        memcpy_htod(encoded_X_gpu, base.reshape(-1))
        self.kernel_encode_batch.prepared_call(
            *kernel_config(N * self.number_of_patches, device_props, self.block_size),
            X_indptr_gpu,
            X_indices_gpu,
            encoded_X_gpu,
            np.int32(N),
        )
        ctx.synchronize()

        encoded_X = np.empty((N * self.number_of_patches * self.number_of_literal_chunks), dtype=np.uint32)
        memcpy_dtoh(encoded_X, encoded_X_gpu)
        return encoded_X.reshape((N, self.number_of_patches, self.number_of_literal_chunks))

    def _init_encoded_X_base(self):
        encoded_X = np.zeros((self.number_of_patches, self.number_of_literal_chunks), dtype=np.uint32)
        for patch_coordinate_y in range(self.dim[1] - self.patch_dim[1] + 1):
            for patch_coordinate_x in range(self.dim[0] - self.patch_dim[0] + 1):
                p = patch_coordinate_y * (self.dim[0] - self.patch_dim[0] + 1) + patch_coordinate_x

                if self.append_negated:
                    for k in range(self.number_of_literals // 2, self.number_of_literals):
                        chunk = k // 32
                        pos = k % 32
                        encoded_X[p, chunk] |= 1 << pos

                for y_threshold in range(self.dim[1] - self.patch_dim[1]):
                    patch_pos = y_threshold
                    if patch_coordinate_y > y_threshold:
                        chunk = patch_pos // 32
                        pos = patch_pos % 32
                        encoded_X[p, chunk] |= 1 << pos

                        if self.append_negated:
                            chunk = (patch_pos + self.number_of_literals // 2) // 32
                            pos = (patch_pos + self.number_of_literals // 2) % 32
                            encoded_X[p, chunk] &= ~np.uint32(1 << pos)

                for x_threshold in range(self.dim[0] - self.patch_dim[0]):
                    patch_pos = (self.dim[1] - self.patch_dim[1]) + x_threshold
                    if patch_coordinate_x > x_threshold:
                        chunk = patch_pos // 32
                        pos = patch_pos % 32
                        encoded_X[p, chunk] |= 1 << pos

                        if self.append_negated:
                            chunk = (patch_pos + self.number_of_literals // 2) // 32
                            pos = (patch_pos + self.number_of_literals // 2) % 32
                            encoded_X[p, chunk] &= ~np.uint32(1 << pos)

        return encoded_X
