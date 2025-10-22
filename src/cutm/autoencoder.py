from typing import Unpack
import numpy as np
from .base import BaseTMOptArgs, FitOptArgs, BaseTM
from .cuda_utils import device_props, kernel_config
from pycuda.driver import mem_alloc, memcpy_dtoh, memcpy_htod  # pyright: ignore[reportAttributeAccessIssue]
from tqdm import tqdm


class AETM(BaseTM):
    def __init__(
        self,
        number_of_clauses_per_class: int,
        T: int,
        s: float,
        dim: tuple[int, int, int],
        active_output: np.ndarray[tuple[int], np.dtype[np.uint32]],
        accumulation: int = 1,
        **opt_args: Unpack[BaseTMOptArgs],
    ):
        # Input validation
        assert active_output.ndim == 1, "active_output must be a 1D array"
        assert np.min(active_output) >= 0 and active_output.dtype == np.uint32, (
            "active_output must contain non-negative integers"
        )
        assert len(active_output) >= 1, "active_output must contain at least one elements"
        assert accumulation >= 1, "accumulation must be at least 1"

        # Active output is the indices of the literasl to be used as output
        self.active_output = np.array(active_output, dtype=np.uint32)
        n_classes = len(self.active_output)
        super().__init__(
            number_of_clauses_per_class=number_of_clauses_per_class, T=T, s=s, dim=dim, n_classes=n_classes, **opt_args
        )
        # self._kernel_init()
        self.accumulation = accumulation

        # For saving and loading the model
        self.init_args["active_output"] = active_output
        self.init_args["accumulation"] = accumulation

    def _samids_per_output(self, Y: np.ndarray):
        # For each output in active output, get the indices of all the sampels that has that output and those that do not.
        # This is going to be a list of variable lenghts. But we need to convert it to a array with fixed length for GPU processing.
        # Pad them with -1, and keep trach of actual lengths.

        inds_per_output = []
        not_inds_per_output = []

        for o in range(len(self.active_output)):
            inds = np.argwhere(Y[:, o] == 1).ravel().tolist()
            not_inds = np.argwhere(Y[:, o] == 0).ravel().tolist()
            inds_per_output.append(inds)
            not_inds_per_output.append(not_inds)

        max_len = max(
            max([len(inds) for inds in inds_per_output]), max([len(not_inds) for not_inds in not_inds_per_output])
        )

        inds_arr = np.full((len(self.active_output), max_len), -1, dtype=np.int32)
        not_inds_arr = np.full((len(self.active_output), max_len), -1, dtype=np.int32)
        inds_lens = np.zeros((len(self.active_output),), dtype=np.uint32)
        not_inds_lens = np.zeros((len(self.active_output),), dtype=np.uint32)

        for o in range(len(self.active_output)):
            inds = inds_per_output[o]
            not_inds = not_inds_per_output[o]
            inds_arr[o, : len(inds)] = inds
            not_inds_arr[o, : len(not_inds)] = not_inds
            inds_lens[o] = len(inds)
            not_inds_lens[o] = len(not_inds)

        return inds_arr, not_inds_arr, inds_lens, not_inds_lens

    def make_XY(self, X: np.ndarray, num_examples_per_output: int):
        newN = num_examples_per_output * len(self.active_output)

        Y = X[:, self.active_output]
        inds_per_output, not_inds_per_output, inds_lens, not_inds_lens = self._samids_per_output(Y)

        encoded_X = self.encode(X)

        # Allocate GPU memory
        encoded_X_gpu = mem_alloc(encoded_X.nbytes)
        active_outputs_gpu = mem_alloc(self.active_output.nbytes)
        inds_per_output_gpu = mem_alloc(inds_per_output.nbytes)  # int32
        not_inds_per_output_gpu = mem_alloc(not_inds_per_output.nbytes)  # int32
        inds_lens_gpu = mem_alloc(inds_lens.nbytes)  # uint32
        not_inds_lens_gpu = mem_alloc(not_inds_lens.nbytes)  # uint32
        # Allocate output memory
        accumulated_X_gpu = mem_alloc(newN * self.number_of_patches * self.number_of_literal_chunks * 4)  # uint32
        targets_gpu = mem_alloc(newN * len(self.active_output) * 4)  # int32

        # Copy data to GPU
        memcpy_htod(encoded_X_gpu, encoded_X)
        memcpy_htod(active_outputs_gpu, self.active_output)
        memcpy_htod(inds_per_output_gpu, inds_per_output)
        memcpy_htod(not_inds_per_output_gpu, not_inds_per_output)
        memcpy_htod(inds_lens_gpu, inds_lens)
        memcpy_htod(not_inds_lens_gpu, not_inds_lens)

        config_acc = kernel_config(newN, device_props, self.block_size, self.grid_size)

        self.kernel_accumulate_examples.prepared_call(
            *config_acc,
            self.rng_gpu.state,
            encoded_X_gpu,
            active_outputs_gpu,
            inds_per_output_gpu,
            not_inds_per_output_gpu,
            np.int32(inds_per_output.shape[1]),
            inds_lens_gpu,
            not_inds_lens_gpu,
            np.uint32(self.accumulation),
            np.uint32(newN),
            accumulated_X_gpu,
            targets_gpu,
        )

        # Copy results back to host
        accumulated_X = np.empty((newN, self.number_of_patches, self.number_of_literal_chunks), dtype=np.uint32)
        targets = np.empty((newN, len(self.active_output)), dtype=np.int32)
        memcpy_dtoh(accumulated_X, accumulated_X_gpu)
        memcpy_dtoh(targets, targets_gpu)

        return accumulated_X, targets

    def fit(self, X: np.ndarray, **opt_args: Unpack[FitOptArgs]):
        accumulated_X, targets = self.make_XY(X, self.accumulation)
        N = accumulated_X.shape[0]

        args = self._validate_fit_args(**opt_args)
        gpu_buffers = self._fit_allocate_gpu(args, accumulated_X, targets)

        # Kernel configurations
        kconfs = {
            "config_n_clauses": kernel_config(
                self.number_of_clauses, device_props, args["block_size"], args["grid_size"]
            ),
            "config_patchwise": kernel_config(
                self.number_of_clauses * self.number_of_patches, device_props, args["block_size"], args["grid_size"]
            ),
            "config_outputs": kernel_config(
                self.number_of_outputs, device_props, args["block_size"], args["grid_size"]
            ),
        }

        pbar = tqdm(range(N), desc="Fitting Batch", leave=False, dynamic_ncols=True)
        for e in pbar:
            # If all the targets are zero, then there is nothing to learn, so skip.
            if np.all(targets[e, :] == 0):
                continue

            # Freeze the current target literal
            self.freeze_literals(self.active_output[np.argwhere(targets[e, :] != 0).ravel()].tolist())

            # Now we can do the same fitting as in every other TMclassifier
            self._fit_sample(gpu_buffers, e, kconfs)

        self.freeze_literals([])  # Unfreeze all literals after training
