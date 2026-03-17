"""Base dataset class"""

from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class BaseDataset(ABC):
    """Base class for all benchmark datasets"""

    def __init__(self, config, test_mode=False, test_samples=100):
        self.config = config
        self.name = config.name
        self.test_mode = test_mode
        self.test_samples = test_samples

    @abstractmethod
    def load_raw_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load raw dataset

        Returns:
            X_train, Y_train, X_test, Y_test (raw, unprocessed)
        """
        pass

    def _subsample_if_test_mode(self, X_train, Y_train, X_test, Y_test):
        """Subsample data if in test mode"""
        if not self.test_mode:
            return X_train, Y_train, X_test, Y_test

        # Subsample training data
        n_train = min(self.test_samples, len(X_train))
        train_indices = np.random.RandomState(42).choice(len(X_train), n_train, replace=False)
        X_train = X_train[train_indices]
        Y_train = Y_train[train_indices]

        # Subsample test data (smaller subset for faster testing)
        n_test = min(self.test_samples // 5, len(X_test))  # 20 samples for test
        test_indices = np.random.RandomState(43).choice(len(X_test), n_test, replace=False)
        X_test = X_test[test_indices]
        Y_test = Y_test[test_indices]

        return X_train, Y_train, X_test, Y_test

    def prepare_binary(self, X_train: np.ndarray, X_test: np.ndarray, bins: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for binary TM

        For multibin data with bins > 1, this uses thermometer encoding
        which creates `bins` channels per feature.

        Args:
            X_train: Raw training data
            X_test: Raw test data
            bins: Number of bins for thermometer encoding

        Returns:
            X_train_bin, X_test_bin: Binary encoded data
        """
        if self.config.type == "binary":
            # Simple threshold-based binarization
            threshold = self.config.threshold
            X_train_bin = np.where(X_train > threshold, 1, 0)
            X_test_bin = np.where(X_test > threshold, 1, 0)
            return X_train_bin.astype(np.int8), X_test_bin.astype(np.int8)
        else:
            # Multibin: use thermometer encoding
            return self._thermometer_encode(X_train, X_test, bins)

    def _thermometer_encode(self, X_train: np.ndarray, X_test: np.ndarray, bins: int) -> Tuple[np.ndarray, np.ndarray]:
        """Apply thermometer encoding to create binary features"""
        # Normalize to [0, 255] if needed
        if X_train.max() <= 1.0:
            X_train = (X_train * 255).astype(np.uint8)
            X_test = (X_test * 255).astype(np.uint8)

        # Create thermometer encoding
        out_train = np.zeros((*X_train.shape, bins), dtype=np.int8)
        out_test = np.zeros((*X_test.shape, bins), dtype=np.int8)

        for j in range(bins):
            threshold = (j + 1) * 255 / (bins + 1)
            out_train[..., j] = (X_train >= threshold).astype(np.int8)
            out_test[..., j] = (X_test >= threshold).astype(np.int8)

        # Reshape to flatten spatial + channel dimensions
        n_train, *spatial_dims = X_train.shape
        n_test = X_test.shape[0]

        out_train = out_train.reshape(n_train, -1)
        out_test = out_test.reshape(n_test, -1)

        return out_train, out_test

    def prepare_continuous(
        self, X_train: np.ndarray, X_test: np.ndarray, bins: int
    ) -> Tuple[np.ndarray, np.ndarray, int, int]:
        """
        Prepare data for continuous TM

        Discretizes the data to `bins` levels.

        Args:
            X_train: Raw training data
            X_test: Raw test data
            bins: Number of discrete levels

        Returns:
            X_train_cont, X_test_cont, feat_mins, feat_maxs
        """
        if self.config.type == "binary":
            # For binary data, just threshold
            threshold = self.config.threshold
            X_train_cont = np.where(X_train > threshold, 1, 0).astype(np.int32)
            X_test_cont = np.where(X_test > threshold, 1, 0).astype(np.int32)
            return X_train_cont, X_test_cont, 0, 1
        else:
            # Discretize to bins levels
            # Normalize to [0, 255] first
            if X_train.max() <= 1.0:
                X_train = (X_train * 255).astype(np.float32)
                X_test = (X_test * 255).astype(np.float32)

            # Discretize to [0, bins]
            X_train_cont = (bins * X_train / 255.0).astype(np.int32)
            X_test_cont = (bins * X_test / 255.0).astype(np.int32)

            return X_train_cont, X_test_cont, 0, bins
