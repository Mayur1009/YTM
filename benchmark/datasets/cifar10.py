"""CIFAR-10 dataset loader"""

import numpy as np
from .base import BaseDataset


class CIFAR10Dataset(BaseDataset):
    """CIFAR-10 dataset (multibin, RGB)"""

    def load_raw_data(self):
        from keras.datasets import cifar10

        (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

        # Flatten labels
        Y_train = Y_train.reshape(-1)
        Y_test = Y_test.reshape(-1)

        # Subsample if in test mode
        X_train, Y_train, X_test, Y_test = self._subsample_if_test_mode(X_train, Y_train, X_test, Y_test)

        # Keep as (N, 32, 32, 3) for now
        return X_train, Y_train, X_test, Y_test
