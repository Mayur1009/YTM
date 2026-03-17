"""MNIST dataset loader"""

import numpy as np
from .base import BaseDataset


class MNISTDataset(BaseDataset):
    """MNIST dataset (binary)"""

    def load_raw_data(self):
        from keras.datasets import mnist

        (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

        # Flatten to (N, 28*28)
        X_train = X_train.reshape(len(X_train), -1)
        X_test = X_test.reshape(len(X_test), -1)

        # Subsample if in test mode
        return self._subsample_if_test_mode(X_train, Y_train, X_test, Y_test)
