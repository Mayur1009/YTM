"""Fashion-MNIST dataset loader"""

import numpy as np
from .base import BaseDataset


class FashionMNISTDataset(BaseDataset):
    """Fashion-MNIST dataset (multibin)"""

    def load_raw_data(self):
        from keras.datasets import fashion_mnist

        (X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()

        # Subsample if in test mode
        X_train, Y_train, X_test, Y_test = self._subsample_if_test_mode(X_train, Y_train, X_test, Y_test)

        # Keep as (N, 28, 28) for now, will be flattened in prepare methods
        return X_train, Y_train, X_test, Y_test
