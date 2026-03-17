"""OCT-MNIST dataset loader"""

import numpy as np
from .base import BaseDataset


class OCTMNISTDataset(BaseDataset):
    """OCT-MNIST dataset (multibin, medical imaging)"""

    def load_raw_data(self):
        from medmnist.dataset import OCTMNIST

        # Load all splits
        train_dataset = OCTMNIST(split="train", download=True)
        val_dataset = OCTMNIST(split="val", download=True)
        test_dataset = OCTMNIST(split="test", download=True)

        # Combine train and val for training
        X_train = np.vstack([train_dataset.imgs, val_dataset.imgs])
        Y_train = np.hstack([train_dataset.labels.squeeze(), val_dataset.labels.squeeze()])

        X_test = test_dataset.imgs
        Y_test = test_dataset.labels.squeeze()

        # Subsample if in test mode
        X_train, Y_train, X_test, Y_test = self._subsample_if_test_mode(X_train, Y_train, X_test, Y_test)

        # Keep as (N, 28, 28) for now
        return X_train, Y_train, X_test, Y_test
