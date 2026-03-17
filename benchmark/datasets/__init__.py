"""Dataset loaders for benchmarks"""

from .base import BaseDataset
from .mnist import MNISTDataset
from .fmnist import FashionMNISTDataset
from .octmnist import OCTMNISTDataset
from .cifar10 import CIFAR10Dataset

__all__ = [
    "BaseDataset",
    "MNISTDataset",
    "FashionMNISTDataset",
    "OCTMNISTDataset",
    "CIFAR10Dataset",
    "get_dataset",
]


def get_dataset(name: str, config, test_mode: bool = False, test_samples: int = 100) -> BaseDataset:
    """Factory function to get dataset by name

    Args:
        name: Dataset name (mnist, fmnist, octmnist, cifar10)
        config: Dataset configuration
        test_mode: If True, use small data subset for quick testing
        test_samples: Number of training samples in test mode (default 100)

    Returns:
        Dataset instance
    """
    datasets = {
        "mnist": MNISTDataset,
        "fmnist": FashionMNISTDataset,
        "fashionmnist": FashionMNISTDataset,
        "octmnist": OCTMNISTDataset,
        "cifar10": CIFAR10Dataset,
    }

    name_lower = name.lower()
    if name_lower not in datasets:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(datasets.keys())}")

    return datasets[name_lower](config, test_mode=test_mode, test_samples=test_samples)
