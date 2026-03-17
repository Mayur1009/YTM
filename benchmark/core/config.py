"""Configuration dataclasses for benchmarks"""

from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any, Optional
import json


@dataclass
class TMHyperparameters:
    """TM hyperparameters"""

    n_clauses: int
    T: int
    s: float
    patch_dim: Tuple[int, int]
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TMHyperparameters":
        return cls(
            n_clauses=data["n_clauses"],
            T=data["T"],
            s=data["s"],
            patch_dim=tuple(data["patch_dim"]),
            seed=data.get("seed", 42),
        )


@dataclass
class DatasetConfig:
    """Dataset configuration"""

    name: str
    shape: Tuple[int, int, int]
    n_classes: int
    type: str  # "binary" or "multibin"
    bins: List[int]
    threshold: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetConfig":
        return cls(
            name=data["name"],
            shape=tuple(data["shape"]),
            n_classes=data["n_classes"],
            type=data["type"],
            bins=data["bins"],
            threshold=data.get("threshold"),
        )


@dataclass
class BenchmarkConfig:
    """Complete benchmark configuration"""

    dataset: DatasetConfig
    hyperparams: TMHyperparameters
    n_runs: int = 5
    n_epochs: int = 10
    device: str = "cpu"
    n_threads: int = 32

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset.to_dict(),
            "hyperparams": self.hyperparams.to_dict(),
            "n_runs": self.n_runs,
            "n_epochs": self.n_epochs,
            "device": self.device,
            "n_threads": self.n_threads,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkConfig":
        return cls(
            dataset=DatasetConfig.from_dict(data["dataset"]),
            hyperparams=TMHyperparameters.from_dict(data["hyperparams"]),
            n_runs=data.get("n_runs", 5),
            n_epochs=data.get("n_epochs", 10),
            device=data.get("device", "cpu"),
            n_threads=data.get("n_threads", 32),
        )

    @classmethod
    def load(cls, filepath: str) -> "BenchmarkConfig":
        """Load configuration from JSON file"""
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def save(self, filepath: str):
        """Save configuration to JSON file"""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class BenchmarkResult:
    """Single benchmark run result"""

    dataset: str
    device: str
    implementation: str  # "binary" or "continuous"
    bins: int
    run: int
    epochs: int
    metrics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkResult":
        return cls(**data)

    def save(self, filepath: str):
        """Save result to JSON file"""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "BenchmarkResult":
        """Load result from JSON file"""
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)
