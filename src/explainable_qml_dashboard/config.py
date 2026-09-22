"""Validated experiment configuration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for one deterministic benchmark run."""

    random_seed: int = 42
    test_fraction: float = 0.2
    validation_fraction: float = 0.2
    qubits: int = 4
    layers: int = 2
    logit_scale: float = 2.0
    learning_rate: float = 0.08
    epochs: int = 45
    patience: int = 10
    min_delta: float = 1e-4
    gradient_clip: float = 5.0
    saliency_step: float = 1e-3
    counterfactual_grid_points: int = 121
    output_root: str = "artifacts"

    def __post_init__(self) -> None:
        if self.qubits != 4:
            raise ValueError("This Iris reference architecture requires exactly four qubits.")
        if self.layers < 1:
            raise ValueError("layers must be at least 1")
        if self.epochs < 1 or self.patience < 1:
            raise ValueError("epochs and patience must be positive")
        if not 0 < self.test_fraction < 0.5:
            raise ValueError("test_fraction must be between 0 and 0.5")
        if not 0 < self.validation_fraction < 0.5:
            raise ValueError("validation_fraction must be between 0 and 0.5")
        if self.test_fraction + self.validation_fraction >= 0.8:
            raise ValueError("training fraction must be at least 0.2")
        if self.learning_rate <= 0 or self.logit_scale <= 0:
            raise ValueError("learning_rate and logit_scale must be positive")
        if self.saliency_step <= 0:
            raise ValueError("saliency_step must be positive")
        if self.counterfactual_grid_points < 3:
            raise ValueError("counterfactual_grid_points must be at least 3")

    @property
    def parameter_count(self) -> int:
        """Number of trainable RY/RZ angles."""

        return self.layers * self.qubits * 2

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable mapping."""

        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> ExperimentConfig:
        """Construct from a mapping and reject unknown keys."""

        known = set(cls.__dataclass_fields__)
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        return cls(**values)

    @classmethod
    def from_json(cls, path: str | Path) -> ExperimentConfig:
        """Load configuration from JSON."""

        with Path(path).open(encoding="utf-8") as handle:
            values = json.load(handle)
        if not isinstance(values, dict):
            raise ValueError("Configuration root must be a JSON object.")
        return cls.from_dict(values)
