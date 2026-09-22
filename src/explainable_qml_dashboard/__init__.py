"""Explainable quantum machine-learning dashboard."""

from .config import ExperimentConfig
from .experiment import run_experiment

__all__ = ["ExperimentConfig", "run_experiment"]
__version__ = "1.0.0"
