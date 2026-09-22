"""Exact-statevector variational quantum classifier with analytic gradients."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from .quantum import simulate, z_expectations


def softmax(logits: np.ndarray) -> np.ndarray:
    """Stable row-wise softmax."""

    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=1, keepdims=True)


def parameter_names(layers: int) -> list[str]:
    """Stable parameter labels in flattened tensor order."""

    return [
        f"layer_{layer + 1}.q{qubit}.{gate.lower()}"
        for layer in range(layers)
        for qubit in range(4)
        for gate in ("RY", "RZ")
    ]


@dataclass(frozen=True)
class TrainingResult:
    history: pd.DataFrame
    gradients: pd.DataFrame
    best_epoch: int
    stopped_epoch: int


class VariationalQuantumClassifier:
    """Three-class classifier using exact Z expectations as trainable logits."""

    def __init__(
        self,
        *,
        layers: int = 2,
        logit_scale: float = 2.0,
        random_seed: int = 42,
        parameters: np.ndarray | None = None,
    ) -> None:
        if layers < 1:
            raise ValueError("layers must be positive")
        self.layers = layers
        self.logit_scale = float(logit_scale)
        self.random_seed = int(random_seed)
        if parameters is None:
            rng = np.random.default_rng(self.random_seed)
            self.parameters = rng.normal(0.0, 0.12, size=(layers, 4, 2))
        else:
            values = np.asarray(parameters, dtype=float)
            if values.shape != (layers, 4, 2):
                raise ValueError(f"Expected parameters with shape {(layers, 4, 2)}")
            self.parameters = values.copy()
        self.training_result_: TrainingResult | None = None

    @property
    def parameter_count(self) -> int:
        return int(self.parameters.size)

    def expectations(
        self, x_angles: np.ndarray, parameters: np.ndarray | None = None
    ) -> np.ndarray:
        values = self.parameters if parameters is None else parameters
        return z_expectations(simulate(x_angles, values))

    def predict_proba(
        self, x_angles: np.ndarray, parameters: np.ndarray | None = None
    ) -> np.ndarray:
        return softmax(self.logit_scale * self.expectations(x_angles, parameters))

    def predict(self, x_angles: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(x_angles), axis=1)

    def loss(self, x_angles: np.ndarray, y: np.ndarray) -> float:
        return float(log_loss(y, self.predict_proba(x_angles), labels=[0, 1, 2]))

    def parameter_shift_gradient(self, x_angles: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Exact cross-entropy gradient using two shifted circuits per parameter."""

        probabilities = self.predict_proba(x_angles)
        one_hot = np.eye(3)[np.asarray(y, dtype=int)]
        loss_to_logits = (probabilities - one_hot) / len(y)
        gradient = np.zeros_like(self.parameters)
        flat_gradient = gradient.ravel()
        for flat_index in range(self.parameter_count):
            plus = self.parameters.copy().ravel()
            minus = self.parameters.copy().ravel()
            plus[flat_index] += np.pi / 2.0
            minus[flat_index] -= np.pi / 2.0
            plus_z = self.expectations(x_angles, plus.reshape(self.parameters.shape))
            minus_z = self.expectations(x_angles, minus.reshape(self.parameters.shape))
            expectation_derivative = 0.5 * (plus_z - minus_z)
            logit_derivative = self.logit_scale * expectation_derivative
            flat_gradient[flat_index] = np.sum(loss_to_logits * logit_derivative)
        return gradient

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_validation: np.ndarray,
        y_validation: np.ndarray,
        *,
        epochs: int = 45,
        learning_rate: float = 0.08,
        patience: int = 10,
        min_delta: float = 1e-4,
        gradient_clip: float = 5.0,
    ) -> TrainingResult:
        """Train with full-batch Adam and validation-loss early stopping."""

        first_moment = np.zeros_like(self.parameters)
        second_moment = np.zeros_like(self.parameters)
        best_parameters = self.parameters.copy()
        best_loss = float("inf")
        best_epoch = 0
        stale_epochs = 0
        history_rows: list[dict[str, float | int]] = []
        gradient_rows: list[dict[str, float | int | str]] = []
        names = parameter_names(self.layers)
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8

        for epoch in range(1, epochs + 1):
            gradient = self.parameter_shift_gradient(x_train, y_train)
            raw_norm = float(np.linalg.norm(gradient))
            if raw_norm > gradient_clip:
                gradient *= gradient_clip / raw_norm
            first_moment = beta1 * first_moment + (1.0 - beta1) * gradient
            second_moment = beta2 * second_moment + (1.0 - beta2) * gradient**2
            corrected_first = first_moment / (1.0 - beta1**epoch)
            corrected_second = second_moment / (1.0 - beta2**epoch)
            self.parameters -= (
                learning_rate * corrected_first / (np.sqrt(corrected_second) + epsilon)
            )

            train_probabilities = self.predict_proba(x_train)
            validation_probabilities = self.predict_proba(x_validation)
            train_loss = float(log_loss(y_train, train_probabilities, labels=[0, 1, 2]))
            validation_loss = float(
                log_loss(y_validation, validation_probabilities, labels=[0, 1, 2])
            )
            history_rows.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                    "train_accuracy": float(
                        accuracy_score(y_train, np.argmax(train_probabilities, axis=1))
                    ),
                    "validation_accuracy": float(
                        accuracy_score(y_validation, np.argmax(validation_probabilities, axis=1))
                    ),
                    "gradient_norm": raw_norm,
                    "parameter_norm": float(np.linalg.norm(self.parameters)),
                }
            )
            for name, value in zip(names, gradient.ravel(), strict=True):
                gradient_rows.append({"epoch": epoch, "parameter": name, "gradient": float(value)})

            if validation_loss < best_loss - min_delta:
                best_loss = validation_loss
                best_epoch = epoch
                best_parameters = self.parameters.copy()
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    break

        self.parameters = best_parameters
        result = TrainingResult(
            history=pd.DataFrame(history_rows),
            gradients=pd.DataFrame(gradient_rows),
            best_epoch=best_epoch,
            stopped_epoch=int(history_rows[-1]["epoch"]),
        )
        self.training_result_ = result
        return result

    def to_dict(self) -> dict[str, object]:
        return {
            "model_type": "exact_statevector_variational_quantum_classifier",
            "layers": self.layers,
            "qubits": 4,
            "classes": 3,
            "logit_scale": self.logit_scale,
            "random_seed": self.random_seed,
            "parameters": self.parameters.tolist(),
            "parameter_names": parameter_names(self.layers),
        }

    def save(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)
            handle.write("\n")

    @classmethod
    def load(cls, path: str | Path) -> VariationalQuantumClassifier:
        with Path(path).open(encoding="utf-8") as handle:
            values = json.load(handle)
        if values.get("model_type") != "exact_statevector_variational_quantum_classifier":
            raise ValueError("Unsupported quantum model format.")
        return cls(
            layers=int(values["layers"]),
            logit_scale=float(values["logit_scale"]),
            random_seed=int(values["random_seed"]),
            parameters=np.asarray(values["parameters"], dtype=float),
        )
