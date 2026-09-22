"""Local and global explanations for the variational quantum classifier."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import AnglePreprocessor
from .quantum import apply_feature_map, bloch_vectors, initial_states, trace_circuit
from .qvc import VariationalQuantumClassifier, softmax


def encoding_frame(
    raw_values: np.ndarray,
    angle_values: np.ndarray,
    feature_names: tuple[str, ...],
) -> pd.DataFrame:
    """Pair raw measurements with the exact rotation angles used by the circuit."""

    states = initial_states(1)
    apply_feature_map(states, np.asarray(angle_values).reshape(1, 4))
    bloch = bloch_vectors(states)[0]
    return pd.DataFrame(
        {
            "feature": feature_names,
            "raw_value": np.asarray(raw_values, dtype=float),
            "angle_radians": np.asarray(angle_values, dtype=float),
            "angle_fraction_of_pi": np.asarray(angle_values, dtype=float) / np.pi,
            "bloch_x_after_encoding": bloch[:, 0],
            "bloch_y_after_encoding": bloch[:, 1],
            "bloch_z_after_encoding": bloch[:, 2],
        }
    )


def circuit_trace_frames(
    angle_values: np.ndarray, model: VariationalQuantumClassifier
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return layer diagnostics and basis-state probabilities for one sample."""

    snapshots = trace_circuit(angle_values, model.parameters)
    trace_rows: list[dict[str, float | str]] = []
    basis_rows: list[dict[str, float | str]] = []
    for snapshot in snapshots:
        probabilities = np.abs(snapshot.state) ** 2
        prediction_probabilities = softmax(
            model.logit_scale * snapshot.z_expectations.reshape(1, -1)
        )[0]
        trace_rows.append(
            {
                "stage": snapshot.stage,
                "z_q0": float(snapshot.z_expectations[0]),
                "z_q1": float(snapshot.z_expectations[1]),
                "z_q2": float(snapshot.z_expectations[2]),
                "probability_setosa": float(prediction_probabilities[0]),
                "probability_versicolor": float(prediction_probabilities[1]),
                "probability_virginica": float(prediction_probabilities[2]),
                "mean_single_qubit_entropy": snapshot.mean_entanglement_entropy,
            }
        )
        for index, probability in enumerate(probabilities):
            basis_rows.append(
                {
                    "stage": snapshot.stage,
                    "basis_state": format(index, "04b"),
                    "probability": float(probability),
                }
            )
    return pd.DataFrame(trace_rows), pd.DataFrame(basis_rows)


def local_input_saliency(
    model: VariationalQuantumClassifier,
    angle_values: np.ndarray,
    feature_names: tuple[str, ...],
    *,
    step: float = 1e-3,
    target_class: int | None = None,
) -> pd.DataFrame:
    """Central-difference probability sensitivity with respect to each input angle."""

    sample = np.asarray(angle_values, dtype=float).reshape(4)
    base_probabilities = model.predict_proba(sample[None, :])[0]
    explained_class = int(np.argmax(base_probabilities)) if target_class is None else target_class
    rows = []
    for feature_index, feature_name in enumerate(feature_names):
        plus = sample.copy()
        minus = sample.copy()
        plus[feature_index] = min(np.pi, plus[feature_index] + step)
        minus[feature_index] = max(0.0, minus[feature_index] - step)
        denominator = plus[feature_index] - minus[feature_index]
        plus_probability = model.predict_proba(plus[None, :])[0, explained_class]
        minus_probability = model.predict_proba(minus[None, :])[0, explained_class]
        derivative = (plus_probability - minus_probability) / denominator
        rows.append(
            {
                "feature": feature_name,
                "target_class": explained_class,
                "target_probability": float(base_probabilities[explained_class]),
                "signed_sensitivity_per_radian": float(derivative),
                "absolute_sensitivity_per_radian": float(abs(derivative)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "absolute_sensitivity_per_radian", ascending=False, ignore_index=True
    )


def global_input_saliency(
    model: VariationalQuantumClassifier,
    x_angles: np.ndarray,
    feature_names: tuple[str, ...],
    *,
    step: float = 1e-3,
) -> pd.DataFrame:
    """Aggregate absolute local sensitivities over a frozen evaluation partition."""

    matrices = []
    for sample in x_angles:
        local = local_input_saliency(model, sample, feature_names, step=step)
        matrices.append(local.set_index("feature")["absolute_sensitivity_per_radian"])
    combined = pd.concat(matrices, axis=1).T
    result = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_absolute_sensitivity": [float(combined[name].mean()) for name in feature_names],
            "std_absolute_sensitivity": [
                float(combined[name].std(ddof=0)) for name in feature_names
            ],
        }
    )
    return result.sort_values("mean_absolute_sensitivity", ascending=False, ignore_index=True)


def nearest_single_feature_counterfactual(
    model: VariationalQuantumClassifier,
    angle_values: np.ndarray,
    preprocessor: AnglePreprocessor,
    *,
    grid_points: int = 121,
) -> dict[str, object]:
    """Find the closest one-feature angle change that flips the quantum prediction."""

    sample = np.asarray(angle_values, dtype=float).reshape(4)
    original_prediction = int(model.predict(sample[None, :])[0])
    original_raw = preprocessor.inverse_transform(sample[None, :])[0]
    best: dict[str, object] | None = None
    for feature_index, feature_name in enumerate(preprocessor.feature_names):
        candidates = np.repeat(sample[None, :], grid_points, axis=0)
        candidates[:, feature_index] = np.linspace(0.0, np.pi, grid_points)
        predictions = model.predict(candidates)
        flipped = np.flatnonzero(predictions != original_prediction)
        for candidate_index in flipped:
            candidate = candidates[candidate_index]
            angle_delta = float(abs(candidate[feature_index] - sample[feature_index]))
            if best is None or angle_delta < float(best["angle_delta"]):
                candidate_raw = preprocessor.inverse_transform(candidate[None, :])[0]
                probability = model.predict_proba(candidate[None, :])[0]
                best = {
                    "found": True,
                    "feature": feature_name,
                    "feature_index": feature_index,
                    "original_prediction": original_prediction,
                    "counterfactual_prediction": int(np.argmax(probability)),
                    "original_angle": float(sample[feature_index]),
                    "counterfactual_angle": float(candidate[feature_index]),
                    "angle_delta": angle_delta,
                    "original_raw_value": float(original_raw[feature_index]),
                    "counterfactual_raw_value": float(candidate_raw[feature_index]),
                    "counterfactual_confidence": float(np.max(probability)),
                }
    if best is None:
        return {
            "found": False,
            "original_prediction": original_prediction,
            "reason": "No single-feature flip was found inside the train-fitted angle range.",
        }
    return best
