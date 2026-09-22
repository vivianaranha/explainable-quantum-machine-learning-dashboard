"""Safe schema validation and saved-model inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .data import FEATURE_NAMES, TARGET_NAMES, AnglePreprocessor
from .explain import encoding_frame, local_input_saliency, nearest_single_feature_counterfactual
from .qvc import VariationalQuantumClassifier

MAX_UPLOAD_ROWS = 10_000


@dataclass(frozen=True)
class InferenceBundle:
    preprocessor: AnglePreprocessor
    quantum_model: VariationalQuantumClassifier
    classical_models: dict[str, object]

    @property
    def model_names(self) -> tuple[str, ...]:
        return ("variational_quantum_classifier", *self.classical_models.keys())


def load_inference_bundle(run_directory: str | Path) -> InferenceBundle:
    """Load models produced by this repository.

    Joblib is pickle-based. Only point this function at artifacts you generated or otherwise trust.
    """

    root = Path(run_directory)
    preprocessor = joblib.load(root / "models" / "preprocessor.joblib")
    classical_models = joblib.load(root / "models" / "classical_baselines.joblib")
    if not isinstance(preprocessor, AnglePreprocessor):
        raise ValueError("The saved preprocessor has an unexpected type.")
    if not isinstance(classical_models, dict):
        raise ValueError("The saved classical model bundle has an unexpected type.")
    quantum_model = VariationalQuantumClassifier.load(root / "models" / "quantum_model.json")
    return InferenceBundle(preprocessor, quantum_model, classical_models)


def validate_input_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Enforce the exact four-column Iris inference contract."""

    if len(frame) == 0:
        raise ValueError("Input CSV has no rows.")
    if len(frame) > MAX_UPLOAD_ROWS:
        raise ValueError(f"Input exceeds the {MAX_UPLOAD_ROWS:,}-row safety limit.")
    missing = [name for name in FEATURE_NAMES if name not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    selected = frame.loc[:, FEATURE_NAMES].copy()
    for column in FEATURE_NAMES:
        selected[column] = pd.to_numeric(selected[column], errors="raise")
    values = selected.to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("All feature values must be finite numbers.")
    return selected


def predict_frame(
    bundle: InferenceBundle,
    frame: pd.DataFrame,
    *,
    model_name: str = "variational_quantum_classifier",
) -> pd.DataFrame:
    validated = validate_input_frame(frame)
    angles = bundle.preprocessor.transform(validated.to_numpy(dtype=float))
    if model_name == "variational_quantum_classifier":
        probabilities = bundle.quantum_model.predict_proba(angles)
    elif model_name in bundle.classical_models:
        probabilities = bundle.classical_models[model_name].predict_proba(angles)
    else:
        raise ValueError(f"Unknown model {model_name!r}; choose from {bundle.model_names}")
    predictions = np.argmax(probabilities, axis=1)
    result = validated.copy()
    result["prediction"] = predictions
    result["prediction_name"] = [TARGET_NAMES[index] for index in predictions]
    result["confidence"] = np.max(probabilities, axis=1)
    for index, target_name in enumerate(TARGET_NAMES):
        result[f"probability_{target_name}"] = probabilities[:, index]
    return result


def explain_raw_row(
    bundle: InferenceBundle,
    raw_values: np.ndarray,
    *,
    saliency_step: float = 1e-3,
    counterfactual_grid_points: int = 121,
) -> dict[str, object]:
    values = np.asarray(raw_values, dtype=float).reshape(1, 4)
    angles = bundle.preprocessor.transform(values)[0]
    probabilities = bundle.quantum_model.predict_proba(angles[None, :])[0]
    return {
        "prediction": int(np.argmax(probabilities)),
        "prediction_name": TARGET_NAMES[int(np.argmax(probabilities))],
        "probabilities": probabilities,
        "angles": angles,
        "encoding": encoding_frame(values[0], angles, FEATURE_NAMES),
        "saliency": local_input_saliency(
            bundle.quantum_model, angles, FEATURE_NAMES, step=saliency_step
        ),
        "counterfactual": nearest_single_feature_counterfactual(
            bundle.quantum_model,
            angles,
            bundle.preprocessor,
            grid_points=counterfactual_grid_points,
        ),
    }
