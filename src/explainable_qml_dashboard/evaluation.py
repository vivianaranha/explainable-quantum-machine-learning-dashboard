"""Matched classical baselines and frozen-test evaluation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.svm import SVC


class ProbabilisticClassifier(Protocol):
    def predict_proba(self, values: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class ModelEvaluation:
    metrics: dict[str, float]
    probabilities: np.ndarray
    predictions: np.ndarray
    confusion: np.ndarray


def expected_calibration_error(
    y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> float:
    """Top-label expected calibration error."""

    predictions = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1)
    correctness = predictions == y_true
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        include = (confidence > lower) & (confidence <= upper)
        if np.any(include):
            result += float(np.mean(include)) * abs(
                float(np.mean(correctness[include])) - float(np.mean(confidence[include]))
            )
    return result


def multiclass_brier_score(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    one_hot = np.eye(probabilities.shape[1])[y_true]
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def evaluate_probabilities(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    inference_ms_per_sample: float,
    fit_seconds: float,
) -> ModelEvaluation:
    predictions = np.argmax(probabilities, axis=1)
    metrics = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "macro_precision": float(
            precision_score(y_true, predictions, average="macro", zero_division=0)
        ),
        "macro_recall": float(recall_score(y_true, predictions, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "roc_auc_ovr": float(roc_auc_score(y_true, probabilities, multi_class="ovr")),
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1, 2])),
        "brier_score": multiclass_brier_score(y_true, probabilities),
        "expected_calibration_error": expected_calibration_error(y_true, probabilities),
        "fit_seconds": float(fit_seconds),
        "inference_ms_per_sample": float(inference_ms_per_sample),
    }
    return ModelEvaluation(
        metrics=metrics,
        probabilities=probabilities,
        predictions=predictions,
        confusion=confusion_matrix(y_true, predictions, labels=[0, 1, 2]),
    )


def timed_evaluation(
    model: ProbabilisticClassifier,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    fit_seconds: float,
) -> ModelEvaluation:
    started = time.perf_counter()
    probabilities = model.predict_proba(x_test)
    elapsed = time.perf_counter() - started
    return evaluate_probabilities(
        y_test,
        probabilities,
        inference_ms_per_sample=elapsed * 1000.0 / len(x_test),
        fit_seconds=fit_seconds,
    )


def fit_classical_baselines(
    x_train: np.ndarray, y_train: np.ndarray, random_seed: int
) -> tuple[dict[str, ProbabilisticClassifier], dict[str, float]]:
    """Fit fixed-hyperparameter baselines on exactly the same training partition."""

    estimators: dict[str, ProbabilisticClassifier] = {
        "logistic_regression": LogisticRegression(C=1.0, max_iter=2000, random_state=random_seed),
        "rbf_svc": SVC(C=2.0, gamma="scale", probability=True, random_state=random_seed),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=2,
            random_state=random_seed,
            n_jobs=1,
        ),
    }
    fit_times: dict[str, float] = {}
    for name, estimator in estimators.items():
        started = time.perf_counter()
        estimator.fit(x_train, y_train)  # type: ignore[attr-defined]
        fit_times[name] = time.perf_counter() - started
    return estimators, fit_times


def metrics_frame(evaluations: dict[str, ModelEvaluation]) -> pd.DataFrame:
    rows = [{"model": name, **evaluation.metrics} for name, evaluation in evaluations.items()]
    return pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)


def predictions_frame(
    indices: np.ndarray,
    y_true: np.ndarray,
    target_names: tuple[str, ...],
    evaluations: dict[str, ModelEvaluation],
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str | bool]] = []
    for model_name, evaluation in evaluations.items():
        for row_id, target, prediction, probabilities in zip(
            indices,
            y_true,
            evaluation.predictions,
            evaluation.probabilities,
            strict=True,
        ):
            rows.append(
                {
                    "row_id": int(row_id),
                    "model": model_name,
                    "target": int(target),
                    "target_name": target_names[int(target)],
                    "prediction": int(prediction),
                    "prediction_name": target_names[int(prediction)],
                    "correct": bool(target == prediction),
                    "confidence": float(np.max(probabilities)),
                    "probability_setosa": float(probabilities[0]),
                    "probability_versicolor": float(probabilities[1]),
                    "probability_virginica": float(probabilities[2]),
                }
            )
    return pd.DataFrame(rows)


def calibration_frame(
    y_true: np.ndarray,
    evaluations: dict[str, ModelEvaluation],
    bins: int = 8,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    edges = np.linspace(0.0, 1.0, bins + 1)
    for model_name, evaluation in evaluations.items():
        confidence = np.max(evaluation.probabilities, axis=1)
        correctness = evaluation.predictions == y_true
        for bin_index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
            include = (confidence > lower) & (confidence <= upper)
            if np.any(include):
                rows.append(
                    {
                        "model": model_name,
                        "bin": bin_index,
                        "lower": float(lower),
                        "upper": float(upper),
                        "count": int(np.sum(include)),
                        "mean_confidence": float(np.mean(confidence[include])),
                        "empirical_accuracy": float(np.mean(correctness[include])),
                    }
                )
    return pd.DataFrame(rows)
