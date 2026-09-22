import numpy as np

from explainable_qml_dashboard.evaluation import (
    calibration_frame,
    evaluate_probabilities,
    expected_calibration_error,
    fit_classical_baselines,
    metrics_frame,
    multiclass_brier_score,
)


def test_perfect_metrics():
    y = np.array([0, 1, 2])
    probabilities = np.eye(3)
    evaluation = evaluate_probabilities(
        y, probabilities, inference_ms_per_sample=0.1, fit_seconds=1.0
    )
    assert evaluation.metrics["accuracy"] == 1.0
    assert evaluation.metrics["macro_f1"] == 1.0
    assert evaluation.metrics["brier_score"] == 0.0


def test_ece_perfect_confidence():
    assert expected_calibration_error(np.array([0, 1]), np.array([[1, 0, 0], [0, 1, 0]])) == 0


def test_brier_nonnegative():
    score = multiclass_brier_score(np.array([0]), np.array([[0.5, 0.25, 0.25]]))
    assert score >= 0.0


def test_classical_models_fit(prepared, dataset, splits):
    _, angles = prepared
    models, fit_times = fit_classical_baselines(
        angles[splits.train_indices], dataset.y[splits.train_indices], 42
    )
    assert set(models) == {"logistic_regression", "rbf_svc", "random_forest"}
    assert all(value >= 0 for value in fit_times.values())
    for model in models.values():
        assert model.predict_proba(angles[splits.test_indices]).shape == (30, 3)


def test_metric_and_calibration_frames():
    y = np.array([0, 1, 2])
    evaluation = evaluate_probabilities(y, np.eye(3), inference_ms_per_sample=0.0, fit_seconds=0.0)
    evaluations = {"perfect": evaluation}
    assert metrics_frame(evaluations).iloc[0]["model"] == "perfect"
    calibration = calibration_frame(y, evaluations)
    assert calibration["count"].sum() == 3
