import numpy as np

from explainable_qml_dashboard.explain import (
    circuit_trace_frames,
    encoding_frame,
    global_input_saliency,
    local_input_saliency,
    nearest_single_feature_counterfactual,
)


def test_encoding_frame(prepared, dataset):
    _, angles = prepared
    frame = encoding_frame(dataset.x_raw[0], angles[0], dataset.feature_names)
    assert len(frame) == 4
    assert frame["angle_fraction_of_pi"].between(0, 1).all()


def test_circuit_trace_frames(random_angles, qvc):
    trace, basis = circuit_trace_frames(random_angles[0], qvc)
    assert trace.shape[0] == 3
    assert basis.shape[0] == 3 * 16
    np.testing.assert_allclose(basis.groupby("stage")["probability"].sum(), 1.0)


def test_local_saliency_shape(random_angles, qvc):
    frame = local_input_saliency(qvc, random_angles[0], ("a", "b", "c", "d"))
    assert len(frame) == 4
    assert np.isfinite(frame["signed_sensitivity_per_radian"]).all()


def test_global_saliency_shape(random_angles, qvc):
    frame = global_input_saliency(qvc, random_angles[:3], ("a", "b", "c", "d"))
    assert len(frame) == 4
    assert frame["mean_absolute_sensitivity"].is_monotonic_decreasing


def test_counterfactual_contract(prepared, random_angles, qvc):
    preprocessor, _ = prepared
    result = nearest_single_feature_counterfactual(
        qvc, random_angles[0], preprocessor, grid_points=11
    )
    assert "found" in result
    assert "original_prediction" in result
