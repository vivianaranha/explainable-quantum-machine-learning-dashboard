import numpy as np

from explainable_qml_dashboard.qvc import (
    VariationalQuantumClassifier,
    parameter_names,
    softmax,
)


def test_softmax_rows_sum_to_one():
    probabilities = softmax(np.array([[1000.0, 1001.0, 999.0], [0.0, 0.0, 0.0]]))
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


def test_parameter_names_are_unique():
    names = parameter_names(2)
    assert len(names) == 16
    assert len(set(names)) == 16


def test_predict_shapes(random_angles, qvc):
    assert qvc.predict_proba(random_angles).shape == (6, 3)
    assert qvc.predict(random_angles).shape == (6,)


def test_probability_contract(random_angles, qvc):
    probabilities = qvc.predict_proba(random_angles)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-12)
    assert np.min(probabilities) >= 0.0


def test_parameter_shift_matches_finite_difference(random_angles, qvc):
    y = np.array([0, 1, 2])
    x = random_angles[:3]
    analytic = qvc.parameter_shift_gradient(x, y).ravel()
    epsilon = 1e-6
    for index in (0, 5, 15):
        original = qvc.parameters.ravel()[index]
        qvc.parameters.ravel()[index] = original + epsilon
        plus = qvc.loss(x, y)
        qvc.parameters.ravel()[index] = original - epsilon
        minus = qvc.loss(x, y)
        qvc.parameters.ravel()[index] = original
        np.testing.assert_allclose(
            analytic[index], (plus - minus) / (2 * epsilon), rtol=2e-5, atol=2e-6
        )


def test_training_records_history(random_angles):
    model = VariationalQuantumClassifier(layers=1, random_seed=1)
    y = np.array([0, 1, 2, 0, 1, 2])
    result = model.fit(random_angles[:4], y[:4], random_angles[4:], y[4:], epochs=2, patience=2)
    assert len(result.history) == 2
    assert len(result.gradients) == 2 * model.parameter_count


def test_model_json_round_trip(tmp_path, random_angles, qvc):
    target = tmp_path / "model.json"
    expected = qvc.predict_proba(random_angles)
    qvc.save(target)
    restored = VariationalQuantumClassifier.load(target)
    np.testing.assert_allclose(restored.predict_proba(random_angles), expected)


def test_seed_is_deterministic():
    left = VariationalQuantumClassifier(random_seed=99)
    right = VariationalQuantumClassifier(random_seed=99)
    np.testing.assert_array_equal(left.parameters, right.parameters)
