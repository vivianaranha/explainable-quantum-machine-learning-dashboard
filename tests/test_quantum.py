import numpy as np
import pytest

from explainable_qml_dashboard.quantum import (
    bloch_vectors,
    build_qiskit_circuit,
    feature_entangling_angle,
    qiskit_state,
    simulate,
    single_qubit_entropy,
    state_fidelity,
    trace_circuit,
    z_expectations,
)


def test_simulation_shape(random_angles, qvc):
    assert simulate(random_angles, qvc.parameters).shape == (6, 16)


def test_states_normalized(random_angles, qvc):
    states = simulate(random_angles, qvc.parameters)
    np.testing.assert_allclose(np.sum(np.abs(states) ** 2, axis=1), 1.0, atol=1e-12)


@pytest.mark.parametrize("sample_index", range(4))
def test_numpy_qiskit_parity(random_angles, qvc, sample_index):
    numpy_state = simulate(random_angles[sample_index], qvc.parameters)[0]
    independent_state = qiskit_state(random_angles[sample_index], qvc.parameters)
    assert state_fidelity(numpy_state, independent_state) > 1.0 - 1e-12


def test_expectation_range(random_angles, qvc):
    expectations = z_expectations(simulate(random_angles, qvc.parameters))
    assert expectations.shape == (6, 3)
    assert np.max(np.abs(expectations)) <= 1.0 + 1e-12


def test_bloch_vector_range(random_angles, qvc):
    vectors = bloch_vectors(simulate(random_angles, qvc.parameters))
    assert vectors.shape == (6, 4, 3)
    assert np.max(np.linalg.norm(vectors, axis=2)) <= 1.0 + 1e-12


def test_entropy_range(random_angles, qvc):
    state = simulate(random_angles[0], qvc.parameters)[0]
    entropies = [single_qubit_entropy(state, qubit) for qubit in range(4)]
    assert min(entropies) >= -1e-12
    assert max(entropies) <= 1.0 + 1e-12


def test_trace_has_encoding_and_layers(random_angles, qvc):
    trace = trace_circuit(random_angles[0], qvc.parameters)
    assert [snapshot.stage for snapshot in trace] == [
        "feature_map",
        "variational_layer_1",
        "variational_layer_2",
    ]


def test_qiskit_circuit_structure(random_angles, qvc):
    circuit = build_qiskit_circuit(random_angles[0], qvc.parameters)
    assert circuit.num_qubits == 4
    assert circuit.count_ops()["rzz"] == 3
    assert circuit.count_ops()["cz"] == 8


def test_feature_entangling_angle_boundaries():
    assert feature_entangling_angle(np.array([np.pi]), np.array([0.0]))[0] == 0.0
    assert feature_entangling_angle(np.array([0.0]), np.array([0.0]))[0] == pytest.approx(np.pi / 2)


@pytest.mark.parametrize(
    ("x_shape", "parameter_shape"),
    [((2, 3), (2, 4, 2)), ((2, 4), (2, 3, 2)), ((2, 4), (8,))],
)
def test_simulator_shape_validation(x_shape, parameter_shape):
    with pytest.raises(ValueError):
        simulate(np.zeros(x_shape), np.zeros(parameter_shape))
