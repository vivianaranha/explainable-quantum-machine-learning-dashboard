"""Transparent four-qubit simulator and an independent Qiskit circuit builder."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

HADAMARD = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)


def ry(angle: float) -> np.ndarray:
    """RY rotation matrix."""

    half = angle / 2.0
    return np.array([[np.cos(half), -np.sin(half)], [np.sin(half), np.cos(half)]], dtype=complex)


def rz(angle: float) -> np.ndarray:
    """RZ rotation matrix."""

    half = angle / 2.0
    return np.diag([np.exp(-1j * half), np.exp(1j * half)]).astype(complex)


def _apply_single_qubit(states: np.ndarray, gate: np.ndarray, qubit: int) -> None:
    stride = 1 << qubit
    period = stride << 1
    for start in range(0, states.shape[1], period):
        lower = np.arange(start, start + stride)
        upper = lower + stride
        a = states[:, lower].copy()
        b = states[:, upper].copy()
        states[:, lower] = gate[0, 0] * a + gate[0, 1] * b
        states[:, upper] = gate[1, 0] * a + gate[1, 1] * b


def _apply_rzz(states: np.ndarray, angle: np.ndarray | float, q0: int, q1: int) -> None:
    indices = np.arange(states.shape[1])
    parity = ((indices >> q0) & 1) ^ ((indices >> q1) & 1)
    signs = np.where(parity == 0, -1.0, 1.0)
    angle_array = np.asarray(angle, dtype=float)
    if angle_array.ndim == 0:
        states *= np.exp(0.5j * float(angle_array) * signs)[None, :]
    else:
        states *= np.exp(0.5j * angle_array[:, None] * signs[None, :])


def _apply_cz(states: np.ndarray, q0: int, q1: int) -> None:
    indices = np.arange(states.shape[1])
    both_one = (((indices >> q0) & 1) == 1) & (((indices >> q1) & 1) == 1)
    states[:, both_one] *= -1.0


def feature_entangling_angle(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Input-dependent RZZ angle used by the feature map."""

    return 0.5 * (np.pi - left) * (np.pi - right) / np.pi


def initial_states(sample_count: int, qubits: int = 4) -> np.ndarray:
    states = np.zeros((sample_count, 1 << qubits), dtype=complex)
    states[:, 0] = 1.0
    return states


def apply_feature_map(states: np.ndarray, x_angles: np.ndarray) -> None:
    """Apply H, data rotations, and data-dependent nearest-neighbor RZZ gates."""

    for qubit in range(4):
        _apply_single_qubit(states, HADAMARD, qubit)
        for row, angle in enumerate(x_angles[:, qubit]):
            row_view = states[row : row + 1]
            _apply_single_qubit(row_view, ry(float(angle)), qubit)
            _apply_single_qubit(row_view, rz(float(angle) / 2.0), qubit)
    for q0, q1 in ((0, 1), (1, 2), (2, 3)):
        angle = feature_entangling_angle(x_angles[:, q0], x_angles[:, q1])
        _apply_rzz(states, angle, q0, q1)


def apply_variational_layer(states: np.ndarray, layer_parameters: np.ndarray) -> None:
    """Apply one RY/RZ layer followed by a CZ ring."""

    for qubit in range(4):
        _apply_single_qubit(states, ry(float(layer_parameters[qubit, 0])), qubit)
        _apply_single_qubit(states, rz(float(layer_parameters[qubit, 1])), qubit)
    for q0, q1 in ((0, 1), (1, 2), (2, 3), (3, 0)):
        _apply_cz(states, q0, q1)


def simulate(x_angles: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    """Return exact final states for one or more angle-encoded samples."""

    x_angles = np.asarray(x_angles, dtype=float)
    if x_angles.ndim == 1:
        x_angles = x_angles[None, :]
    if x_angles.ndim != 2 or x_angles.shape[1] != 4:
        raise ValueError("x_angles must have shape (samples, 4)")
    parameters = np.asarray(parameters, dtype=float)
    if parameters.ndim != 3 or parameters.shape[1:] != (4, 2):
        raise ValueError("parameters must have shape (layers, 4, 2)")
    states = initial_states(len(x_angles))
    apply_feature_map(states, x_angles)
    for layer_parameters in parameters:
        apply_variational_layer(states, layer_parameters)
    return states


def z_expectations(states: np.ndarray, readout_qubits: tuple[int, ...] = (0, 1, 2)) -> np.ndarray:
    """Calculate exact Pauli-Z expectations for selected qubits."""

    probabilities = np.abs(states) ** 2
    indices = np.arange(states.shape[1])
    columns = []
    for qubit in readout_qubits:
        signs = np.where(((indices >> qubit) & 1) == 0, 1.0, -1.0)
        columns.append(probabilities @ signs)
    return np.column_stack(columns)


def bloch_vectors(states: np.ndarray) -> np.ndarray:
    """Return exact (X, Y, Z) expectations for each qubit and sample."""

    result = np.zeros((len(states), 4, 3), dtype=float)
    indices = np.arange(states.shape[1])
    for qubit in range(4):
        zero_indices = indices[((indices >> qubit) & 1) == 0]
        one_indices = zero_indices | (1 << qubit)
        overlap = np.sum(np.conj(states[:, zero_indices]) * states[:, one_indices], axis=1)
        result[:, qubit, 0] = 2.0 * np.real(overlap)
        result[:, qubit, 1] = 2.0 * np.imag(overlap)
        signs = np.where(((indices >> qubit) & 1) == 0, 1.0, -1.0)
        result[:, qubit, 2] = np.abs(states) ** 2 @ signs
    return result


def single_qubit_entropy(state: np.ndarray, qubit: int) -> float:
    """Von Neumann entropy of one qubit in a pure four-qubit state."""

    indices = np.arange(len(state))
    zero_indices = indices[((indices >> qubit) & 1) == 0]
    one_indices = zero_indices | (1 << qubit)
    a = state[zero_indices]
    b = state[one_indices]
    density = np.array(
        [
            [np.vdot(a, a), np.vdot(b, a)],
            [np.vdot(a, b), np.vdot(b, b)],
        ],
        dtype=complex,
    )
    eigenvalues = np.linalg.eigvalsh(density).real
    eigenvalues = eigenvalues[eigenvalues > 1e-12]
    return float(-np.sum(eigenvalues * np.log2(eigenvalues)))


@dataclass(frozen=True)
class LayerSnapshot:
    stage: str
    state: np.ndarray
    z_expectations: np.ndarray
    mean_entanglement_entropy: float


def trace_circuit(x_angles: np.ndarray, parameters: np.ndarray) -> list[LayerSnapshot]:
    """Capture the state and diagnostics after encoding and each trainable layer."""

    x_angles = np.asarray(x_angles, dtype=float).reshape(1, 4)
    states = initial_states(1)
    apply_feature_map(states, x_angles)
    snapshots: list[LayerSnapshot] = []

    def capture(stage: str) -> None:
        state = states[0].copy()
        snapshots.append(
            LayerSnapshot(
                stage=stage,
                state=state,
                z_expectations=z_expectations(states)[0],
                mean_entanglement_entropy=float(
                    np.mean([single_qubit_entropy(state, qubit) for qubit in range(4)])
                ),
            )
        )

    capture("feature_map")
    for index, layer_parameters in enumerate(parameters, start=1):
        apply_variational_layer(states, layer_parameters)
        capture(f"variational_layer_{index}")
    return snapshots


def build_qiskit_circuit(x_angles: np.ndarray, parameters: np.ndarray) -> QuantumCircuit:
    """Construct the same circuit in Qiskit for visualization and parity tests."""

    x_angles = np.asarray(x_angles, dtype=float).reshape(4)
    parameters = np.asarray(parameters, dtype=float)
    circuit = QuantumCircuit(4, name="Explainable QVC")
    for qubit, angle in enumerate(x_angles):
        circuit.h(qubit)
        circuit.ry(float(angle), qubit)
        circuit.rz(float(angle) / 2.0, qubit)
    for q0, q1 in ((0, 1), (1, 2), (2, 3)):
        angle = float(feature_entangling_angle(x_angles[q0], x_angles[q1]))
        circuit.rzz(angle, q0, q1)
    circuit.barrier(label="encoded")
    for layer_index, layer_parameters in enumerate(parameters, start=1):
        for qubit in range(4):
            circuit.ry(float(layer_parameters[qubit, 0]), qubit)
            circuit.rz(float(layer_parameters[qubit, 1]), qubit)
        for q0, q1 in ((0, 1), (1, 2), (2, 3), (3, 0)):
            circuit.cz(q0, q1)
        circuit.barrier(label=f"layer {layer_index}")
    return circuit


def qiskit_state(x_angles: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    """Run the independent Qiskit statevector implementation."""

    return np.asarray(Statevector.from_instruction(build_qiskit_circuit(x_angles, parameters)).data)


def state_fidelity(left: np.ndarray, right: np.ndarray) -> float:
    """Global-phase-invariant pure-state fidelity."""

    return float(np.abs(np.vdot(left, right)) ** 2)
