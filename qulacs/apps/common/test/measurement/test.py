import pytest
import numpy as np
from qulacs import QuantumState, QuantumCircuit
from common.measurement import sampling_indirect_measurement


def a_state(n_qubits: int) -> QuantumState:
    state = QuantumState(n_qubits)
    circuit = QuantumCircuit(n_qubits)
    circuit.add_H_gate(0)
    circuit.add_H_gate(1)
    circuit.add_X_gate(3)
    circuit.add_RY_gate(2, np.pi / 3)
    circuit.update_quantum_state(state)
    return state


class TestMeasurement:
    def test_sampling_indirect_measurement(self) -> None:
        n_qubits = 4
        n_shots = 2
        state = a_state(n_qubits)
        exp_value = sampling_indirect_measurement(state, n_shots)
        print(f"exp: {exp_value}")
        assert exp_value == 0.0