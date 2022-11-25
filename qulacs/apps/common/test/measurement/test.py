import pytest
import numpy as np
from qulacs import QuantumState, QuantumCircuit
from common.measurement import sampling_indirect_measurement

def a_state(n_qubits: int) -> QuantumState:
    state = QuantumState(n_qubits)
    circuit = QuantumCircuit(n_qubits)
    circuit.add_H_gate(0)
    circuit.update_quantum_state(state)
    return state

class TestMeasurement:
    def test_sampling_indirect_measurement(self) -> None:
        n_qubits = 8
        n_shots = 1000
        state = a_state(n_qubits)
        exp_value = 1
        exp_value = sampling_indirect_measurement(state, n_shots)
        print(f"exp: {exp_value}")
        assert exp_value == 0.0