import pytest
import numpy as np
from qulacs import QuantumState, QuantumCircuit
from common.measurement import sampling_indirect_measurement, create_swap_gate_for_xy_model
from qulacs.gate import Measurement, DenseMatrix

def a_state(n_qubits: int) -> QuantumState:
    state = QuantumState(n_qubits)
    circuit = QuantumCircuit(n_qubits)
    circuit.add_X_gate(0)
    circuit.add_X_gate(2)
    circuit.update_quantum_state(state)
    return state

def b_state(n_qubits: int) -> QuantumState:
    state = QuantumState(n_qubits)
    circuit = QuantumCircuit(n_qubits)
    circuit.add_H_gate(0)
    circuit.add_H_gate(1)
    circuit.add_H_gate(2)
    circuit.add_H_gate(3)
    circuit.update_quantum_state(state)
    return state

class TestMeasurement:
    # def test_sampling_indirect_measurement(self) -> None:
    #     n_qubits = 4
    #     n_shots = 1000
    #     state = a_state(n_qubits)
    #     exact_value = -3.0
    #     exp_value = sampling_indirect_measurement(state, n_shots)
    #     print(f"exp: {exp_value}")
    #     assert abs(exp_value - exact_value) < 0.1

    def test_sampling_indirect_measurement2(self) -> None:
        n_qubits = 4
        n_shots = 1
        state = b_state(n_qubits)
        exact_value = -3.0
        exp_value = sampling_indirect_measurement(state, n_shots)
        print(f"exp: {exp_value}")
        assert abs(exp_value - exact_value) < 0.1


    # def test_sampling_indirect_measurement3(self) -> None:
    #     n_qubits = 4
    #     state = b_state(n_qubits)
    #     circuit = QuantumCircuit(n_qubits)
    #     circuit.add_H_gate(0)
    #     circuit.add_H_gate(1)
    #     circuit.add_gate(Measurement(0, 0))
    #     circuit.add_gate(Measurement(1, 1))

    #     circuit.update_quantum_state(state)
    #     print(f"1: {state.get_classical_value(0)}, 2: {state.get_classical_value(1)}")
    #     create_swap_gate_for_xy_model(state, 0, 2).update_quantum_state(state)
    #     create_swap_gate_for_xy_model(state, 1, 3).update_quantum_state(state)
    #     circuit4 = QuantumCircuit(n_qubits)
    #     circuit4.add_gate(Measurement(0, 0))
    #     circuit4.add_gate(Measurement(1, 1))
    #     circuit4.update_quantum_state(state)
    #     print(f"3: {state.get_classical_value(0)}, 4: {state.get_classical_value(1)}")
    #     assert False