from qulacs import QuantumState, QuantumCircuit
from qulacs.gate import Measurement, DenseMatrix
import numpy as np


def _apply_sign_correction(circuit: QuantumCircuit, control_qubit_num: int) -> None:
    circuit.add_X_gate(control_qubit_num)
    circuit.add_Z_gate(control_qubit_num)
    circuit.add_X_gate(control_qubit_num)
    if control_qubit_num == 1:
        circuit.add_Z_gate(control_qubit_num)


def create_swap_gate_for_xy_model(state: QuantumState, control_qubit_num: int, target_qubit_num: int) -> QuantumCircuit:
    """
    When control_qubit_num is 0,
        In the case of control_qubit state is 1, operate Identity j-1.
        In the case of control_qubit state is 0, operate `X_{j-1} Z_{j-1} X_{j-1}` for sign correction.

    When control_qubit_num is 1,
        In the case of control_qubit state is 1, operate Identity j-1.
        In the case of control_qubit state is 0, operate `X_{j-1} Z_{j-1} 
        After operation above, operate Z_{1} for sign correction.
    """
    
    circuit = QuantumCircuit(state.get_qubit_count())
    swap = DenseMatrix([control_qubit_num, target_qubit_num], [[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]])
    circuit.add_gate(swap)
    for i in range(control_qubit_num + 1, target_qubit_num):
        circuit.add_Z_gate(i)

    if state.get_classical_value(control_qubit_num) == 0:
        _apply_sign_correction(circuit, control_qubit_num)

    return circuit


def create_measurement_gate(n_qubits: int, qubit_num: int, register_address: int, operator: str) -> QuantumCircuit:
    circuit = QuantumCircuit(n_qubits)
    if operator == "X":
        circuit.add_H_gate(qubit_num)
    elif operator == "Y":
        circuit.add_RZ_gate(qubit_num, -np.pi / 2)

    circuit.add_gate(Measurement(qubit_num, register_address))
    return circuit


def _exec_XX_measurement(state: QuantumState, n_qubits: int, register_address: int) -> None:
    create_measurement_gate(n_qubits, 0, register_address, "X").update_quantum_state(state)
    create_measurement_gate(n_qubits, 1, register_address + 1, "X").update_quantum_state(state)


def _exec_ZZ_measurement(state: QuantumState, n_qubits: int, register_address: int):
    create_measurement_gate(n_qubits, 0, register_address, "Z").update_quantum_state(state)
    create_measurement_gate(n_qubits, 1, register_address + 1, "Z").update_quantum_state(state)


def exec_measurement(state: QuantumState, n_qubits: int, measurement_type: str) -> list[int]:
    result = []
    qubit_num = 0
    if measurement_type == "ZZ":
        _exec_ZZ_measurement(state, n_qubits, qubit_num)
    else:
        _exec_XX_measurement(state, n_qubits, qubit_num)
    result.append(state.get_classical_value(qubit_num))
    result.append(state.get_classical_value(qubit_num + 1))

    while qubit_num + 2 < state.get_qubit_count():
        create_swap_gate_for_xy_model(state, 0, qubit_num + 2).update_quantum_state(state)
        if qubit_num + 1 + 2 < state.get_qubit_count():
            create_swap_gate_for_xy_model(state, 1, qubit_num + 1 + 2).update_quantum_state(state)

        qubit_num += 2
        if measurement_type == "ZZ":
            _exec_ZZ_measurement(state, n_qubits, qubit_num)
        else:
            _exec_XX_measurement(state, n_qubits, qubit_num)
        result.append(state.get_classical_value(qubit_num))
        result.append(state.get_classical_value(qubit_num + 1))

    return result[0:n_qubits]


def sampling_indirect_measurement(state: QuantumState, n_shots: int) -> float:
    # in the case of transversal ising.
    n_qubits = state.get_qubit_count()
    samples_ZZ = []
    samples_XX = []
    for i in range(n_shots):
        samples_ZZ.append(exec_measurement(state, n_qubits, "ZZ"))
        samples_XX.append(exec_measurement(state, n_qubits, "XX"))

    # print(f"ZZ: {samples_ZZ}")
    # print(f"XX: {samples_XX}")
    # X1X2 など成分ごとの合計 / n_shots で平均値を計算
    estimates_ZZ = []
    estimates_XX = []
    for i in range(n_qubits-1):
        left = np.array(samples_ZZ)[:,i]
        right = np.array(samples_ZZ)[:,i+1]
        estimates_ZZ.append((left * right).sum(dtype='float') / n_shots)

    estimates_XX = np.sum(samples_XX, axis=0, dtype='float') / n_shots
    # print(f"ZZ: {estimates_ZZ}")
    # print(f"XX: {estimates_XX}")

    # calculate the results of X1X2 + X2X3 + X3X4 + X4X5 + X5X6 + Z1 + Z2 + Z3 + ...
    expectation_value = np.sum(estimates_ZZ, dtype='float') + np.sum(estimates_XX, dtype='float')
    return expectation_value
