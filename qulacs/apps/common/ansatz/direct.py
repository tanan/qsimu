import sys
import numpy as np
from qulacs import QuantumCircuit
from qulacs.gate import CZ, RY, RZ, merge, TwoQubitDepolarizingNoise

sys.path.append("..")
from .ansatz import Ansatz


class DirectAnsatz(Ansatz):
    def __init__(self, nqubit, depth, noise, gate_set):
        super().__init__(nqubit, depth, noise, gate_set)

    def create_hamiltonian(self, cn=None, bn=None):
        return None, None

    def create_ansatz(self, random_list):
        circuit = QuantumCircuit(self.nqubit)
        for d in range(self.depth):
            for i in range(self.nqubit):
                circuit.add_gate(
                    merge(
                        RY(i, random_list[2 * i + 2 * self.nqubit * d]),
                        RZ(i, random_list[2 * i + 1 + 2 * self.nqubit * d]),
                    )
                )
            for i in range(self.nqubit // 2):
                circuit = self.add_cz_gate(circuit, 2 * i, 2 * i + 1)
            for i in range(self.nqubit // 2 - 1):
                circuit = self.add_cz_gate(circuit, 2 * i + 1, 2 * i + 2)
        for i in range(self.nqubit):
            circuit.add_gate(
                merge(
                    RY(i, random_list[2 * i + 2 * self.nqubit * self.depth]),
                    RZ(i, random_list[2 * i + 1 + 2 * self.nqubit * self.depth]),
                )
            )

        return circuit

    def add_cz_gate(self, circuit, control_qubit, target_qubit):
        circuit.add_gate(CZ(control_qubit, target_qubit))

        if self.noise.twoqubit.enabled:
            circuit.add_gate(
                TwoQubitDepolarizingNoise(
                    control_qubit, target_qubit, self.noise.twoqubit.value
                )
            )
        return circuit
