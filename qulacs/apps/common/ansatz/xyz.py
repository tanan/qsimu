import sys
import numpy as np
from qulacs import QuantumCircuit
from qulacs.gate import RZ

from .ansatz import Ansatz
from .pauli_gate import PauliGate


class XYZAnsatz(Ansatz):
    def __init__(self, nqubit, depth, gate_set, time, bn):
        super().__init__(nqubit, depth, gate_set, time, bn)

    def create_hamiltonian(self, cn, bn=None, gamma=None):
        XX = np.array(np.zeros(2**self.nqubit))
        YY = np.array(np.zeros(2**self.nqubit))
        ZZ = np.array(np.zeros(2**self.nqubit))
        for k in range(self.nqubit - 1):
            for l in range(self.nqubit):
                if k == l:
                    if l == 0:
                        hamiX = PauliGate.X_gate.value
                        hamiY = PauliGate.Y_gate.value
                        hamiZ = PauliGate.Z_gate.value
                    else:
                        hamiX = np.kron(hamiX, PauliGate.X_gate.value)
                        hamiY = np.kron(hamiY, PauliGate.Y_gate.value)
                        hamiZ = np.kron(hamiZ, PauliGate.Z_gate.value)

                elif k + 1 == l:
                    hamiX = np.kron(hamiX, PauliGate.X_gate.value)
                    hamiY = np.kron(hamiY, PauliGate.Y_gate.value)
                    hamiZ = np.kron(hamiZ, PauliGate.Z_gate.value)
                else:
                    if l == 0:
                        hamiX = PauliGate.I_gate.value
                        hamiY = PauliGate.I_gate.value
                        hamiZ = PauliGate.I_gate.value
                    else:
                        hamiX = np.kron(hamiX, PauliGate.I_gate.value)
                        hamiY = np.kron(hamiY, PauliGate.I_gate.value)
                        hamiZ = np.kron(hamiZ, PauliGate.I_gate.value)
            XX = XX + 0.5 * cn[k] * hamiX
            YY = YY + 0.5 * cn[k] * hamiY
            ZZ = ZZ + 0.5 * cn[k] * hamiZ

        hamiltonian = XX + YY + ZZ
        return np.linalg.eigh(hamiltonian)

    def create_ansatz(self, random_list):
        """
        ansatzを作成する
        ハミルトニアンゲートはdepthごとに別々のパラメータとなる.
        bnについてはdepth x nqubitの配列をパラメータとして渡す.
        TODO: r, cnのrandomizeは未実装
        """
        circuit = QuantumCircuit(self.nqubit)
        for d in range(self.depth):
            if self.bn["type"] == "random":
                circuit.add_gate(
                    RZ(
                        0,
                        random_list[
                            self.depth
                            + (self.depth * self.nqubit)
                            + (self.gate_set * d)
                        ],
                    )
                )
                circuit.add_gate(
                    RZ(
                        1,
                        random_list[
                            self.depth
                            + (self.depth * self.nqubit)
                            + (self.gate_set * d)
                            + 1
                        ],
                    )
                )
                circuit.add_gate(
                    self.create_hamiltonian_gate(
                        random_list[d + self.depth : d + self.depth + self.nqubit],
                        random_list[d],
                    )
                )
            elif self.bn["type"] == "static" or self.bn["type"] == "static_random":
                if self.time["type"] == "random":
                    circuit.add_gate(
                        RZ(0, random_list[self.depth + (self.gate_set * d)])
                    )
                    circuit.add_gate(
                        RZ(1, random_list[self.depth + (self.gate_set * d) + 1])
                    )
                    circuit.add_gate(self.create_hamiltonian_gate(random_list[d]))
                else:
                    circuit.add_gate(RZ(0, random_list[(self.gate_set * d)]))
                    circuit.add_gate(RZ(1, random_list[(self.gate_set * d) + 1]))
                    circuit.add_gate(self.create_hamiltonian_gate(self.time["min_val"]))

        return circuit
