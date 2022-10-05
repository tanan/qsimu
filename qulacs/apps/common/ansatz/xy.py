import sys
import numpy as np
from qulacs import QuantumCircuit
from qulacs.gate import (
    CNOT,
    RY,
    RZ,
    merge,
    DepolarizingNoise,
    TwoQubitDepolarizingNoise,
)

from .ansatz import Ansatz, ParametricGateCountError
from .pauli_gate import PauliGate


class XYAnsatz(Ansatz):
    def __init__(self, nqubit, depth, noise, gate_set, time, bn, gamma=0):
        super().__init__(nqubit, depth, noise, gate_set, time, bn, gamma)

    def create_hamiltonian(self, cn, bn, gamma):
        XX = np.array(np.zeros(2**self.nqubit))
        YY = np.array(np.zeros(2**self.nqubit))
        Zn = np.array(np.zeros(2**self.nqubit))
        for k in range(self.nqubit - 1):
            for l in range(self.nqubit):
                if k == l:
                    if l == 0:
                        hamiX = PauliGate.X_gate.value
                        hamiY = PauliGate.Y_gate.value
                    else:
                        hamiX = np.kron(hamiX, PauliGate.X_gate.value)
                        hamiY = np.kron(hamiY, PauliGate.Y_gate.value)

                elif k + 1 == l:
                    hamiX = np.kron(hamiX, PauliGate.X_gate.value)
                    hamiY = np.kron(hamiY, PauliGate.Y_gate.value)
                else:
                    if l == 0:
                        hamiX = PauliGate.I_gate.value
                        hamiY = PauliGate.I_gate.value
                    else:
                        hamiX = np.kron(hamiX, PauliGate.I_gate.value)
                        hamiY = np.kron(hamiY, PauliGate.I_gate.value)
            XX = XX + 0.5 * cn[k] * (1 + gamma) * hamiX
            YY = YY + 0.5 * cn[k] * (1 - gamma) * hamiY

        for m in range(self.nqubit):
            for n in range(self.nqubit):
                if m == n:
                    if n == 0:
                        hamiZ = PauliGate.Z_gate.value
                    else:
                        hamiZ = np.kron(hamiZ, PauliGate.Z_gate.value)

                else:
                    if n == 0:
                        hamiZ = PauliGate.I_gate.value
                    else:
                        hamiZ = np.kron(hamiZ, PauliGate.I_gate.value)

            Zn = Zn + bn["value"][m] * hamiZ

        hamiltonian = XX + YY + Zn
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
            circuit.add_gate(CNOT(0, 1))
            if self.noise["twoqubit"]["enabled"]:
                circuit.add_gate(
                    TwoQubitDepolarizingNoise(
                        0, 1, self.noise["twoqubit"]["value"]
                    )
                )

            if self.bn["type"] == "random":
                circuit = self.add_parametric_rotation_gate(
                    circuit,
                    random_list[
                        self.depth
                        + (self.depth * self.nqubit)
                        + (self.gate_set * d) : self.depth
                        + (self.depth * self.nqubit)
                        + (self.gate_set * d)
                        + 4
                    ],
                )

                ## When bn is random, recalculate daig and eigen_vecs so that the calculated values in advance can not use.
                self.diag, self.eigen_vecs = self.create_hamiltonian(
                    [1] * self.nqubit,
                    random_list[d + self.depth : d + self.depth + self.nqubit],
                    self.gamma,
                )
                circuit.add_gate(self.create_hamiltonian_gate(random_list[d]))

            if self.time["type"] == "random":
                circuit = self.add_parametric_rotation_gate(
                    circuit,
                    random_list[
                        self.depth
                        + (self.gate_set * d) : self.depth
                        + (self.gate_set * d)
                        + 4
                    ],
                )

                circuit.add_gate(self.create_hamiltonian_gate(random_list[d]))
            else:
                circuit = self.add_parametric_rotation_gate(
                    circuit,
                    random_list[(self.gate_set * d) : (self.gate_set * d) + 4],
                )

                ## When time is static, get time from static array.
                circuit.add_gate(self.create_hamiltonian_gate(self.time["value"][d]))

        return circuit

    def add_parametric_rotation_gate(self, circuit, params):
        """
        add parametric rotation gate to current circuit
        """
        if self.gate_set == 4:
            circuit.add_gate(merge(RY(0, params[0]), RZ(0, params[1])))
            circuit.add_gate(merge(RY(1, params[2]), RZ(1, params[3])))

            if self.noise["singlequbit"]["enabled"]:
                circuit.add_gate(
                    DepolarizingNoise(0, self.noise["singlequbit"]["value"])
                )
                circuit.add_gate(
                    DepolarizingNoise(1, self.noise["singlequbit"]["value"])
                )

        else:
            raise ParametricGateCountError("The number of parametric gates is invalid.")

        return circuit
