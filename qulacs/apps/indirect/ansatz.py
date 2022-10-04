from mimetypes import init
from qulacs import QuantumCircuit
from qulacs.gate import CNOT, CZ, RX, RY, RZ, merge, DenseMatrix
import numpy as np
import random
from scipy import sparse
from scipy.sparse.linalg import expm


class AnsatzDirect:
    def __init__(self, n_qubit, random_list, depth):
        self.n_qubit = n_qubit
        self.random_list = random_list
        self.depth = depth

    def create_ansatz(self):
        circuit = QuantumCircuit(self.n_qubit)
        for d in range(self.depth):
            for i in range(self.n_qubit):
                circuit.add_gate(
                    merge(
                        RY(i, self.random_list[2 * i + 2 * self.n_qubit * d]),
                        RZ(i, self.random_list[2 * i + 1 + 2 * self.n_qubit * d]),
                    )
                )
            for i in range(self.n_qubit // 2):
                circuit.add_gate(CZ(2 * i, 2 * i + 1))
            for i in range(self.n_qubit // 2 - 1):
                circuit.add_gate(CZ(2 * i + 1, 2 * i + 2))
        for i in range(self.n_qubit):
            circuit.add_gate(
                merge(
                    RY(i, self.random_list[2 * i + 2 * self.n_qubit * self.depth]),
                    RZ(i, self.random_list[2 * i + 1 + 2 * self.n_qubit * self.depth]),
                )
            )

        return circuit


class AnsatzIndirectByIsing:
    def __init__(self, n_qubit, random_list, depth, is_bn_random):
        self.n_qubit = n_qubit
        self.random_list = random_list
        self.depth = depth
        self.gate_set = 2
        self.is_bn_random = is_bn_random
        self.bn = random.uniform(-1.0, 1.0)

    def create_hamiltonian_gate(self, cn, bn, t):
        I_gate = [[1, 0], [0, 1]]
        X_gate = [[0, 1], [1, 0]]
        Y_gate = [[0, 0 - 1j], [0 + 1j, 0]]

        XX = np.array(np.zeros(2**self.n_qubit))
        Y = np.array(np.zeros(2**self.n_qubit))
        for k in range(self.n_qubit - 1):
            for l in range(self.n_qubit):
                if k == l:
                    if l == 0:
                        hamiX = X_gate
                    else:
                        hamiX = np.kron(hamiX, X_gate)

                elif k + 1 == l:
                    hamiX = np.kron(hamiX, X_gate)
                else:
                    if l == 0:
                        hamiX = I_gate
                    else:
                        hamiX = np.kron(hamiX, I_gate)
            XX = XX + 0.5 * cn[k] * hamiX

        for m in range(self.n_qubit):
            for n in range(self.n_qubit):
                if m == n:
                    if n == 0:
                        hamiY = Y_gate
                    else:
                        hamiY = np.kron(hamiY, Y_gate)

                else:
                    if n == 0:
                        hamiY = I_gate
                    else:
                        hamiY = np.kron(hamiY, I_gate)

            Y = Y + bn[m] * hamiY

        hamiltonian = XX + Y
        hamiltonian_sparse = sparse.csc_matrix(hamiltonian)
        hami_gate = expm(-1j * hamiltonian_sparse * t)
        return DenseMatrix(list(range(self.n_qubit)), hami_gate.toarray())

    def create_ansatz(self):
        """
        ansatzを作成する
        ハミルトニアンゲートはdepthごとに別々のパラメータとなる.
        bnについてはdepth x n_qubitの配列をパラメータとして渡す.
        TODO: r, cnのrandomizeは未実装
        """
        circuit = QuantumCircuit(self.n_qubit)
        for d in range(self.depth):
            if self.is_bn_random:
                # + time + bn
                # circuit.add_gate(merge(RX(0, self.random_list[self.depth+(self.depth*self.n_qubit)+(self.gate_set*d)]), RY(0, self.random_list[self.depth+(self.depth*self.n_qubit)+(self.gate_set*d)+1])))
                # circuit.add_gate(merge(RX(1, self.random_list[self.depth+(self.depth*self.n_qubit)+(self.gate_set*d)+2]), RY(1, self.random_list[self.depth+(self.depth*self.n_qubit)+(self.gate_set*d)+3])))
                circuit.add_gate(
                    RZ(
                        0,
                        self.random_list[
                            self.depth
                            + (self.depth * self.n_qubit)
                            + (self.gate_set * d)
                        ],
                    )
                )
                circuit.add_gate(
                    RZ(
                        1,
                        self.random_list[
                            self.depth
                            + (self.depth * self.n_qubit)
                            + (self.gate_set * d)
                            + 1
                        ],
                    )
                )
                circuit.add_gate(
                    self.create_hamiltonian_gate(
                        [1] * self.n_qubit,
                        self.random_list[
                            d + self.depth : d + self.depth + self.n_qubit
                        ],
                        self.random_list[d],
                    )
                )
            else:
                # + time
                # circuit.add_gate(merge(RX(0, self.random_list[self.depth+(self.gate_set*d)]), RY(0, self.random_list[self.depth+(self.gate_set*d)+1])))
                # circuit.add_gate(merge(RX(1, self.random_list[self.depth+(self.gate_set*d)+2]), RY(1, self.random_list[self.depth+(self.gate_set*d)+3])))
                circuit.add_gate(
                    RZ(0, self.random_list[self.depth + (self.gate_set * d)])
                )
                circuit.add_gate(
                    RZ(1, self.random_list[self.depth + (self.gate_set * d) + 1])
                )
                circuit.add_gate(
                    self.create_hamiltonian_gate(
                        [1] * self.n_qubit,
                        [self.bn] * self.n_qubit,
                        self.random_list[d],
                    )
                )

        return circuit


class AnsatzIndirectByXYZ:
    def __init__(self, n_qubit, random_list, depth):
        self.n_qubit = n_qubit
        self.random_list = random_list
        self.depth = depth
        self.gate_set = 2

    def create_hamiltonian_gate(self, cn, t):
        I_gate = [[1, 0], [0, 1]]
        X_gate = [[0, 1], [1, 0]]
        Y_gate = [[0, 0 - 1j], [0 + 1j, 0]]
        Z_gate = [[1, 0], [0, -1]]

        XX = np.array(np.zeros(2**self.n_qubit))
        YY = np.array(np.zeros(2**self.n_qubit))
        ZZ = np.array(np.zeros(2**self.n_qubit))
        for k in range(self.n_qubit - 1):
            for l in range(self.n_qubit):
                if k == l:
                    if l == 0:
                        hamiX = X_gate
                        hamiY = Y_gate
                        hamiZ = Z_gate
                    else:
                        hamiX = np.kron(hamiX, X_gate)
                        hamiY = np.kron(hamiY, Y_gate)
                        hamiZ = np.kron(hamiZ, Z_gate)

                elif k + 1 == l:
                    hamiX = np.kron(hamiX, X_gate)
                    hamiY = np.kron(hamiY, Y_gate)
                    hamiZ = np.kron(hamiZ, Z_gate)
                else:
                    if l == 0:
                        hamiX = I_gate
                        hamiY = I_gate
                        hamiZ = I_gate
                    else:
                        hamiX = np.kron(hamiX, I_gate)
                        hamiY = np.kron(hamiY, I_gate)
                        hamiZ = np.kron(hamiZ, I_gate)
            XX = XX + 0.5 * cn[k] * hamiX
            YY = YY + 0.5 * cn[k] * hamiY
            ZZ = ZZ + 0.5 * cn[k] * hamiZ

        hamiltonian = XX + YY + ZZ
        hamiltonian_sparse = sparse.csc_matrix(hamiltonian)
        np.set_printoptions(threshold=10000)
        hami_gate = expm(-1j * hamiltonian_sparse * t)
        return DenseMatrix(list(range(self.n_qubit)), hami_gate.toarray())

    def create_ansatz(self):
        circuit = QuantumCircuit(self.n_qubit)
        for d in range(self.depth):
            circuit.add_gate(
                merge(
                    RX(0, self.random_list[self.depth + (self.gate_set * d)]),
                    RY(0, self.random_list[self.depth + (self.gate_set * d) + 1]),
                )
            )
            # TODO: cnをrandom化する
            circuit.add_gate(
                self.create_hamiltonian_gate([1] * self.n_qubit, self.random_list[d])
            )

        return circuit


class AnsatzIndirectByXY:
    def __init__(self, n_qubit, random_list, depth, is_bn_random):
        self.n_qubit = n_qubit
        self.random_list = random_list
        self.depth = depth
        self.gate_set = 4
        self.is_bn_random = is_bn_random
        self.bn = random.uniform(-1.0, 1.0)

    def create_hamiltonian_gate(self, cn, gamma, bn, t):
        I_gate = [[1, 0], [0, 1]]
        X_gate = [[0, 1], [1, 0]]
        Y_gate = [[0, 0 - 1j], [0 + 1j, 0]]
        Z_gate = [[1, 0], [0, -1]]

        XX = np.array(np.zeros(2**self.n_qubit))
        YY = np.array(np.zeros(2**self.n_qubit))
        Zn = np.array(np.zeros(2**self.n_qubit))
        for k in range(self.n_qubit - 1):
            for l in range(self.n_qubit):
                if k == l:
                    if l == 0:
                        hamiX = X_gate
                        hamiY = Y_gate
                    else:
                        hamiX = np.kron(hamiX, X_gate)
                        hamiY = np.kron(hamiY, Y_gate)

                elif k + 1 == l:
                    hamiX = np.kron(hamiX, X_gate)
                    hamiY = np.kron(hamiY, Y_gate)
                else:
                    if l == 0:
                        hamiX = I_gate
                        hamiY = I_gate
                    else:
                        hamiX = np.kron(hamiX, I_gate)
                        hamiY = np.kron(hamiY, I_gate)
            XX = XX + 0.5 * cn[k] * (1 + gamma) * hamiX
            YY = YY + 0.5 * cn[k] * (1 - gamma) * hamiY

        for m in range(self.n_qubit):
            for n in range(self.n_qubit):
                if m == n:
                    if n == 0:
                        hamiZ = Z_gate
                    else:
                        hamiZ = np.kron(hamiZ, Z_gate)

                else:
                    if n == 0:
                        hamiZ = I_gate
                    else:
                        hamiZ = np.kron(hamiZ, I_gate)

            Zn = Zn + bn[m] * hamiZ

        hamiltonian = XX + YY + Zn
        hamiltonian_sparse = sparse.csc_matrix(hamiltonian)
        np.set_printoptions(threshold=10000)
        hami_gate = expm(-1j * hamiltonian_sparse * t)
        return DenseMatrix(list(range(self.n_qubit)), hami_gate.toarray())

    def create_ansatz(self):
        """
        ansatzを作成する
        ハミルトニアンゲートはdepthごとに別々のパラメータとなる.
        bnについてはdepth x n_qubitの配列をパラメータとして渡す.
        TODO: r, cnのrandomizeは未実装
        """
        circuit = QuantumCircuit(self.n_qubit)
        for d in range(self.depth):
            circuit.add_gate(CNOT(0, 1))
            if self.is_bn_random:
                circuit.add_gate(
                    merge(
                        RY(
                            0,
                            self.random_list[
                                self.depth
                                + (self.depth * self.n_qubit)
                                + (self.gate_set * d)
                            ],
                        ),
                        RZ(
                            0,
                            self.random_list[
                                self.depth
                                + (self.depth * self.n_qubit)
                                + (self.gate_set * d)
                                + 1
                            ],
                        ),
                    )
                )
                circuit.add_gate(
                    merge(
                        RY(
                            1,
                            self.random_list[
                                self.depth
                                + (self.depth * self.n_qubit)
                                + (self.gate_set * d)
                                + 2
                            ],
                        ),
                        RZ(
                            1,
                            self.random_list[
                                self.depth
                                + (self.depth * self.n_qubit)
                                + (self.gate_set * d)
                                + 3
                            ],
                        ),
                    )
                )
                circuit.add_gate(
                    self.create_hamiltonian_gate(
                        [1] * self.n_qubit,
                        0,
                        self.random_list[
                            d + self.depth : d + self.depth + self.n_qubit
                        ],
                        self.random_list[d],
                    )
                )
            else:
                circuit.add_gate(
                    merge(
                        RY(0, self.random_list[self.depth + (self.gate_set * d)]),
                        RZ(0, self.random_list[self.depth + (self.gate_set * d) + 1]),
                    )
                )
                circuit.add_gate(
                    merge(
                        RY(1, self.random_list[self.depth + (self.gate_set * d) + 2]),
                        RZ(1, self.random_list[self.depth + (self.gate_set * d) + 3]),
                    )
                )
                # circuit.add_gate(self.create_hamiltonian_gate([1]*self.n_qubit, 0, self.bn, self.random_list[d]))
                circuit.add_gate(
                    self.create_hamiltonian_gate(
                        [1] * self.n_qubit,
                        0,
                        [self.bn] * self.n_qubit,
                        self.random_list[d],
                    )
                )

        return circuit


class AnsatzIndirectByNone:
    def __init__(self, n_qubit, random_list, depth):
        self.n_qubit = n_qubit
        self.random_list = random_list
        self.depth = depth
        self.gate_set = 4

    def create_ansatz(self):
        circuit = QuantumCircuit(self.n_qubit)
        for d in range(self.depth):
            circuit.add_gate(CNOT(0, 1))
            circuit.add_gate(
                merge(
                    RY(0, self.random_list[self.gate_set * d + self.depth]),
                    RZ(0, self.random_list[self.gate_set * d + (self.depth + 1)]),
                )
            )
            circuit.add_gate(
                merge(
                    RY(1, self.random_list[self.gate_set * d + (self.depth + 2)]),
                    RZ(1, self.random_list[self.gate_set * d + (self.depth + 3)]),
                )
            )

        return circuit
