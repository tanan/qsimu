from abc import ABCMeta

from abc import ABCMeta, abstractmethod
import numpy as np
from qulacs.gate import DenseMatrix


class Ansatz(metaclass=ABCMeta):
    def __init__(
        self, nqubit, depth, noise, gate_set=None, time=None, bn=None, gamma=None
    ):
        self.nqubit = nqubit
        self.depth = depth
        self.noise = noise
        self.gate_set = gate_set
        self.bn = bn
        self.time = time
        self.diag, self.eigen_vecs = self.create_hamiltonian(
            [1] * self.nqubit, bn, gamma
        )

    @abstractmethod
    def create_hamiltonian(self, cn, bn=None, gamma=None):
        pass

    def create_hamiltonian_gate(self, t):
        time_evol_op = np.dot(
            np.dot(self.eigen_vecs, np.diag(np.exp(-1j * t * self.diag))),
            self.eigen_vecs.T.conj(),
        )
        return DenseMatrix([i for i in range(self.nqubit)], time_evol_op)

    @abstractmethod
    def create_ansatz(self, random_list):
        pass


class ParametricGateCountError(Exception):
    pass
