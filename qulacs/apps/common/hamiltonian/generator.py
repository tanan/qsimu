from functools import cached_property
from typing import Iterable, Tuple
from qulacs import QuantumCircuit
from qulacs.gate import DenseMatrix

import numpy as np

from common.hamiltonian import Coefficients, Hamiltonian, HamiltonianModel
from common.hamiltonian.ising import create_ising_hamiltonian

class _Hamiltonian:
    def __init__(self, nqubit: int, coef: Coefficients, model: HamiltonianModel) -> None:
        self.nqubit = nqubit
        self.coef = coef
        self.model = model
        pass

    @cached_property
    def value(self) -> np.ndarray:
        if self.model == HamiltonianModel.TRANSVERSE_ISING:
            return create_ising_hamiltonian(self.nqubit, self.coef)
        else:
            pass
        return

    @cached_property
    def eigh(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.linalg.eigh(self.value)
    
    def circuit(self, t) -> QuantumCircuit:
        circuit = QuantumCircuit(self.nqubit)
        time_evol_op = np.dot(
            np.dot(self.eigh[1], np.diag(np.exp(-1j * t * self.eigh[0]))),
            self.eigh[1].T.conj(),
        )
        circuit.add_gate(DenseMatrix([i for i in range(self.nqubit)], time_evol_op))
        return circuit


def create_transverse_ising_hamiltonian_generator(
    nqubit: int,
    coef_vertical: Iterable[float],
    coef_transverse: Iterable[float],
    model: HamiltonianModel,
) -> Hamiltonian:
    return _Hamiltonian(nqubit, (coef_vertical, coef_transverse), model)