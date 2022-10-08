from functools import cached_property
from typing import Iterable, List
import numpy as np

from common.estimator import Observables
from common.hamiltonian import (
    Coefficients,
    Hamiltonian,
    HamiltonianModel,
    HamiltonianModelError,
)
from qulacs import QuantumState, Observable

from common.hamiltonian.generator import create_transverse_ising_hamiltonian_generator


class _HamiltonianEstimate:
    def __init__(
        self, nqubit: int, obs: Observables, hamiltonian: Hamiltonian, xs: np.ndarray
    ) -> Hamiltonian:
        self.nqubit = nqubit
        self.obs = obs
        self.hamiltonian = hamiltonian
        self.xs = xs

    @cached_property
    def value(self) -> np.ndarray:
        y_train = []
        for x in self.xs:
            y = []
            if type(self.obs) is not list:
                observables = [self.obs]
                self.obs = observables

            for obs in self.obs:
                state = QuantumState(self.nqubit)
                state.set_zero_state()
                self.hamiltonian.circuit(300 + (4 * (x + 1))).update_quantum_state(
                    state
                )
                y.append(obs.get_expectation_value(state))
            y_train.append(y)

        return np.array(y_train)


def create_hamiltonian_estimator(
    nqubit: int,
    obs: Observables,
    coef: Coefficients,
    hamiltonianModel: HamiltonianModel,
    xs,
) -> Hamiltonian:
    if hamiltonianModel == HamiltonianModel.TRANSVERSE_ISING:
        hamiltonian = create_transverse_ising_hamiltonian_generator(
            nqubit, coef[0], coef[1], hamiltonianModel
        )
    else:
        raise HamiltonianModelError("cannot find a valid hamiltonian model.")

    return _HamiltonianEstimate(nqubit, obs, hamiltonian, xs)
