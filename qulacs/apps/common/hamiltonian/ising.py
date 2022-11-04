import numpy as np
from common.ansatz.pauli_gate import PauliGate
from common.hamiltonian import Coefficients


def create_ising_hamiltonian(nqubit: int, coef: Coefficients) -> np.ndarray:
    XX = np.array(np.zeros(2**nqubit))
    Y = np.array(np.zeros(2**nqubit))
    for k in range(nqubit - 1):
        for l in range(nqubit):
            if k == l:
                if l == 0:
                    hamiX = PauliGate.X_gate.value
                else:
                    hamiX = np.kron(hamiX, PauliGate.X_gate.value)

            elif k + 1 == l:
                hamiX = np.kron(hamiX, PauliGate.X_gate.value)
            else:
                if l == 0:
                    hamiX = PauliGate.I_gate.value
                else:
                    hamiX = np.kron(hamiX, PauliGate.I_gate.value)
        XX = XX + coef[0][k] * hamiX

    for m in range(nqubit):
        for n in range(nqubit):
            if m == n:
                if n == 0:
                    hamiY = PauliGate.Y_gate.value
                else:
                    hamiY = np.kron(hamiY, PauliGate.Y_gate.value)

            else:
                if n == 0:
                    hamiY = PauliGate.I_gate.value
                else:
                    hamiY = np.kron(hamiY, PauliGate.I_gate.value)

        Y = Y + coef[1][m] * hamiY

    return XX + Y
